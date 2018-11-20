import itertools
import os.path
import tempfile
from os import remove
from os.path import join, basename

import cv2
import imageio
import numpy as np
import pandas as pd
import scipy
import yaml
from graph_tool import Graph
from joblib import Memory
from joblib import Parallel, delayed
from keras.applications.mobilenet import mobilenet
from keras.models import model_from_yaml, model_from_json
from tqdm import tqdm

from core.interactions.generate_data import DataGenerator
from core.interactions.train import TrainInteractions
from core.interactions.visualization import save_prediction_img, show_prediction
from core.region.transformableregion import TransformableRegion
from utils.img import safe_crop
from utils.objectsarray import ObjectsArray
from utils.gt.mot import load_mot, results_to_mot, eval_mot, mot_in_roi
from core.region.ellipse import Ellipse
from utils.roi import ROI

memory = Memory('out/cache', verbose=0)


def get_hull_poly(region, epsilon_px=5):
    poly = cv2.approxPolyDP(np.fliplr(region.contour()), epsilon_px, True)
    hull = cv2.convexHull(poly.reshape((-1, 2))).reshape((-1, 2))
    return np.vstack((hull, hull[0:1]))


class InteractionDetector:
    def __init__(self, model_dir, project=None):
        """
        Load interaction detector keras model.

        :param model_dir: directory with model.{yaml or json}, weights.h5 and config.yaml
        """
        self.TRACKING_COST_WEIGHT = 0.8
        self.TRACKING_CONFIDENCE_LOC = 20
        self.TRACKING_CONFIDENCE_SCALE = 0.2
        keras_custom_objects = {
            'relu6': mobilenet.relu6,
        }
        with open(join(model_dir, 'config.yaml'), 'r') as fr:
            self.config = yaml.load(fr)
        self.ti = TrainInteractions(self.config['num_objects'], detector_input_size_px=self.config['input_size_px'])  # TODO: remove dependency

        if os.path.exists(join(model_dir, 'model.yaml')):
            with open(join(model_dir, 'model.yaml'), 'r') as fr:
                self.m = model_from_yaml(fr.read(), custom_objects=keras_custom_objects)
        elif os.path.exists(join(model_dir, 'model.json')):
            with open(join(model_dir, 'model.json'), 'r') as fr:
                self.m = model_from_json(fr.read(), custom_objects=keras_custom_objects)
        else:
            # temporary handling of obsolete models
            self.m = self.ti.model_6conv_3dense_legacy()
            self.config['properties'] = ['x', 'y', 'major', 'minor', 'angle_deg']
            # assert False, '{} or {} not found'.format(join(model_dir, 'model.yaml'), join(model_dir, 'model.json'))
        self.m.load_weights(join(model_dir, 'weights.h5'))
        self.predictions = ObjectsArray(self.config['properties'], self.config['num_objects'])
        self.project = project
        # self._detect_single_frame = memory.cache(self._detect_single_frame, ignore=['self'])  # uncomment to enable caching

    def _add_tracklet_to_graph(self, t, graph, vertices):
        for t2 in self.project.gm.get_incoming_tracklets(t.start_vertex(self.project.gm)):
            if t2.is_multi() or t2.is_single():
                if t2.id() not in vertices:
                    v = graph.add_vertex()
                    vertices[t2.id()] = v
                    graph.vertex_properties['tracklet'][v] = t2
                    graph.vertex_properties['text'][v] = t2.id()
                    graph.vertex_properties['category'][v] = 'in' if t2.is_single() else 'multi'
                elif graph.vertex_properties['category'][vertices[t2.id()]] == 'out':
                    graph.vertex_properties['category'][vertices[t2.id()]] = 'in_out'
                if not graph.edge(vertices[t2.id()], vertices[t.id()]):
                    graph.add_edge(vertices[t2.id()], vertices[t.id()])
                    if t2.is_multi():
                        self._add_tracklet_to_graph(t2, graph, vertices)
            else:
                print(t2)

        for t2 in self.project.gm.get_outcoming_tracklets(t.end_vertex(self.project.gm)):
            if t2.is_multi() or t2.is_single():
                if t2.id() not in vertices:
                    v = graph.add_vertex()
                    vertices[t2.id()] = v
                    graph.vertex_properties['tracklet'][v] = t2
                    graph.vertex_properties['text'][v] = t2.id()
                    graph.vertex_properties['category'][v] = 'out' if t2.is_single() else 'multi'
                elif graph.vertex_properties['category'][vertices[t2.id()]] == 'in':
                    graph.vertex_properties['category'][vertices[t2.id()]] = 'in_out'
                if not graph.edge(vertices[t.id()], vertices[t2.id()]):
                    graph.add_edge(vertices[t.id()], vertices[t2.id()])
                    if t2.is_multi():
                        self._add_tracklet_to_graph(t2, graph, vertices)
            else:
                print(t2)

    def _find_dense_subgraph(self, initial_tracklet):
        interaction_graph = Graph()
        interaction_graph.vertex_properties['tracklet'] = \
            interaction_graph.new_vertex_property('object')
        interaction_graph.vertex_properties['text'] = \
            interaction_graph.new_vertex_property('string')
        interaction_graph.vertex_properties['category'] = \
            interaction_graph.new_vertex_property('string')
        v = interaction_graph.add_vertex()
        interaction_graph.vertex_properties['tracklet'][v] = initial_tracklet
        interaction_graph.vertex_properties['text'][v] = initial_tracklet.id()
        interaction_graph.vertex_properties['category'][v] = 'multi'
        vertices = {initial_tracklet.id(): v}
        self._add_tracklet_to_graph(initial_tracklet, interaction_graph, vertices)
        return interaction_graph, vertices

    def find_dense_subgraphs(self):
        """
        Find dense situations in a tracking graph.

        Dense situation is a continuous series of multi tracklets with neighbouring incoming and outcoming tracklets.

        :return: list of dictionaries with 'graph' (dense situation Graph) and
                                           'ids' entries (tracklet id -> Vertex mapping)
        """
        multi = [t for t in self.project.chm.chunk_gen() if t.is_multi()]
        # multi = sorted(multi, key=len, reverse=True)
        # multi = sorted(multi, key=lambda x: x.get_cardinality(project.gm), reverse=True)
        dense_subgraphs = []
        while multi:
            graph, ids = self._find_dense_subgraph(multi[-1])
            dense_subgraphs.append({
                'graph': graph,
                'ids': ids,
            })
            for v in graph.vertices():
                if graph.vp.tracklet[v] in multi:
                    multi.remove(graph.vp.tracklet[v])
        return dense_subgraphs

    def draw_graph(self, graph):
        from graph_tool.draw import graph_draw, sfdp_layout
        fill_color = graph.new_vertex_property('string')
        category_prop = graph.vertex_properties['category']
        for v in graph.vertices():
            if category_prop[v] == 'multi':
                fill_color[v] = 'red'
            elif category_prop[v] == 'in':
                fill_color[v] = 'white'
            elif category_prop[v] == 'out':
                fill_color[v] = 'black'
            elif category_prop[v] == 'in_out':
                fill_color[v] = 'gray'
            else:
                assert False

        graph_draw(graph, pos=sfdp_layout(graph), vertex_fill_color=fill_color,
                      # inline=True,
                      vertex_text=graph.vertex_properties['text'],
                      edge_marker_size=10,
                      vertex_size=10,
                      # output_size=(1200, 600),
                      output_size=(2800, 2800),
                      # output='dense_subgraph.svg'
                   )

    def track_single_object(self, starting_region, multi_tracklet=None, forward=True, max_frames=np.inf):
        prediction = starting_region.to_dict()
        if multi_tracklet is not None:
            start_frame = multi_tracklet.start_frame(self.project.gm)
            end_frame = multi_tracklet.end_frame(self.project.gm)
        else:
            start_frame = starting_region.frame
            assert not np.isinf(max_frames)
            end_frame = starting_region.frame + max_frames
        if forward:
            frame = start_frame
        else:
            frame = end_frame

        i = 0
        regions = []
        while (start_frame <= frame <= end_frame) and i <= max_frames:  # overlap > 0 and
            prev_prediction = prediction.copy()

            prediction, _, _, _ = self.detect_single_frame(frame, prev_prediction)
            prediction_r = Ellipse.from_dict(prediction)
            prediction_r.frame = frame

            if multi_tracklet is not None:
                multi_region = multi_tracklet.get_region_in_frame(self.project.gm, frame)
                # img = self.project.img_manager.get_whole_img(frame)
                # multi_poly = get_hull_poly(multi_region)
                # cv2.polylines(img, [prediction_r.to_poly()], True, (255, 0, 0), 1)
                # cv2.polylines(img, [multi_poly], True, (0, 255, 0))
                # # save_prediction_img(None, 1, img, pred=prediction, gt=None, scale=1.1)
                # images.append(safe_crop(img, tuple([prediction[x] for x in ('0_x', '0_y')]), 200)[0])
                overlap = prediction_r.get_overlap(get_hull_poly(multi_region))
                if float(overlap) / prediction_r.area < 0.25:
                    # print('overlap bellow threshold')
                    break
            regions.append(prediction_r)
            i += 1
            if forward is True:
                frame += 1
            else:
                frame -= 1

        if forward:
            last_frame = frame - 1
        else:
            last_frame = frame + 1
        if forward and (last_frame == end_frame) or \
           not forward and (last_frame == start_frame):
            success = True
        else:
            success = False
        return regions, last_frame, success

    @staticmethod
    def get_tracklets_from_dense(graph):
        """
        Get incoming, outcoming and multi tracklets from a dense tracklets graph.

        :param graph: dense graph; Graph with vertex properties 'category' and 'tracklet'
        :return: incoming, outcoming, multi; lists of tracklets (Chunk)
        """
        category_prop = graph.vertex_properties['category']
        tracklet_prop = graph.vertex_properties['tracklet']
        incoming = [tracklet_prop[v] for v in graph.vertices() if
                          (category_prop[v] == 'in') or (category_prop[v] == 'in_out')]
        outcoming = [tracklet_prop[v] for v in graph.vertices() if
                           (category_prop[v] == 'out') or (category_prop[v] == 'in_out')]
        multi = [tracklet_prop[v] for v in graph.vertices() if category_prop[v] == 'multi']
        return incoming, outcoming, multi

    def track_single_object_in_dense_subgraph(self, graph, vertices, single_tracklet, forward=True):

        def track_recursive(vertex, graph, regions_path, processed_tracklets, forward=True):
            if forward:
                # last_region_idx = -1
                neighbors = vertex.out_neighbors()
            else:
                # last_region_idx = 0
                neighbors = vertex.in_neighbors()
            results = []
            # tracklet = graph.vp.tracklet[vertex]
            results_tracklets = []
            for v in neighbors:
                if graph.vp.category[v] == 'out' or graph.vp.category[v] == 'in_out':
                    pass
                elif graph.vp.category[v] == 'multi' and graph.vp.tracklet[v] not in processed_tracklets:
                    multi_tracklet = graph.vp.tracklet[v]
                    if multi_tracklet in processed_tracklets:
                        #TODO: remove if not needed
                        print('Skipping cycle: multi tracklet id: {}'.format(multi_tracklet.id()))
                        continue
                    # last_region = Ellipse.from_region(tracklet.get_region(self.project.gm, last_region_idx))
                    regions, last_frame, success = \
                        self.track_single_object(regions_path[-1], multi_tracklet, forward)
                    multi_tracklets = processed_tracklets[:] + [multi_tracklet,]
                    if success:
                        track_recursive(v, graph, regions, multi_tracklets, forward)
                    # print('---')
                    results.append(regions)
                    results_tracklets.append(multi_tracklets)
                elif graph.vp.category[v] == 'in' or graph.vp.category[v] == 'in_out':
                    pass
                else:
                    assert False
            if results:
                path_lengths = [len(result) for result in results]
                best_idx = path_lengths.index(max(path_lengths))
                regions_path.extend(results[best_idx])
                processed_tracklets.extend(results_tracklets[best_idx])

        v = vertices[single_tracklet.id()]
        if forward:
            adjacent_region_idx = -1
        else:
            adjacent_region_idx = 0

        regions_path = [Ellipse.from_region(single_tracklet.get_region(self.project.gm, adjacent_region_idx))]
        processed_tracklets = []
        track_recursive(v, graph, regions_path, processed_tracklets, forward)
        regions_path = regions_path[1:]  # remove adjacent single_tracklet region
        if not forward:
            regions_path = sorted(regions_path, key=lambda x: x.frame)
        return regions_path

    def region_to_dict(self, r):
        return({
            '0_x': r.centroid()[1],
            '0_y': r.centroid()[0],
            '0_major': 2 * r.major_axis_,
            '0_minor': 2 * r.minor_axis_,
            '0_angle_deg_cw': -np.rad2deg(r.theta_),
        })

    def _detect_single_frame(self, video_file, frame, prev_detection):
        """
        Detect object in single frame.

        Function signature suitable for caching. See __init__.

        :param video_file:
        :param frame:
        :param prev_detection:
        :return:
        """
        img = self.project.img_manager.get_whole_img(frame)
        pred_dict, pred_, delta_xy, img_crop = self.detect_single(img, prev_detection)
        return pred_dict, pred_, delta_xy, img_crop

    def detect_single_frame(self, frame, prev_detection):
        return self._detect_single_frame(basename(self.project.video_paths), frame, prev_detection)

    def detect_single(self, img, prev_detection):
        """
        Detect objects in interaction.

        :param img: input image
        :param xy: position of the interaction
        :return: dict, properties of the detected objects
                       keys: '0_x', '0_y', '0_angle_deg', .... 'n_x', 'n_y', 'n_angle_deg'
        """
        timg = TransformableRegion()
        prev_xy = (prev_detection['0_x'], prev_detection['0_y'])
        timg.rotate(-prev_detection['0_angle_deg_cw'], prev_xy[::-1])
        timg.set_img(img)
        img_rotated = timg.get_img()

        img_crop, delta_xy = safe_crop(img_rotated, prev_xy, self.config['input_size_px'])
        img = np.expand_dims(img_crop, 0).astype(np.float) / 255.
        pred = self.m.predict(img)
        pred_ = pred.copy().flatten()
        pred = self.ti.postprocess_predictions(pred)

        pred_dict = self.predictions.array_to_dict(pred)
        if '0_angle_deg_cw' not in pred_dict:
            pred_dict['0_angle_deg_cw'] = pred_dict['0_angle_deg']  # backwards compatibility
        pred_dict['0_x'] += delta_xy[0]
        pred_dict['0_y'] += delta_xy[1]
        pred_dict['0_major'] = prev_detection['0_major']
        pred_dict['0_minor'] = prev_detection['0_minor']
        pred_dict['0_angle_deg_cw'] += prev_detection['0_angle_deg_cw']
        # pred_dict['0_angle_deg'] = -timg.get_inverse_transformed_angle(pred_dict['0_angle_deg'])
        pred_dict['0_x'], pred_dict['0_y'] = timg.get_inverse_transformed_coords(
            np.array((pred_dict['0_x'], pred_dict['0_y'])))

        return pred_dict, pred_, delta_xy, img_crop

    def draw_detections(self, img, detections):
        ax = show_prediction(img, self.config['num_objects'], detections)
        return ax

    def write_interaction_movie(self, images, tracks, out_filename):
        _, tmp_file = tempfile.mkstemp(suffix='.png')
        out_images = []
        if isinstance(tracks, pd.DataFrame):
            tracks_items = [t for _, t in tracks.iterrows()]
        elif isinstance(tracks, list):
            tracks_items = tracks
        for img, props in tqdm(zip(images, tracks_items), desc='interaction movie'):
    #         for obj_i in range(2):
    #             predictions['{}_major'.format(obj_i)] = 60
    #             predictions['{}_minor'.format(obj_i)] = 15
            save_prediction_img(tmp_file, 2, img, pred=props, scale=0.8)
            out_images.append(imageio.imread(tmp_file))

        imageio.mimwrite(out_filename, out_images)
        os.unlink(tmp_file)

    def get_bounds(self, graph):
        rois = []
        for v in graph.vertices():
            t = graph.vp.tracklet[v]
            if graph.vp.category[v] == 'multi':
                rois.extend([r.roi() for r in t.r_gen(self.project.gm, self.project.rm)])
            elif graph.vp.category[v] == 'in':
                rois.append(t.get_region(self.project.gm, -1).roi())
            elif graph.vp.category[v] == 'out':
                rois.append(t.get_region(self.project.gm, 0).roi())
            elif graph.vp.category[v] == 'in_out':
                rois.append(t.get_region(self.project.gm, 0).roi())
                rois.append(t.get_region(self.project.gm, -1).roi())
            else:
                assert False
        roi_union = reduce(lambda x, y: x.union(y), rois)
        incoming, outcoming, multi = self.get_tracklets_from_dense(graph)
        try:
            start_frame = min([t.end_frame(self.project.gm) for t in incoming])
        except ValueError:
            # dense sections without incoming single tracklets - on the start of the video sequence
            start_frame = min([t.end_frame(self.project.gm) for t in multi])
        try:
            end_frame = max([t.start_frame(self.project.gm) for t in outcoming])
        except ValueError:
            # dense sections without outcoming single tracklets - on the end of the video sequence
            end_frame = max([t.start_frame(self.project.gm) for t in multi])
        return roi_union, (start_frame, end_frame)

    def visualize_single_tracklet(self, regions, gt, out_dir):
        try:
            os.makedirs(out_dir)
        except OSError:
            pass
        vm = self.project.get_video_manager()
        rois = [ROI(r.y - r.major / 2, r.x - r.major / 2, r.major, r.major) for r in regions]
        roi_union = reduce(lambda x, y: x.union(y), rois)
        start_frame = regions[0].frame
        end_frame = regions[-1].frame
        imgs = [vm.get_frame(frame) for frame in tqdm(range(start_frame, end_frame + 1), desc='gathering images')]
        Parallel(n_jobs=-1, verbose=10)(delayed(plot_frame)(frame, i, imgs[i], out_dir,
                                                           paths=[{'regions': regions}], roi=roi_union, gt=gt)
                                       for i, frame in enumerate(range(start_frame, end_frame + 1)))

    def visualize_tracklets(self, graph, paths, out_dir, gt=None):
        vm = self.project.get_video_manager()
        roi, (start_frame, end_frame) = self.get_bounds(graph)
        incoming, outcoming, _ = self.get_tracklets_from_dense(graph)

        try:
            os.makedirs(out_dir)
        except OSError:
            pass

        imgs = [vm.get_frame(frame) for frame in tqdm(range(start_frame, end_frame + 1), desc='gathering images')]
        regions = []
        for frame in tqdm(range(start_frame, end_frame + 1), desc='gathering regions'):
            regions_in_frame = []
            for t in self.project.chm.tracklets_in_frame(frame):
                r = t.get_region_in_frame(self.project.gm, frame)
                yx = r.contour_without_holes()
                if t.is_single():
                    color = 'white'
                elif t.is_multi():
                    color = 'gray'
                else:
                    color = 'red'
                regions_in_frame.append({'region': r, 'color': color, 'contour_yx': yx})
            regions.append(regions_in_frame)

        Parallel(n_jobs=1, verbose=10)(delayed(plot_frame)(frame, i, imgs[i], out_dir, regions[i], paths, roi, gt)
                                       for i, frame in enumerate(range(start_frame, end_frame + 1)))




        # for i, frame in tqdm(enumerate(range(start_frame, end_frame + 1))):
        #     plot_dense_frame(frame, i, imgs[i], regions[i], out_dir, paths, roi)

    def track_dense(self, graph, ids):

        def min_dist(p1, p2):
            p1 = {el.frame: el for el in p1}
            p2 = {el.frame: el for el in p2}
            frames = sorted(list(set(p1.keys()).intersection(p2.keys())))
            if not frames:
                return 99999, -1
            else:
                dists = np.linalg.norm(
                    np.array([p1[frame].xy for frame in frames]) - np.array([p2[frame].xy for frame in frames]), axis=1)
                i = np.argmin(dists)
                return dists[i], frames[i]

        incoming, outcoming, _ = self.get_tracklets_from_dense(graph)
        fwd = []
        for t in tqdm(incoming, desc='incoming'):
            regions_path = self.track_single_object_in_dense_subgraph(graph, ids, t, forward=True)
            fwd.append({'regions': regions_path,
                        'in_tracklet': t,
                        'in_region': t.get_region(self.project.gm, -1)})
        bwd = []
        for t in tqdm(outcoming, desc='outcoming'):
            regions_path = self.track_single_object_in_dense_subgraph(graph, ids, t, forward=False)
            bwd.append({'regions': regions_path,
                        'out_tracklet': t,
                        'out_region': t.get_region(self.project.gm, 0)})

        if len(bwd) != 0 and len(fwd) != 0:
            # np.fromfunction(np.vectorize(lambda i, j: min_dist(fwd[i[0]], bwd[j[1]])), (len(fwd), len(bwd)))
            cost_matrix, frames_matrix = np.vectorize(lambda i, j: min_dist(fwd[i]['regions'], bwd[j]['regions']))(
                *np.indices((len(fwd), len(bwd))))
            ii, jj = scipy.optimize.linear_sum_assignment(cost_matrix)
        else:
            frames_matrix = -np.ones((max(len(fwd), 1), max(len(bwd), 1)))
            ii = np.arange(len(fwd)) if fwd else np.zeros(len(bwd), dtype=int)
            jj = np.arange(len(bwd)) if bwd else np.zeros(len(fwd), dtype=int)

        paths = []
        for i, j in zip(ii, jj):
            frame = frames_matrix[i, j]
            if frame == -1:
                # no overlap between tracklets
                try:
                    if fwd[i]['regions']:
                        print('no overlap between tracklets: adding fwd part')
                        paths.append({
                            'regions': fwd[i]['regions'],
                            'in_tracklet': fwd[i]['in_tracklet'],
                            'in_region': fwd[i]['in_region'],
                            'out_tracklet': None,
                            'out_region': None, })
                except IndexError:
                    pass
                try:
                    if bwd[j]['regions']:
                        print('no overlap between tracklets: adding bwd part')
                        paths.append({
                            'regions': bwd[j]['regions'],
                            'in_tracklet': None,
                            'in_region': None,
                            'out_tracklet': bwd[j]['out_tracklet'],
                            'out_region': bwd[j]['out_region'], })
                except IndexError:
                    pass
            else:
                # join tracklets where the cost is minimal
                mean_major_px = np.concatenate(([el.major for el in fwd[i]['regions']],
                                                [el.major for el in bwd[j]['regions']])).mean()
                if cost_matrix[i, j] > 0.5 * mean_major_px:
                    # TODO, add confident parts of fwd, bwd
                    import warnings
                    warnings.warn(
                        'TODO: fwd bwd distance high: distance {} px, threshold {} px'.format(cost_matrix[i, j],
                                                                                              0.5 * mean_major_px))
                fwd_idx = [el.frame for el in fwd[i]['regions']].index(frame)
                bwd_idx = [el.frame for el in bwd[j]['regions']].index(frame)
                assert fwd[i]['regions'][fwd_idx].frame == bwd[j]['regions'][bwd_idx].frame
                mean_el = fwd[i]['regions'][fwd_idx] + bwd[j]['regions'][bwd_idx] # mean region
                mean_el.angle_deg = fwd[i]['regions'][fwd_idx].angle_deg
                paths.append({
                    'regions': fwd[i]['regions'][:fwd_idx] + [mean_el] + bwd[j]['regions'][(bwd_idx + 1):],
                    'in_tracklet': fwd[i]['in_tracklet'],
                    'in_region': fwd[i]['in_region'],
                    'out_tracklet': bwd[j]['out_tracklet'],
                    'out_region': bwd[j]['out_region'], })
        return paths

    def eval_dense_section(self, graph, paths):
        # (n_frames, n_animals, 2); coordinates are in yx order
        roi, (start_frame, end_frame) = self.get_bounds(graph)
        # first_frame = min([path['regions'][0].frame for path in paths])
        # last_frame = max([path['regions'][-1].frame for path in paths])
        results = np.ones(shape=(end_frame + 1, len(paths), 2)) * np.nan
        for idx_path, path in enumerate(paths):
            for region in path['regions']:
                results[region.frame, idx_path] = region.xy[::-1]

        df_results = results_to_mot(results).set_index(['frame', 'id'])
        df_gt = mot_in_roi(gt, roi).loc[start_frame + 1:end_frame - 1]

        return eval_mot(df_gt, df_results)


def plot_frame(frame, i, img, out_dir, regions_in_frame=None, paths=None, roi=None, gt=None):
    from scripts.montage import save_figure_as_image
    from core.interactions.visualization import plot_interaction
    import matplotlib.pylab as plt
    fig = plt.figure()
    plt.imshow(img)
    colors = itertools.cycle(['red', 'blue', 'green', 'yellow', 'white'])
    # for path_set, color in zip(paths, ('r', 'b')):
    #                for regions in path_set:

    if paths is not None:
        for path, color in zip(paths, colors):
            regions = path['regions']
            # plt.plot([r.x for r in regions], [r.y for r in regions], color=color)
            frames = [r.frame for r in regions]
            try:
                r = regions[frames.index(frame)]
                plot_interaction(1, pred=r.to_dict(), color=color)
                plt.plot(r.x, r.y, 'o', color=color)
            except ValueError:
                pass
            if 'in_region' in path and path['in_region'] is not None and frame == path['in_region'].frame():
                plot_interaction(1, gt=Ellipse.from_region(path['in_region']).to_dict(), color=color)
            if 'out_region' in path and path['out_region'] is not None and frame == path['out_region'].frame():
                plot_interaction(1, gt=Ellipse.from_region(path['out_region']).to_dict(), color=color)

            if regions_in_frame is not None:
                for r in regions_in_frame:
                    plt.plot(r['contour_yx'][:, 1], r['contour_yx'][:, 0], color=r['color'], linestyle='dotted', linewidth=0.5)

            # for t in self.project.chm.tracklets_in_frame(frame):
            #     r = t.get_region_in_frame(self.project.gm, frame)
            #     yx = r.contour_without_holes()
            #     if t.is_single():
            #         color = 'white'
            #     elif t.is_multi():
            #         color = 'gray'
            #     else:
            #         color = 'red'
            #     plt.plot(yx[:, 1], yx[:, 0], color=color, linestyle='dotted', linewidth=0.5)

    if gt is not None:
        for track_id, row in gt.loc[frame + 1].iterrows():
            plt.plot(row.x, row.y, '*w')

    if roi is not None:
        plt.xlim(roi.x_, roi.x_max_)
        plt.ylim(roi.y_max_, roi.y_)

    # for t in incoming:
    #     r = Ellipse.from_region(t.get_region(self.project.gm, -1))
    #     plt.plot(r.x, r.y, '.r')
    #     if r.frame == frame:
    #         plot_interaction(1, gt=r.to_dict())
    # for t in outcoming:
    #     r = Ellipse.from_region(t.get_region(self.project.gm, 0))
    #     plt.plot(r.x, r.y, '.b')
    #     if r.frame == frame:
    #         plot_interaction(1, gt=r.to_dict())
    save_figure_as_image(os.path.join(out_dir, '%03d.jpg' % i), fig)
    plt.close(fig)


def train(project_dir, output_model_dir, temp_dataset_dir, num_objects=2, delete_data=False, video_file=None):  # model_dir=None,
    """

    Creates interaction detection model

    :param project_dir:
    :param output_model_dir: the model weights are saved in weights.h5
    :param temp_dataset_dir:
    :param model_dir:
    :param num_objects:
    :return:
    """
    dg = DataGenerator()
    dg.write_synthetized_interactions(project_dir, 1080, num_objects,
                                      join(temp_dataset_dir, 'train.csv'), 10,
                                      join(temp_dataset_dir, 'images.h5'), 'train',
                                      write_masks=True, video_file=video_file)
    dg.write_synthetized_interactions(project_dir, 100, num_objects,
                                      join(temp_dataset_dir, 'test.csv'), 'random',
                                      join(temp_dataset_dir, 'images.h5'), 'test',
                                      write_masks=True, video_file=video_file)
    ti = TrainInteractions(num_objects, num_input_layers=1)
    ti.train_and_evaluate(temp_dataset_dir, 0.42, 10, model='mobilenet', input_layers=1,
                          experiment_dir=output_model_dir)
    if delete_data:
        remove(join(temp_dataset_dir, 'images.h5'))
        remove(join(temp_dataset_dir, 'train.csv'))
        remove(join(temp_dataset_dir, 'test.csv'))


def detect_and_visualize(model_dir, in_img, x, y, out_img=None):
    """
    Run detector of objects in interaction on single image.

    :param model_dir: directory with trained detector model (config.yaml, model.yaml)
    :param in_img: input image file
    :param x: position of the interaction
    :param y: position of the interaction
    :param out_img: output image file with results visualization
    """
    img = imageio.imread(in_img)
    detector = InteractionDetector(model_dir)
    detections = detector.detect(img, (x, y))
    for obj_i in range(2):
        detections['{}_major'.format(obj_i)] = 60
        detections['{}_minor'.format(obj_i)] = 15
    detector.draw_detections(img, detections)
    if out_img is None:
        root, ext = os.path.splitext(in_img)
        out_img = root + '_detections' + ext
    save_prediction_img(out_img, 2, img, detections)


if __name__ == '__main__':
    # fire.Fire({
    #   'detect_and_visualize': detect_and_visualize,
    #   'train': train,
    # })
    from core.project.project import Project
    import matplotlib.pylab as plt
    import random
    # plt.switch_backend('Qt4Agg')

    project_dir = '/datagrid/ferda/projects/1_initial_projects_180921_ILP/180810_2359_Cam1_ILP_cardinality'
#     project_dir = '/home/matej/prace/ferda/projects/2_temp/180810_2359_Cam1_ILP_cardinality'
    project = Project(project_dir)
    # ../projects/2_temp/180810_2359_Cam1_ILP_cardinality
    project_name = '181101_ants1'

    gt = load_mot('data/GT/Cam1_clip.avi.txt')
    # gt = None
    gt[['x', 'y']] -= [project.video_crop_model[key] for key in ['x1', 'y1']]

    #detector = InteractionDetector('/home/matej/prace/ferda/experiments/180913_1533_single_concat_conv3_alpha0_01', project)
#    detector = InteractionDetector('/datagrid/ferda/models/181101_1537_tracker_cam1_1k_aug', project)
    # detector = InteractionDetector('/home/matej/prace/ferda/experiments/181109_1114_cam1_5k_aug100', project)
    detector = InteractionDetector('/datagrid/personal/smidm1/ferda/interactions/experiments/181116_1604_cam1_5k_aug100_rot/', project) # TODO

    if False:
        from generate_data import DataGenerator
        dg = DataGenerator()
        dg._load_project(project_dir)
        idx, single_tracklets = dg._get_single_region_tracklets()

        # random_tracklets_idx = random.sample(range(len(single_tracklets)), 60)
        for i, tracklet_regions in enumerate(single_tracklets):
            regions, last_frame, success = detector.track_single_object(
                Ellipse.from_region(tracklet_regions[0]),
                forward=True, max_frames=tracklet_regions[-1].frame() - tracklet_regions[0].frame())

            detector.visualize_single_tracklet(regions, gt, 'out/single_tracking_rot/%03d' % i)

    else:
        dense_subgraphs = detector.find_dense_subgraphs()
        dense_subgraphs = sorted(dense_subgraphs, key=lambda x: len(x['ids']), reverse=True)
        if True:
            for i, dense in enumerate(tqdm(dense_subgraphs)):
                incoming, outcoming, _ = detector.get_tracklets_from_dense(dense['graph'])
                try:
                    min_frame = min([t.end_frame(detector.project.gm) for t in incoming])
                    max_frame = max([t.start_frame(detector.project.gm) for t in outcoming])
                except ValueError:
                    continue
                for j, tracklet in enumerate(incoming): # [t for t in incoming if t.start_frame(detector.project.gm) == min_frame]):
                    regions, last_frame, success = detector.track_single_object(
                        Ellipse.from_region(tracklet.get_region(detector.project.gm, -1)),
                        forward=True, max_frames=max_frame-tracklet.end_frame(detector.project.gm))  # max_frame-min_frame
                    detector.visualize_single_tracklet(regions, gt, 'out/simple_multi_tracking_rot/%03d_%d' % (i, j))
                break
        else:
            import pickle
            out_filename = project_name + '_dense_sections_tracklets.pkl'
            try:
                with file(out_filename, 'rb') as fr:
                    dense_sections_tracklets = pickle.load(fr)
            except:
                dense_sections_tracklets = {}

            for i, dense in enumerate(tqdm(dense_subgraphs)):
                # if i != 8:
                #     continue
                if i not in dense_sections_tracklets:
                    dense_sections_tracklets[i] = detector.track_dense(dense['graph'], dense['ids'])
                    with file(out_filename, 'wb') as fw:
                        pickle.dump(dense_sections_tracklets, fw)
                detector.visualize_tracklets(dense['graph'], dense_sections_tracklets[i], 'out/dense_tracking/%03d' % i, gt)

            # i = 10
            # detector.eval_dense_section(dense_subgraphs[i]['graph'], dense_sections_tracklets[i])
