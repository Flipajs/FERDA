import tempfile

import fire
import imageio
from keras.applications.mobilenet import mobilenet
import numpy as np
import pandas as pd
import yaml
from keras.models import model_from_yaml, model_from_json
import os.path
from os.path import join
from os import remove
from scipy.special import expit
from tqdm import tqdm
import graph_tool.all as gt
import cv2
import scipy
import itertools

from core.interactions.visualization import save_prediction_img, show_prediction
from utils.angles import angle_absolute_error_direction_agnostic
from utils.geometry import roi_corners
from utils.img import safe_crop
from utils.objectsarray import ObjectsArray
from core.interactions.generate_data import DataGenerator
from core.interactions.train import TrainInteractions
from core.region.transformableregion import TransformableRegion
from core.region.region import Region

from joblib import Memory
# memory = Memory('out/cache', verbose=0)
memory = Memory(None, verbose=0)


def get_hull_poly(region, epsilon_px=5):
    poly = cv2.approxPolyDP(np.fliplr(region.contour()), epsilon_px, True)
    hull = cv2.convexHull(poly.reshape((-1, 2))).reshape((-1, 2))
    return np.vstack((hull, hull[0:1]))


class EllipticRegion(object):
    @classmethod
    def from_region(cls, region):
        yx = region.centroid()
        tmp = cls(yx[1], yx[0], -np.rad2deg(region.theta_), 2 * region.major_axis_, 2 * region.minor_axis_,
                  region.frame())
        return tmp

    def to_region(self):
        r = Region(is_origin_interaction=True, frame=self.frame)
        r.centroid_ = self.xy[::-1]
        r.theta_ = -np.deg2rad(self.angle_deg)
        r.major_axis_ = self.major / 2
        r.minor_axis_ = self.minor / 2
        return r

    @classmethod
    def from_dict(cls, region_dict):
        return cls(region_dict['0_x'], region_dict['0_y'], region_dict['0_angle_deg'], region_dict['0_major'],
                   region_dict['0_minor'])

    def __init__(self, x=None, y=None, angle_deg=None, major=None, minor=None, frame=None):
        self.x = x
        self.y = y
        self.angle_deg = angle_deg  # img cw
        self.major = major
        self.minor = minor
        self.frame = frame

    def to_dict(self):
        return ({
            '0_x': self.x,
            '0_y': self.y,
            '0_angle_deg': self.angle_deg,
            '0_major': self.major,
            '0_minor': self.minor,
        })

    @property
    def xy(self):
        return np.array((self.x, self.y))

    @property
    def area(self):
        return cv2.contourArea(self.to_poly())

    def to_poly(self):
        return cv2.ellipse2Poly((int(self.x), int(self.y)), (int(self.major), int(self.minor)), int(self.angle_deg), 0,
                                360, 30)

    def get_overlap(self, el_region):
        if isinstance(el_region, EllipticRegion):
            el_region = el_region.to_poly()
        area, poly = cv2.intersectConvexConvex(self.to_poly(), el_region)
        #         poly = poly.reshape((-1, 2))
        return area

    def to_array(self):
        return np.array([self.x, self.y, self.angle_deg, self.major, self.minor, self.frame])

    def __add__(self, other):
        assert self.frame == other.frame
        mean = np.vstack((self.to_array(), other.to_array())).mean(axis=0)
        el = EllipticRegion(*mean)
        el.frame = int(el.frame)
        return el


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
        interaction_graph = gt.Graph()
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
        pos = graph.new_vertex_property('vector<float>')
        fill_color = graph.new_vertex_property('string')
        category_prop = graph.vertex_properties['category']
        for v in graph.vertices():
            t = graph.vertex_properties['tracklet'][v]
            if category_prop[v] == 'multi':
                pos[v] = ((t.end_frame(self.project.gm) + t.start_frame(self.project.gm)) / 2., 0)
                fill_color[v] = 'red'
            elif category_prop[v] == 'in':
                pos[v] = (t.end_frame(self.project.gm), np.random.randint(-30, -15))
                fill_color[v] = 'white'
            elif category_prop[v] == 'out':
                pos[v] = (t.start_frame(self.project.gm), np.random.randint(15, 30))
                fill_color[v] = 'black'
            elif category_prop[v] == 'in_out':
                pos[v] = (t.start_frame(self.project.gm), np.random.randint(15, 30))
                fill_color[v] = 'gray'
            else:
                assert False

        pos = gt.sfdp_layout(graph, pos=pos)
        # pos = gt.planar_layout(interaction_graph, pos=pos)
        gt.graph_draw(graph, pos=pos, vertex_fill_color=fill_color,
                      inline=True,
                      vertex_text=graph.vertex_properties['text'],
                      edge_marker_size=10,
                      vertex_size=10,
                      output_size=(1200, 600),
#                      output_size=(1800, 1800),
                      #                  output='dense_subgraph.svg'
                      )
        # output_size=(800, 400)

    def track_single_object_in_multi_tracklet(self, starting_region, multi_tracklet, forward=True, max_frames=np.inf):
        prediction = starting_region.to_dict()
        start_frame = multi_tracklet.start_frame(self.project.gm)
        end_frame = multi_tracklet.end_frame(self.project.gm)
        if forward:
            frame = start_frame
        else:
            frame = end_frame

        i = 0
        regions = []
        while (start_frame <= frame <= end_frame) and i <= max_frames:  # overlap > 0 and
            multi_region = multi_tracklet.get_region_in_frame(self.project.gm, frame)
            prev_prediction = prediction.copy()
            # img = self.project.img_manager.get_whole_img(frame)
            prediction, _, _, _ = self.detect_single_frame(frame, prev_prediction)
            prediction_r = EllipticRegion.from_dict(prediction)
            prediction_r.frame = frame

            # multi_poly = get_hull_poly(multi_region)
            # cv2.polylines(img, [prediction_r.to_poly()], True, (255, 0, 0), 1)
            # cv2.polylines(img, [multi_poly], True, (0, 255, 0))
            # # save_prediction_img(None, 1, img, pred=prediction, gt=None, scale=1.1)
            # images.append(safe_crop(img, tuple([prediction[x] for x in ('0_x', '0_y')]), 200)[0])

            overlap = prediction_r.get_overlap(get_hull_poly(multi_region))
            if overlap == 0:
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

    def get_tracklets_from_dense(self, graph):
        category_prop = graph.vertex_properties['category']
        tracklet_prop = graph.vertex_properties['tracklet']
        incoming = [tracklet_prop[v] for v in graph.vertices() if
                          (category_prop[v] == 'in') or (category_prop[v] == 'in_out')]
        outcoming = [tracklet_prop[v] for v in graph.vertices() if
                           (category_prop[v] == 'out') or (category_prop[v] == 'in_out')]
        multi = [tracklet_prop[v] for v in graph.vertices() if category_prop[v] == 'multi']
        return incoming, outcoming, multi

    def track_single_object_in_dense_subgraph(self, graph, vertices, single_tracklet, forward=True):

        def track_recursive(vertex, graph, regions_path, forward=True):
            if forward:
                # last_region_idx = -1
                neighbors = vertex.out_neighbors()
            else:
                # last_region_idx = 0
                neighbors = vertex.in_neighbors()
            results = []
            # tracklet = graph.vp.tracklet[vertex]
            for v in neighbors:
                if graph.vp.category[v] == 'out' or graph.vp.category[v] == 'in_out':
                    pass
                elif graph.vp.category[v] == 'multi':
                    multi_tracklet = graph.vp.tracklet[v]
                    # last_region = EllipticRegion.from_region(tracklet.get_region(self.project.gm, last_region_idx))
                    regions, last_frame, success = \
                        self.track_single_object_in_multi_tracklet(regions_path[-1], multi_tracklet, forward)
                    if success:
                        track_recursive(v, graph, regions, forward)
                    results.append(regions)
                elif graph.vp.category[v] == 'in' or graph.vp.category[v] == 'in_out':
                    pass
                else:
                    assert False
            if results:
                path_lengths = [len(result) for result in results]
                best_idx = path_lengths.index(max(path_lengths))
                regions_path.extend(results[best_idx])

        v = vertices[single_tracklet.id()]
        if forward:
            adjacent_region_idx = -1
        else:
            adjacent_region_idx = 0

        regions_path = [EllipticRegion.from_region(single_tracklet.get_region(self.project.gm, adjacent_region_idx))]
        track_recursive(v, graph, regions_path, forward)
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
            '0_angle_deg': np.rad2deg(r.theta_),
        })

    def detect_single_frame(self, frame, prev_detection):
        @memory.cache
        def _detect_single_frame(frame, prev_detection):
            img = self.project.img_manager.get_whole_img(frame)
            pred_dict, pred_, delta_xy, img_crop = self.detect_single(img, prev_detection)
            return pred_dict, pred_, delta_xy, img_crop

        return _detect_single_frame(frame, prev_detection)

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
        timg.rotate(-prev_detection['0_angle_deg'], prev_xy[::-1])
        timg.set_img(img)
        img_rotated = timg.get_img()

        img_crop, delta_xy = safe_crop(img_rotated, prev_xy, self.config['input_size_px'])
        img = np.expand_dims(img_crop, 0).astype(np.float) / 255.
        pred = self.m.predict(img)
        pred_ = pred.copy().flatten()
        pred = self.ti.postprocess_predictions(pred)

        pred_dict = self.predictions.array_to_dict(pred)
        pred_dict['0_x'] += delta_xy[0]
        pred_dict['0_y'] += delta_xy[1]
        pred_dict['0_major'] = prev_detection['0_major']
        pred_dict['0_minor'] = prev_detection['0_minor']
        pred_dict['0_angle_deg'] += prev_detection['0_angle_deg']
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

    def visualize_tracklets(self, graph, paths, out_dir):
        from utils.img import safe_crop
        from scripts.montage import save_figure_as_image
        from core.interactions.visualization import plot_interaction, save_prediction_img
        import matplotlib.pylab as plt
        vm = self.project.get_video_manager()

        roi, (start_frame, end_frame) = self.get_bounds(graph)
        incoming, outcoming, _ = self.get_tracklets_from_dense(graph)

        try:
            os.makedirs(out_dir)
        except OSError:
            pass

        for i, frame in enumerate(range(start_frame, end_frame + 1)):
            fig = plt.figure()
            plt.imshow(vm.get_frame(frame))
            colors = itertools.cycle(['red', 'blue', 'green', 'yellow', 'white'])
            # for path_set, color in zip(paths, ('r', 'b')):
#                for regions in path_set:
            for path, color in zip(paths, colors):
                regions = path['regions']
                plt.plot([r.x for r in regions], [r.y for r in regions], color=color)
                frames = [r.frame for r in regions]
                try:
                    r = regions[frames.index(frame)]
                    plot_interaction(1, pred=r.to_dict(), color=color)
                    plt.plot(r.x, r.y, 'o', color=color)
                except ValueError:
                    pass
                if frame == path['in_region'].frame():
                    plot_interaction(1, gt=EllipticRegion.from_region(path['in_region']).to_dict(), color=color)
                if frame == path['out_region'].frame():
                    plot_interaction(1, gt=EllipticRegion.from_region(path['out_region']).to_dict(), color=color)

            plt.xlim(roi.x_, roi.x_max_)
            plt.ylim(roi.y_, roi.y_max_)
            # for t in incoming:
            #     r = EllipticRegion.from_region(t.get_region(self.project.gm, -1))
            #     plt.plot(r.x, r.y, '.r')
            #     if r.frame == frame:
            #         plot_interaction(1, gt=r.to_dict())
            # for t in outcoming:
            #     r = EllipticRegion.from_region(t.get_region(self.project.gm, 0))
            #     plt.plot(r.x, r.y, '.b')
            #     if r.frame == frame:
            #         plot_interaction(1, gt=r.to_dict())
            save_figure_as_image(os.path.join(out_dir, '%03d.jpg' % i), fig)
            plt.close(fig)

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

        # np.fromfunction(np.vectorize(lambda i, j: min_dist(fwd[i[0]], bwd[j[1]])), (len(fwd), len(bwd)))
        cost_matrix, frames_matrix = np.vectorize(lambda i, j: min_dist(fwd[i]['regions'], bwd[j]['regions']))(
            *np.indices((len(fwd), len(bwd))))
        ii, jj = scipy.optimize.linear_sum_assignment(cost_matrix)

        paths = []
        for i, j in zip(ii, jj):
            mean_major_px = np.concatenate(([el.major for el in fwd[i]['regions']],
                                            [el.major for el in bwd[j]['regions']])).mean()
            if cost_matrix[i, j] > 0.5 * mean_major_px:
                # TODO, add confident parts of fwd, bwd
                import warnings
                warnings.warn('TODO: fwd bwd distance high: distance {} px, threshold {} px'.format(cost_matrix[i, j], 0.5 * mean_major_px))

            frame = frames_matrix[i, j]
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
        #     paths.append(fwd[i][:(fwd_idx + 1)])
        #     paths.append([mean_el])
        #     paths.append(bwd[j][bwd_idx:])
        return paths  # , fwd, bwd


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
    from skimage.util import montage
    import matplotlib.pylab as plt
    # plt.switch_backend('Qt4Agg')

    project = Project('../projects/2_temp/180810_2359_Cam1_ILP_cardinality')

    # '/home/matej/prace/ferda/experiments/180913_1533_single_concat_conv3_alpha0_01'
    detector = InteractionDetector('/datagrid/ferda/models/180913_1533_tracker_single_concat_conv3_alpha0_01',
                                   project)
    dense_subgraphs = detector.find_dense_subgraphs()
    dense_subgraphs = sorted(dense_subgraphs, key=lambda x: len(x['ids']), reverse=True)

    ##

    for i, dense in enumerate(tqdm(dense_subgraphs)):
        # if i != 5:
        #     continue
        paths = detector.track_dense(dense['graph'], dense['ids'])
        detector.visualize_tracklets(dense['graph'], paths, 'out/%03d' % i)

        # detector.draw_graph(dense['graph'])

    ##

    # %load_ext autoreload
    # %autoreload 2


