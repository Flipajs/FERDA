__author__ = 'fnaiser'
import os
import sys
import inspect
import multiprocessing
pool=multiprocessing.Pool(processes=4)
from subprocess import call
call(["hostname"])
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import sys
from utils.video_manager import get_auto_video_manager
import cPickle as pickle
from core.region.mser import ferda_filtered_msers
from core.graph.solver import Solver
from core.project.project import Project
from core.settings import Settings as S_
from utils.img import prepare_for_segmentation
from core.region.region_manager import RegionManager
from core.graph.chunk_manager import ChunkManager
import numpy as np
import time
import cv2
from core.region.mser_operations import children_filter
import fire
import subprocess
import tqdm


def segment(proj, img):
    proj.segmentation_model.set_image(img)
    seg = proj.segmentation_model.predict()

    # make hard threshold
    if False:
        result = seg < 0.5
        result = np.asarray(result, dtype=np.uint8) * 255
    else:
        result = np.asarray((-seg * 255) + 255, dtype=np.uint8)

    return result


def prepare_img(proj, img):
    grayscale = True
    if hasattr(proj, 'segmentation_model') and proj.segmentation_model is not None:
        grayscale = False

    return prepare_for_segmentation(img, proj, grayscale)


def check_intersection(rois, roi):
    intersect = -1

    for i, r in enumerate(rois):
        if r.is_intersecting(roi):
            return i

    return intersect


def get_rois(msers, img, prediction_optimisation_border):
    rois = []

    for m in msers:
        roi = m.roi().safe_expand(prediction_optimisation_border, img)
        # if roi.width() > 400 or roi.height() > 400:
        #     continue

        while True:
            intersect = check_intersection(rois, roi)

            if intersect > -1:
                roi = rois[intersect].union(roi)
                rois.pop(intersect)
            else:
                rois.append(roi)
                break

        # rois.append(roi)

    return rois


def run_parallelization(working_dir, proj_name, id=None, frames_in_row=None, last_n_frames=None, sge=False):
    # /home/matej/prace/ferda/projects/after_par/5Zebrafish_nocover_22min 5Zebrafish_nocover_22min 0 100 0
    if id is None and frames_in_row is None and last_n_frames is None:
        try:
            with open(os.path.join(working_dir, 'limits.txt'), 'r') as fr:
                # the file should look like:
                # 0	100	0
                # 1	100	0
                # 2	100	0
                # ...
                parameters = []
                for line in fr.readlines():
                    split = line.split()
                    parameters.append({'id': int(split[0]),
                                       'frames_in_row': int(split[1]),
                                       'last_n_frames': int(split[2])})
            for par in tqdm.tqdm(parameters, desc='processing segmentation jobs in sequence'):
                if sge:
                    assert False, 'not implemented'
                    subprocess.check_call(
                        ['scripts/sun_grid_engine_parallelization/compute_single_id.sh',
                         working_dir, proj_name, par['id'], par['frames_in_row'], par['last_n_frames']
                         ])
                else:
                    compute_single_id(working_dir, proj_name, par['id'], par['frames_in_row'], par['last_n_frames'])

        except IOError as e:
            print('Can''t open limits.txt file: {}'.format(e))
            raise
    else:
        assert id is not None and frames_in_row is not None and last_n_frames is not None
        compute_single_id(working_dir, proj_name, id, frames_in_row, last_n_frames)


def compute_single_id(working_dir, proj_name, id, frames_in_row, last_n_frames):
    # check if part was computed before
    if os.path.isfile(working_dir + '/temp/part' + str(id) + '_rm.sqlite3'):
        if os.stat(working_dir + '/temp/part' + str(id) + '_rm.sqlite3') != 0:
            if os.path.isfile(working_dir + '/temp/part' + str(id) + '.pkl'):
                if os.stat(working_dir + '/temp/part' + str(id) + '.pkl') != 0:
                    print('Part {} already processed'.format(id))
                    return
    proj = Project()
    proj.load(working_dir + '/' + proj_name + '.fproj')
    # proj.stats.major_axis_median = 18
    # proj.solver_parameters.max_edge_distance_in_ant_length = 2
    try:
        use_roi_prediction_optimisation = proj.other_parameters.segmentation_use_roi_prediction_optimisation

        # TODO: express based on major axis in body length
        prediction_optimisation_border = proj.other_parameters.segmentation_prediction_optimisation_border
        full_segmentation_refresh = proj.other_parameters.segmentation_full_segmentation_refresh_in
    except:
        use_roi_prediction_optimisation = True  # obsolete, per-pixel classification
        prediction_optimisation_border = 25
        full_segmentation_refresh = 25
    if not os.path.exists(proj.working_directory + '/temp'):
        try:
            os.mkdir(proj.working_directory + '/temp')
        except OSError:
            pass
    if not proj.is_cluster():
        temp_local_path = proj.working_directory + '/temp'
    else:
        temp_local_path = '/localhome/casillas/'

        if not os.path.exists(temp_local_path + proj_name):
            try:
                os.mkdir(temp_local_path + proj_name)
            except:
                print(temp_local_path + proj_name + "   was created between check and mkdir")

        temp_local_path = temp_local_path + proj_name

        if not os.path.exists(temp_local_path + '/temp'):
            try:
                os.mkdir(temp_local_path + '/temp')
            except:
                print(temp_local_path + '/temp' + "   was created between check and mkdir")

        temp_local_path = temp_local_path + '/temp'

    # init managers
    solver = Solver(proj)
    from core.graph.graph_manager import GraphManager
    proj.gm = GraphManager(proj, proj.solver.assignment_score)
    # TODO: add global params
    proj.rm = RegionManager(db_wd=temp_local_path, db_name='part' + str(id) + '_rm.sqlite3', cache_size_limit=10000)
    proj.chm = ChunkManager()
    proj.color_manager = None
    S_.general.log_graph_edits = False
    vid = get_auto_video_manager(proj)
    if id * frames_in_row > 0:
        img = vid.seek_frame(id * frames_in_row)
    else:
        img = vid.next_frame()
    if img is None:
        raise Exception("img is None, there is something wrong with frame: " + str(id * frames_in_row))
    rois = []
    img = prepare_img(proj, img)
    msers_t = 0
    solver_t = 0
    vid_t = 0
    file_t = 0
    border2 = 3
    jj = 0
    # for all frames: extract regions and add them to the graph
    for i in range(frames_in_row + last_n_frames):
        frame = id * frames_in_row + i

        s = time.time()

        # per pixel classification -> fg, bg (not used)
        if hasattr(proj, 'segmentation_model'):
            if rois and i % full_segmentation_refresh != 0:
                try:
                    t = time.time()

                    t2 = 0
                    new_im = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

                    area = 0
                    for roi in rois:
                        area += roi.width() * roi.height()

                    for roi in rois:
                        tl = roi.top_left_corner()
                        br = roi.bottom_right_corner()

                        h1 = tl[0]
                        h2 = min(img.shape[0] - 1, br[0])

                        w1 = tl[1]
                        w2 = br[1]

                        crop = img[h1:h2, w1:w2, :].copy()

                        # add border2 (to prevent segmentation artefacts
                        crop = cv2.copyMakeBorder(crop, border2, border2, border2, border2, cv2.BORDER_REPLICATE)

                        t2_ = time.time()
                        proj.segmentation_model.set_image(crop)
                        # t2_ = time.time() - t2_
                        # t2 += t2_
                        # print t2_
                        # t2_ = time.time()
                        seg = proj.segmentation_model.predict()
                        t2_ = time.time() - t2_
                        t2 += t2_

                        # print t2_, crop.shape

                        # remove border2
                        seg = seg[border2:-border2, border2:-border2].copy()

                        jj += 1

                        # make hard threshold
                        if True:
                            seg_img = seg < 0.5
                            seg_img = np.asarray(seg_img, dtype=np.uint8) * 255
                        else:
                            seg_img = np.asarray((-seg * 255) + 255, dtype=np.uint8)

                        new_im[h1:h2, w1:w2] = seg_img.copy()

                    print "segmentation time: {:.3f}, #roi: {} roi area: {} roi coverage: {:.3f}".format(
                        time.time() - t, len(rois), area, area / float(img.shape[0] * img.shape[1]))
                    # t = time.time()
                    # segment(proj, img)
                    # print "without ", time.time() - t

                    img = new_im
                except:
                    img = segment(proj, img)
            else:
                img = segment(proj, img)

        # get segmented regions
        msers = ferda_filtered_msers(img, proj, frame)

        if proj.colormarks_model:
            proj.colormarks_model.assign_colormarks(proj, msers)

        proj.rm.add(msers)
        msers_t += time.time() - s

        s = time.time()

        # Check for last frame...
        if i + 1 < frames_in_row + last_n_frames:
            img = vid.next_frame()
            if img is None:
                raise Exception("img is None, there is something wrong with frame: " + str(frame))

            img = prepare_img(proj, img)

        vid_t += time.time() - s

        s = time.time()

        if use_roi_prediction_optimisation:
            rois = get_rois(msers, img, prediction_optimisation_border)

        # add regions to graph
        proj.gm.add_regions_in_t(msers, frame, fast=True)
        solver_t += time.time() - s

    # if proj.solver_parameters.use_emd_for_split_merge_detection():
    #     solver.detect_split_merge_cases()
    s = time.time()
    print "#Edges BEFORE: ", proj.gm.g.num_edges()
    try:
        # TODO:
        if proj.type == 'colony':
            rules = [solver.adaptive_threshold]

            while True:
                if solver.simplify(rules=rules) == 0:
                    break
        else:
            solver.simplify(rules=[solver.one2one])
    except:
        solver.one2one()
    print "#Edges AFTER: ", proj.gm.g.num_edges()
    solver_t += time.time() - s
    s = time.time()
    with open(proj.working_directory + '/temp/part' + str(id) + '.pkl', 'wb') as f:
        p = pickle.Pickler(f, -1)
        p.dump(proj.gm.g)
        p.dump(proj.gm.get_all_relevant_vertices())
        p.dump(proj.chm)
    file_t = time.time() - s
    print "#Vertices: {}, #Edges: {}".format(proj.gm.g.num_vertices(), proj.gm.g.num_edges())
    print "MSERS t:", round(msers_t, 2), "SOLVER t: ", round(solver_t, 2), "VIDEO t:", round(vid_t,
                                                                                             2), "FILE t: ", round(
        file_t, 2), "SUM t / frames_in_row:", round((msers_t + solver_t + vid_t + file_t) / float(frames_in_row), 2)
    if proj.is_cluster():
        import shutil
        import glob
        for file in glob.glob(temp_local_path + '/part' + str(id) + '_rm.sqlite3'):
            shutil.move(file, working_dir + '/temp')


if __name__ == '__main__':
    fire.Fire(run_parallelization)
