from __future__ import print_function

__author__ = 'fnaiser'
import inspect
import multiprocessing
import sys
import os
from os.path import join

# pool = multiprocessing.Pool(processes=4)
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)

from utils.video_manager import get_auto_video_manager
import cPickle as pickle
from core.region.mser import get_filtered_msers
from core.graph.solver import Solver
from core.project.project import Project
from config import config
from utils.img import prepare_for_segmentation
from core.region.region_manager import RegionManager
from core.graph.chunk_manager import ChunkManager
import numpy as np
import time
import cv2
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


def segmentation(project_file, sge=False):
    """
    Process all segmentation parts according to limits.txt.

    :param project_file: project file
    :return: int, number of parts
    """
    working_dir, _ = Project.get_project_dir_and_file(project_file)
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
                # subprocess.check_call(
                #     ['scripts/sun_grid_engine_parallelization/compute_single_id.sh',
                #      working_dir, proj_name, par['id'], par['frames_in_row'], par['last_n_frames']
                #      ])
            else:
                do_segmentation_part(project_file, par['id'], par['frames_in_row'], par['last_n_frames'])
        return len(parameters)

    except IOError as e:
        print('Can''t open limits.txt file: {}'.format(e))
        raise


def do_segmentation_part(project_file, part_id, frames_in_row, last_n_frames):
    working_dir, _ = Project.get_project_dir_and_file(project_file)
    # check if part was computed before
    temp_path = os.path.join(working_dir, 'temp')
    sqlite_filename = join(temp_path, 'part{}_rm.sqlite3'.format(part_id))
    pkl_filename = join(temp_path, 'part{}.pkl'.format(part_id))
    if os.path.isfile(sqlite_filename) and os.path.getsize(sqlite_filename) != 0 \
            and os.path.isfile(pkl_filename) and os.path.getsize(pkl_filename) != 0:
        print('Part {} already processed.'.format(part_id))
        return
    proj = Project()
    proj.load(project_file)
    try:
        use_roi_prediction_optimisation = proj.other_parameters.segmentation_use_roi_prediction_optimisation

        # TODO: express based on major axis in body length
        prediction_optimisation_border = proj.other_parameters.segmentation_prediction_optimisation_border
        full_segmentation_refresh = proj.other_parameters.segmentation_full_segmentation_refresh_in
    except:
        use_roi_prediction_optimisation = True  # obsolete, per-pixel classification
        prediction_optimisation_border = 25
        full_segmentation_refresh = 25

    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)

    # init managers
    solver = Solver(proj)
    from core.graph.graph_manager import GraphManager
    proj.gm = GraphManager(proj, proj.solver.assignment_score)
    # TODO: add global params
    proj.rm = RegionManager(db_wd=temp_path, db_name='part{}_rm.sqlite3'.format(part_id), cache_size_limit=10000)
    proj.chm = ChunkManager()
    proj.color_manager = None
    config['general']['log_graph_edits'] = False
    vid = get_auto_video_manager(proj)
    if part_id * frames_in_row > 0:
        img = vid.seek_frame(part_id * frames_in_row)
    else:
        img = vid.next_frame()
    if img is None:
        raise Exception("img is None, there is something wrong with frame: {}".format(part_id * frames_in_row))
    rois = []
    img = prepare_img(proj, img)
    msers_t = 0
    solver_t = 0
    vid_t = 0
    # for all frames: extract regions and add them to the graph
    for i in range(frames_in_row + last_n_frames):
        frame = part_id * frames_in_row + i

        s = time.time()

        # per pixel classification -> fg, bg (not used)
        if hasattr(proj, 'segmentation_model'):
            img = frame_segmentation(img, i, proj, rois, full_segmentation_refresh=full_segmentation_refresh)
        # get segmented regions
        msers = get_filtered_msers(img, proj, frame)

        if proj.colormarks_model:
            proj.colormarks_model.assign_colormarks(proj, msers)

        proj.rm.add(msers)
        msers_t += time.time() - s

        s = time.time()

        # Check for last frame...
        if i + 1 < frames_in_row + last_n_frames:
            img = vid.next_frame()
            if img is None:
                raise Exception("img is None, there is something wrong with frame: {}".format(frame))

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
    print("#Edges BEFORE: ", proj.gm.g.num_edges())
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
    print ("#Edges AFTER: ", proj.gm.g.num_edges())
    solver_t += time.time() - s
    s = time.time()
    with open(pkl_filename, 'wb') as f:
        p = pickle.Pickler(f, -1)
        p.dump(proj.gm.g)
        p.dump(proj.gm.get_all_relevant_vertices())
        p.dump(proj.chm)
    file_t = time.time() - s
    print("#Vertices: {}, #Edges: {}".format(proj.gm.g.num_vertices(), proj.gm.g.num_edges()))
    print("MSERS t:", round(msers_t, 2), "SOLVER t: ", round(solver_t, 2), "VIDEO t:", round(vid_t,
                                                                                              2), "FILE t: ", round(
        file_t, 2), "SUM t / frames_in_row:", round((msers_t + solver_t + vid_t + file_t) / float(frames_in_row), 2))


def frame_segmentation(img, i, project, rois, border_px=3, full_segmentation_refresh=25):
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

                # add border_px (to prevent segmentation artefacts
                crop = cv2.copyMakeBorder(crop, border_px, border_px, border_px, border_px, cv2.BORDER_REPLICATE)

                t2_ = time.time()
                project.segmentation_model.set_image(crop)
                # t2_ = time.time() - t2_
                # t2 += t2_
                # print t2_
                # t2_ = time.time()
                seg = project.segmentation_model.predict()
                t2_ = time.time() - t2_
                t2 += t2_

                # print t2_, crop.shape

                # remove border_px
                seg = seg[border_px:-border_px, border_px:-border_px].copy()

                # make hard threshold
                if True:
                    seg_img = seg < 0.5
                    seg_img = np.asarray(seg_img, dtype=np.uint8) * 255
                else:
                    seg_img = np.asarray((-seg * 255) + 255, dtype=np.uint8)

                new_im[h1:h2, w1:w2] = seg_img.copy()

            print("segmentation time: {:.3f}, #roi: {} roi area: {} roi coverage: {:.3f}".format(
                time.time() - t, len(rois), area, area / float(img.shape[0] * img.shape[1])))
            # t = time.time()
            # segment(proj, img)
            # print "without ", time.time() - t

            img = new_im
        except:
            img = segment(project, img)
    else:
        img = segment(project, img)

    return img


if __name__ == '__main__':
    fire.Fire({'segmentation': segmentation,
              'part_segmentation': do_segmentation_part})

