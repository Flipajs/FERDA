from __future__ import print_function

__author__ = 'fnaiser'
import inspect
import multiprocessing
import sys
import os
from os.path import join
import numpy as np
import time
import cv2
import fire
import subprocess
import tqdm
import json
from joblib import Parallel, delayed

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
from utils.video_manager import get_auto_video_manager


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


def segmentation(project_dir):
    """
    Segment regions in all frames.

    :param project_dir: project directory
    :return: int, number of parts
    """
    project = Project(project_dir)
    vid = get_auto_video_manager(project)
    frame_num = int(vid.total_frame_count())
    frames_in_row = config['segmentation']['frames_in_row']
    Parallel(n_jobs=config['general']['n_jobs'], verbose=10)\
        (delayed(do_segmentation_part)(project_dir, i, frame_start)
         for i, frame_start in enumerate(range(0, frame_num, frames_in_row)))
    # with tqdm.tqdm(total=frame_num, desc='segmenting regions') as pbar:
    #     for i, frame_start in enumerate(range(0, frame_num, frames_in_row)):
    #         do_segmentation_part(project_file, i, frame_start, pbar.update)


def do_segmentation_part(project_dir, part_id, frame_start, frame_done_func=None):
    # check if part was computed before
    temp_path = os.path.join(project_dir, 'temp')
    sqlite_filename = join(temp_path, 'part{}_rm.sqlite3'.format(part_id))
    pkl_filename = join(temp_path, 'part{}.pkl'.format(part_id))
    if os.path.isfile(sqlite_filename) and os.path.getsize(sqlite_filename) != 0 \
            and os.path.isfile(pkl_filename) and os.path.getsize(pkl_filename) != 0:
        print('Part {} already processed.'.format(part_id))
        return
    proj = Project()
    proj.load(project_dir)
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
    frames_num = int(vid.total_frame_count())
    frames_in_row = config['segmentation']['frames_in_row']
    frame_end = frame_start + frames_in_row - 1
    if frame_end > frames_num - 1:
        frame_end = frames_num - 1

    # for all frames: extract regions and add them to the graph
    for frame in range(frame_start, frame_end + 1):
        if frame == frame_start:
            img = vid.seek_frame(frame_start)
        else:
            img = vid.next_frame()
            if img is None:
                raise Exception("failed to load frame {}".format(frame))

        # rois = []
        img = prepare_img(proj, img)

        # # per pixel classification -> fg, bg (not used)
        # if hasattr(proj, 'segmentation_model'):
        #     img = frame_segmentation(img, i, proj, rois, full_segmentation_refresh=full_segmentation_refresh)

        # get segmented regions
        msers = get_filtered_msers(img, proj, frame)

        if proj.colormarks_model:
            proj.colormarks_model.assign_colormarks(proj, msers)

        proj.rm.add(msers)

        # if use_roi_prediction_optimisation:
        #     rois = get_rois(msers, img, prediction_optimisation_border)

        # add regions to graph
        proj.gm.add_regions_in_t(msers, frame, fast=True)

        if frame_done_func is not None:
            frame_done_func()

    print("#Edges BEFORE: ", proj.gm.g.num_edges())
    try:
        solver.simplify(rules=[solver.one2one])
    except:
        solver.one2one()
    print ("#Edges AFTER: ", proj.gm.g.num_edges())
    with open(pkl_filename, 'wb') as f:
        p = pickle.Pickler(f, -1)
        p.dump(proj.gm.g)
        p.dump(proj.gm.get_all_relevant_vertices())
        p.dump(proj.chm)
    save_segmentation_info(temp_path, part_id, frame_start=frame_start, frame_end=frame_end)
    print("#Vertices: {}, #Edges: {}".format(proj.gm.g.num_vertices(), proj.gm.g.num_edges()))


def save_segmentation_info(dirname, part_id, **kwargs):
    with open(join(dirname, 'part{}.json'.format(part_id)), 'w') as fw:
        json.dump(kwargs, fw)


def load_segmentation_info(dirname, part_id):
    with open(join(dirname, 'part{}.json'.format(part_id)), 'r') as fr:
        info = json.load(fr)
    return info


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

