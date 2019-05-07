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
import math
from joblib import Parallel, delayed

# pool = multiprocessing.Pool(processes=4)
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)

from utils.video_manager import get_auto_video_manager
import cPickle as pickle
from core.region.mser import get_filtered_regions
from core.graph.solver import Solver
from core.project.project import Project, set_managers
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


def preprocess_img(proj, img):
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
    project = Project()
    project.load(project_dir, regions_optional=True, graph_optional=True, tracklets_optional=True)
    vid = get_auto_video_manager(project)
    frame_num = int(vid.total_frame_count())
    frames_in_row = config['segmentation']['frames_in_row']
    Parallel(n_jobs=config['general']['n_jobs'], verbose=10)\
        (delayed(do_segmentation_part)(project_dir, i, frame_start)
         for i, frame_start in enumerate(range(0, frame_num, frames_in_row)))  # frame_num


def is_segmentation_completed(project):
    n_frames = project.video_end_t - project.video_start_t + 1
    n_parts = int(math.ceil(n_frames / config['segmentation']['frames_in_row']))
    for i in range(n_parts + 1):
        temp_path = os.path.join(project.working_directory, 'temp/{}'.format(i))
        if not (os.path.isfile(join(temp_path, 'regions.csv')) and os.path.isfile(join(temp_path, 'regions.h5'))):
            return False
    return True


def do_segmentation_part(project_dir, part_id, frame_start, frame_done_func=None):
    # check if part was computed before
    temp_path = os.path.join(project_dir, 'temp/{}'.format(part_id))

    if os.path.isfile(join(temp_path, 'regions.csv')) and os.path.isfile(join(temp_path, 'regions.h5')):
        print('Part {} already processed.'.format(part_id))
        return
    p = Project.from_dir(project_dir, tracklets_optional=True)
    p.rm = RegionManager()  # reset RegionManager to force temporary hdf5
    set_managers(p, p.rm, p.chm, p.gm)

    config['general']['log_graph_edits'] = False
    vid = get_auto_video_manager(p)
    frames_num = vid.total_frame_count()
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

        img = preprocess_img(p, img)
        # get segmented regions
        regions = get_filtered_regions(img, p, frame)
        for r in regions:
            r.id_ = len(p.rm)
            p.rm.append(r)
        # add regions to the graph
        # p.gm.add_regions_in_t(regions, frame)

        if frame_done_func is not None:
            frame_done_func()

    # print("#Edges BEFORE: ", p.gm.g.num_edges())
    # p.solver.create_tracklets()
    # print ("#Edges AFTER: ", p.gm.g.num_edges())
    p.save(temp_path)
    # with open(pkl_filename, 'wb') as f:
    #     p = pickle.Pickler(f, -1)
    #     p.dump(p.gm.g)
    #     p.dump(p.gm.get_all_relevant_vertices())
    #     p.dump(p.chm)
    # save_segmentation_info(temp_path, frame_start=frame_start, frame_end=frame_end)
    # print("#Vertices: {}, #Edges: {}".format(p.gm.g.num_vertices(), p.gm.g.num_edges()))


def save_segmentation_info(dirname, **kwargs):
    with open(join(dirname, 'segmentation.json'), 'w') as fw:
        json.dump(kwargs, fw)


def load_segmentation_info(dirname, part_id):
    with open(join(dirname, 'part{}.json'.format(part_id)), 'r') as fr:
        info = json.load(fr)
    return info

#
# def frame_segmentation(img, i, project, rois, border_px=3, full_segmentation_refresh=25):
#     if rois and i % full_segmentation_refresh != 0:
#         try:
#             t = time.time()
#
#             t2 = 0
#             new_im = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
#
#             area = 0
#             for roi in rois:
#                 area += roi.width() * roi.height()
#
#             for roi in rois:
#                 tl = roi.top_left_corner()
#                 br = roi.bottom_right_corner()
#
#                 h1 = tl[0]
#                 h2 = min(img.shape[0] - 1, br[0])
#
#                 w1 = tl[1]
#                 w2 = br[1]
#
#                 crop = img[h1:h2, w1:w2, :].copy()
#
#                 # add border_px (to prevent segmentation artefacts
#                 crop = cv2.copyMakeBorder(crop, border_px, border_px, border_px, border_px, cv2.BORDER_REPLICATE)
#
#                 t2_ = time.time()
#                 project.segmentation_model.set_image(crop)
#                 # t2_ = time.time() - t2_
#                 # t2 += t2_
#                 # print t2_
#                 # t2_ = time.time()
#                 seg = project.segmentation_model.predict()
#                 t2_ = time.time() - t2_
#                 t2 += t2_
#
#                 # print t2_, crop.shape
#
#                 # remove border_px
#                 seg = seg[border_px:-border_px, border_px:-border_px].copy()
#
#                 # make hard threshold
#                 if True:
#                     seg_img = seg < 0.5
#                     seg_img = np.asarray(seg_img, dtype=np.uint8) * 255
#                 else:
#                     seg_img = np.asarray((-seg * 255) + 255, dtype=np.uint8)
#
#                 new_im[h1:h2, w1:w2] = seg_img.copy()
#
#             print("segmentation time: {:.3f}, #roi: {} roi area: {} roi coverage: {:.3f}".format(
#                 time.time() - t, len(rois), area, area / float(img.shape[0] * img.shape[1])))
#             # t = time.time()
#             # segment(proj, img)
#             # print "without ", time.time() - t
#
#             img = new_im
#         except:
#             img = segment(project, img)
#     else:
#         img = segment(project, img)
#
#     return img


if __name__ == '__main__':
    fire.Fire({'segmentation': segmentation,
              'part_segmentation': do_segmentation_part})

