from __future__ import print_function
import os
from os.path import join
import numpy as np
import fire
from joblib import Parallel, delayed
import logging
from core.region.mser import get_filtered_regions
from core.project.project import Project, set_managers
from config import config
from utils.img import prepare_for_segmentation
from core.region.region_manager import RegionManager
from utils.video_manager import get_auto_video_manager

logger = logging.getLogger(__name__)


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


def segmentation(project):
    """
    Segment regions in all frames.

    :param project_dir: project directory
    :return: int, number of parts
    """
    # import ipdb; ipdb.set_trace()
    vid = get_auto_video_manager(project)
    frame_num = int(vid.total_frame_count())
    frames_in_row = config['segmentation']['frames_in_row']
    Parallel(n_jobs=config['general']['n_jobs'], verbose=10)\
        (delayed(do_segmentation_part)(project.working_directory, i, frame_start)
         for i, frame_start in enumerate(range(0, frame_num, frames_in_row)))  # frame_num


def do_segmentation_part(project_dir, part_id, frame_start, frame_done_func=None):
    # check if part was computed before
    temp_path = os.path.join(project_dir, 'temp/{}'.format(part_id))

    if os.path.isfile(join(temp_path, 'regions.csv')) and os.path.isfile(join(temp_path, 'regions.h5')):
        print('Part {} already processed.'.format(part_id))
        return
    p = Project.from_dir(project_dir, tracklets_optional=True)
    p.reset_managers()  # clean possible previously computed data, force RegionManager to temporary hdf5

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
            p.rm.append(r)  # RegionManager.append takes care about id
        if frame_done_func is not None:
            frame_done_func()

    p.save(temp_path)
    logger.debug('Part number {}, regions found: {}'.format(part_id, len(p.rm)))


if __name__ == '__main__':
    fire.Fire({'segmentation': segmentation,
              'part_segmentation': do_segmentation_part})

