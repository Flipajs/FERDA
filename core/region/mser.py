__author__ = 'fnaiser'

import cv2

from core.region.region import Region
from core.region import cyMser
import pickle
from utils.video_manager import get_auto_video_manager
from core.settings import Settings as S_
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
import time
from utils.misc import is_flipajs_pc


class Mser():
    def __init__(self, max_area=0.005, min_margin=5, min_area=5):
        self.mser = cyMser.PyMser()
        self.mser.set_min_margin(min_margin)
        self.mser.set_max_area(max_area)
        self.mser.set_min_size(min_area)

    def process_image(self, img, frame=-1, intensity_threshold=256):
        # if is_flipajs_pc():
        #     intensity_threshold = 200

        if len(img.shape) > 2:
            if img.shape[2] > 1:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img[:, :, 0]
        else:
            gray = img

        if intensity_threshold > 256:
            intensity_threshold = 256

        self.mser.process_image(gray, intensity_threshold)
        regions = self.mser.get_regions()
        regions = [Region(dr, frame, id) for dr, id in zip(regions, range(len(regions)))]

        return regions

    def set_max_area(self, max_area):
        self.mser.set_max_area(max_area)


def get_mser(frame_number, id, project):
    """
    Tries to use cached MSERs, if cache is empty, MSERs are computed and if caching is allowed, then stored.
    Returns region based on id
    :param frame_number:
    :param id:
    :param project:
    :return:
    """
    return get_mser(frame_number, id, project.video_paths, project.working_directory)

def get_mser(frame_number, id, video_paths, working_dir):
    """
    Tries to use cached MSERs, if cache is empty, MSERs are computed and if caching is allowed, then stored.
    Returns region based on id
    """

    return get_all_msers(frame_number, video_paths, working_dir)[id]

def get_all_msers(frame_number, project):
    """
    Tries to use cached MSERs, if cache is empty, MSERs are computed and if caching is allowed, then stored.
    Returns all regions
    """

    if S_.cache.mser:
        try:
            with open(project.working_directory+'/mser/'+str(frame_number)+'.pkl', 'rb') as f:
                msers = pickle.load(f)

            return msers
        except IOError:
            vid = get_auto_video_manager(project)
            msers = get_msers_(vid.seek_frame(frame_number), frame_number)

            try:
                with open(project.working_directory+'/mser/'+str(frame_number)+'.pkl', 'wb') as f:
                    pickle.dump(msers, f)
            except IOError:
                pass

            return msers

    else:
        vid = get_auto_video_manager(project)
        return get_msers_(vid.seek_frame(frame_number))


def get_msers_(img, project, frame=-1):
    """
    Returns msers using MSER algorithm with default settings.

    """
    max_area = project.mser_parameters.max_area
    min_area = project.mser_parameters.min_area
    min_margin = project.mser_parameters.min_margin

    mser = Mser(max_area=max_area, min_margin=min_margin, min_area=min_area)
    return mser.process_image(img, frame, intensity_threshold=project.mser_parameters.intensity_threshold)


def ferda_filtered_msers(img, project, frame=-1):
    m = get_msers_(img, project, frame)
    groups = get_region_groups(m)
    ids = margin_filter(m, groups)
    # min_area = project.stats.area_median * 0.2
    # ids = area_filter(m, ids, min_area)
    ids = children_filter(m, ids)

    return [m[id] for id in ids]