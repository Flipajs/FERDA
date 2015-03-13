__author__ = 'fnaiser'

import cv2

from core.region import cyMser
from region import Region
import pickle
from utils.video_manager import get_auto_video_manager
from core.settings import Settings as S_

class Mser():
    def __init__(self, max_area=0.005, min_margin=5, min_area=5):
        self.mser = cyMser.PyMser()
        self.mser.set_min_margin(min_margin)
        self.mser.set_max_area(max_area)
        self.mser.set_min_size(min_area)

    def process_image(self, img, intensity_threshold=256):
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

        regions = [Region(dr) for dr in regions]

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
    return get_all_msers(frame_number, project.video_paths, project.working_directory)

def get_all_msers(frame_number, video_paths, working_dir):
    """
    Tries to use cached MSERs, if cache is empty, MSERs are computed and if caching is allowed, then stored.
    Returns all regions

    """

    if S_.cache.mser:
        try:
            with open(working_dir+'/mser/'+str(frame_number)+'.pkl', 'rb') as f:
                msers = pickle.load(f)

            return msers
        except IOError:
            vid = get_auto_video_manager(video_paths)
            msers = get_msers_(vid.seek_frame(frame_number))

            try:
                with open(working_dir+'/mser/'+str(frame_number)+'.pkl', 'wb') as f:
                    pickle.dump(msers, f)
            except IOError:
                pass

            return msers

    else:
        vid = get_auto_video_manager(video_paths)
        return get_msers_(vid.seek_frame(frame_number))


def get_msers_(img):
    """
    Returns msers using MSER algorithm with default settings.

    """
    mser = Mser(max_area=S_.mser.max_area, min_margin=S_.mser.min_margin, min_area=S_.mser.min_area)
    return mser.process_image(img)