__author__ = 'fnaiser'

import cv2
import warnings
import time
from core.region.region import Region
from core.region import cyMser
import pickle
from utils.video_manager import get_auto_video_manager
from core.settings import Settings as S_
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
from utils.misc import is_flipajs_pc
from mser_operations import get_region_groups_dict_, margin_filter_dict_, min_intensity_filter_dict_, antlikeness_filter
import numpy as np


class Mser():
    def __init__(self, max_area=0.005, min_margin=5, min_area=5):
        self.mser = cyMser.PyMser()
        self.mser.set_min_margin(min_margin)
        self.mser.set_max_area(max_area)
        self.mser.set_min_size(min_area)

    def process_image(self, img, frame=-1, intensity_threshold=256, prefiltered=False,
                      region_min_intensity=None, intensity_percentile=-1, use_margin_filter=True,
                      use_children_filter=True):

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

        if prefiltered:
            groups = get_region_groups_dict_(regions)
            if use_margin_filter:
                ids = margin_filter_dict_(regions, groups)
            else:
                ids = range(len(regions))

            if region_min_intensity is not None and region_min_intensity < 256:

                # fix minI:
                for r_id in ids:
                    r = regions[r_id]
                    min_i_ = 255

                    if intensity_percentile > 0:
                        dd = []

                    for it in r['rle']:
                        d = img[it['line'], it['col1']:it['col2'] + 1]

                        if intensity_percentile > 0:
                            dd.extend(d)
                        m_ = d.min()

                        min_i_ = min(min_i_, m_)

                    r['minI'] = min_i_
                    if intensity_percentile > 0:
                        r['intensity_percentile'] = np.percentile(dd, intensity_percentile)

                ids = min_intensity_filter_dict_(regions, ids, region_min_intensity, intensity_percentile > 0)

            regions = [Region(regions[id], frame, id) for id in ids]
        else:
            regions = [Region(dr, frame, id) for dr, id in zip(regions, range(len(regions)))]

        if use_children_filter:
            ids = range(len(regions))
            ids = children_filter(regions, ids)

            regions = [regions[i] for i in ids]

        return regions

    def set_max_area_relative(self, max_area_relative):
        self.mser.set_max_area(max_area_relative)


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


def get_msers_(img, project, frame=-1, prefiltered=False):
    """
    Returns msers using MSER algorithm with default settings.

    """
    max_area = project.mser_parameters.max_area
    min_area = project.mser_parameters.min_area
    min_margin = project.mser_parameters.min_margin

    max_area_relative = max_area / float(img.shape[0]*img.shape[1])

    region_min_intensity = project.mser_parameters.region_min_intensity

    intensity_percentile = -1

    try:
        if project.mser_parameters.use_intensity_percentile_threshold:
            intensity_percentile = project.mser_parameters.intensity_percentile
    except:
        pass

    use_margin_filter = False
    if not hasattr(project.mser_parameters, 'use_min_margin_filter') or project.mser_parameters.use_min_margin_filter:
        use_margin_filter = True

    use_children_filter = False
    if project.mser_parameters.use_children_filter:
        use_children_filter = True

    mser = Mser(max_area=max_area_relative, min_margin=min_margin, min_area=min_area)
    return mser.process_image(img,
                              frame,
                              intensity_threshold=project.mser_parameters.intensity_threshold,
                              prefiltered=prefiltered,
                              region_min_intensity=region_min_intensity,
                              intensity_percentile=intensity_percentile,
                              use_margin_filter=use_margin_filter,
                              use_children_filter=use_children_filter
                              )


def ferda_filtered_msers(img, project, frame=-1):
    # if project.mser_parameters.use_children_filter:
    #     m = get_msers_(img, project, frame, prefiltered=True)
        # groups = get_region_groups(m)

        # ids = range(len(m))
        # if not hasattr(project.mser_parameters, 'use_min_margin_filter') or project.mser_parameters.use_min_margin_filter:
        #     ids = margin_filter(m, groups)

        # # min_area = project.stats.area_median * 0.2
        # # ids = area_filter(m, ids, min_area)
        # if project.mser_parameters.use_children_filter:
        #     ids = children_filter(m, ids)

        # if project.stats:
        #     num_before = len(ids)
        #     ids = antlikeness_filter(project.stats.antlikeness_svm, project.solver_parameters.antlikeness_threshold, m, ids)
        #     if len(ids) == 0 and num_before > 0:
        #         warnings.warn("There is something fishy with antlikeness filter. After filtering, there is 0 regions")

        # return [m[id] for id in ids]
    # else:
    return get_msers_(img, project, frame, prefiltered=True)

