from __future__ import unicode_literals
from builtins import object
__author__ = 'flipajs'


class OtherParameters(object):
    def __init__(self, initial_data=None):
        self.img_subsample_factor = 1.0
        self.use_only_red_channel = False
        self.store_area_info = False
        self.segmentation_use_roi_prediction_optimisation = False
        self.segmentation_prediction_optimisation_border = 25
        self.segmentation_full_segmentation_refresh_in = 25

        if initial_data:
            self.__dict__.update(initial_data)
