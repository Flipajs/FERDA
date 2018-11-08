from __future__ import unicode_literals
from builtins import str
from builtins import object
__author__ = 'flipajs'


class MSERParameters(object):
    def __init__(self, initial_data=None):
        self.max_area = 50000
        self.min_area = 50
        self.min_margin = 5
        self.use_min_margin_filter = True
        self.gaussian_kernel_std = 0.0
        self.intensity_threshold = 256
        self.region_min_intensity = 256
        self.use_children_filter = True
        self.intensity_percentile = 10
        # if 0, ignore... we suggest 0.1 as it will ignore all thick regions on arena borders
        self.area_roi_ratio_threshold = 0
        self.use_intensity_percentile_threshold = False

        if initial_data:
            self.__dict__.update(initial_data)

    def __str__(self):
        s = "MSER parameters:\n"
        s += "max_area: "+str(self.max_area)+"\n"
        s += "min_area: "+str(self.min_area)+"\n"
        s += "min_margin: "+str(self.min_margin)+"\n"
        s += "gaussian_kernel_std: "+str(self.gaussian_kernel_std)+"\n"
        s += "intensity_threshold: "+str(self.intensity_threshold)+"\n"
        s += "region_min_intensity: "+str(self.region_min_intensity)+"\n"
        s += "use_children_filter: "+str(self.use_children_filter)+"\n"
        return s