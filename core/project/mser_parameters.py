__author__ = 'flipajs'


class MSERParameters():
    def __init__(self, refresh=None):
        self.max_area = 0.005
        self.min_area = 5
        self.min_margin = 5
        self.gaussian_kernel_std = 0.0
        self.intensity_threshold = 256
        self.min_area_relative = 0.2

        if refresh:
            self.__dict__.update(refresh.__dict__)