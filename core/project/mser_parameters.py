__author__ = 'flipajs'


class MSERParameters():
    def __init__(self, refresh=None):
        self.max_area = 0.005
        self.min_area = 15
        self.min_margin = 5
        self.gaussian_kernel_std = 0.0
        self.intensity_threshold = 256
        self.min_area_relative = 0.2
        self.region_min_intensity = 256

        if refresh:
            self.__dict__.update(refresh.__dict__)

    def __str__(self):
        s = "MSER parameters:\n"
        s += "max_area: "+str(self.max_area)+"\n"
        s += "min_area: "+str(self.min_area)+"\n"
        s += "min_margin: "+str(self.min_margin)+"\n"
        s += "gaussian_kernel_std: "+str(self.gaussian_kernel_std)+"\n"
        s += "intensity_threshold: "+str(self.intensity_threshold)+"\n"
        s += "min_area_relative: "+str(self.min_area_relative)+"\n"
        s += "region_min_intensity: "+str(self.region_min_intensity)+"\n"

        return s