__author__ = 'fnaiser'


class Reduced():
    def __init__(self, region):
        self.centroid = region.centroid()
        self.t = region.frame_
        self.mser_id = region.id_
