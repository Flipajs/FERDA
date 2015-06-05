__author__ = 'fnaiser'


class Configuration():
    def __init__(self, id, regions_t1, regions_t2, certainty, confs, scores):
        self.regions_t1 = regions_t1
        self.regions_t2 = regions_t2

        self.certainty = certainty
        self.configurations = confs
        self.scores = scores
        self.id = id
        self.t = regions_t1[0].frame_
