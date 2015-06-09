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


def get_length_of_longest_chunk(solver, cc):
    longest1 = -1
    longest2 = -1
    for n in cc.regions_t1:
        is_ch, _, ch = solver.is_chunk(n)
        if is_ch:
            longest1 = max(longest1, ch.length())

    for n in cc.regions_t2:
        is_ch, _, ch = solver.is_chunk(n)
        if is_ch:
            longest2 = max(longest2, ch.length())

    return max(0, longest1) + max(0, longest2)