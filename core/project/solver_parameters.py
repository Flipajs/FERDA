__author__ = 'flipajs'


class SolverParameters():
    def __init__(self):
        self.max_edge_distance_in_ant_length = 2.5
        self.antlikeness_threshold = 0.1
        self.certainty_threshold = 0.3
        self.global_view_min_chunk_len = 0