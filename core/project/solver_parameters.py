__author__ = 'flipajs'


class SolverParameters():
    def __init__(self, refresh=None):
        self.max_edge_distance_in_ant_length = 1.0
        self.antlikeness_threshold = 0.1
        self.certainty_threshold = 0.5
        self.global_view_min_chunk_len = 0
        self.frames_in_row = 100

        self.chunk_id = -1

        if refresh:
            self.__dict__.update(refresh.__dict__)

    def new_chunk_id(self):
        self.chunk_id += 1
        return self.chunk_id