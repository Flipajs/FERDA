__author__ = 'flipajs'


class SolverParameters(object):
    def __init__(self, initial_data=None):
        self.max_edge_distance_in_ant_length = 1.0
        self.antlikeness_threshold = 0.1
        self.certainty_threshold = .01
        self.global_view_min_chunk_len = 0
        self.graph_max_distance = None

        self.chunk_id = -1

        self.use_emd_for_split_merge_detection_ = True
        self.use_colony_split_merge_relaxation_ = False

        if initial_data:
            self.__dict__.update(initial_data)

    def use_emd_for_split_merge_detection(self):
        if hasattr(self, 'use_emd_for_split_merge_detection_'):
            return self.use_emd_for_split_merge_detection_

        return True

    def use_colony_split_merge_relaxation(self):
        if hasattr(self, 'use_colony_split_merge_relaxation_'):
            return self.use_colony_split_merge_relaxation_
        return False