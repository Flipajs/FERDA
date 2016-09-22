__author__ = 'flipajs'
import numpy as np


class SolverParameters():
    def __init__(self, refresh=None):
        self.max_edge_distance_in_ant_length = 1.0
        self.antlikeness_threshold = 0.1
        self.certainty_threshold = 0.5
        self.global_view_min_chunk_len = 0
        self.frames_in_row = 100

        # Zernike descriptor parameters
        self.zernike_thresh = 0.1
        self.zernike_plus = 0.2
        self.zernike_minus = -0.5
        self.zernike_sigma = 0.2
        self.zernike_radius = 21
        self.zernike_norm = np.array([30, 100])

        self.chunk_id = -1

        if refresh:
            self.__dict__.update(refresh.__dict__)

    def new_chunk_id(self):
        self.chunk_id += 1
        return self.chunk_id

    def use_emd_for_split_merge_detection(self):
        if hasattr(self, 'use_emd_for_split_merge_detection'):
            return self.use_emd_for_split_merge_detection

        return True

    def use_colony_split_merge_relaxation(self):
        if hasattr(self, 'use_colony_split_merge_relaxation'):
            return self.use_colony_split_merge_relaxation

        return False