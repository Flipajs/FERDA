__author__ = 'fnaiser'

import numpy as np
from reduced import Reduced


class Chunk():
    def __init__(self):
        self.reduced = []
        self.start_t = np.inf
        self.end_t = -1
        self.is_sorted = False
        self.start_n = None
        self.end_n = None

    def __str__(self):
        s = "CHUNK --- start_t: "+str(self.start_t)+" end_t: "+str(self.end_t)+" reduced_len: "+str(len(self.reduced))+"\n"
        return s

    def add_region(self, r):
        # if r.frame_ < self.start_t:
        self.start_t = min(self.start_t, r.frame_)
        self.end_t = max(self.end_t, r.frame_)

        self.reduced.append(Reduced(r))
        self.is_sorted = False

    def merge(self, chunk):
        self.reduced += chunk.reduced
        self.start_t = min(self.start_t, chunk.start_t)
        self.end_t = max(self.end_t, chunk.end_t)

    def remove_from_beginning(self):
        if not self.is_sorted:
            self.reduced = sorted(self.reduced, key=lambda k: k.t)
            self.is_sorted = True

        if not self.reduced:
            return None

        return self.reduced.pop(0)

    def remove_from_end(self):
        if not self.is_sorted:
            self.reduced = sorted(self.reduced, key=lambda k: k.t)
            self.is_sorted = True

        if not self.reduced:
            return None

        return self.reduced.pop()

    def get_reduced_at(self, t):
        if not self.is_sorted:
            self.reduced = sorted(self.reduced, key=lambda k: k.t)
            self.is_sorted = True

        t -= self.start_t
        if 0 <= t < len(self.reduced):
            return self.reduced[t]

        return None

    def length(self):
        return self.end_t - self.start_t
