__author__ = 'fnaiser'

import numpy as np
from reduced import Reduced
from utils.constants import EDGE_CONFIRMED


class Chunk:
    def __init__(self, start_n=None, middle_n=None, end_n=None):
        self.reduced = []
        self.is_sorted = False
        self.start_n = start_n
        self.end_n = end_n
        if middle_n:
            self.add_to_reduced_(middle_n)

    def __str__(self):
        s = "CHUNK --- start_t: "+str(self.start_n.frame_)+" end_t: "+str(self.end_n.frame_)+" reduced_len: "+str(len(self.reduced))+"\n"
        return s

    def start_t(self):
        return self.start_n.frame_

    def end_t(self):
        return self.end_n.frame_

    def append_left(self, r, solver):
        if r.frame_ + 1 != self.start_t():
            raise Exception("DISCONTINUITY in chunk.py/append_left")

        solver.remove_node(self.start_n)
        self.add_to_reduced_(self.start_n)
        self.start_n = r

        self.chunk_reconnect_(solver)

    def append_right(self, r, solver):
        if r.frame_ != self.end_t() + 1:
            raise Exception("DISCONTINUITY in chunk.py/append_right")

        solver.remove_node(self.end_n)
        self.add_to_reduced_(self.end_n)
        self.end_n = r

        self.chunk_reconnect_(solver)

    def chunk_reconnect_(self, solver):
        solver.add_edge(self.start_n, self.end_n, type=EDGE_CONFIRMED, chunk_ref=self, score=1.0)

    def simple_reconnect(self, solver):
        solver.add_edge(self.start_n, self.end_n, type=EDGE_CONFIRMED, score=1.0)

    def add_to_reduced_(self, r):
        it = Reduced(r)

        try:
            if r.is_virtual:
                # in this case, save whole region, it is much easier...
                it = r
        except:
            pass

        self.reduced.append(it)
        self.is_sorted = False

    def merge(self, second_chunk, solver):
        if self.start_t() > second_chunk.start_t():
            second_chunk.merge(self)
            return

        solver.remove_node(self.end_n)
        solver.remove_node(second_chunk.start_n)

        self.add_to_reduced_(self.end_n)
        self.add_to_reduced_(second_chunk.start_n)
        self.reduced += second_chunk.reduced
        self.end_n = second_chunk.end_n

        self.chunk_reconnect_(solver)

    def pop_first(self, solver):
        first = self.start_n

        popped = self.reduced.pop(0)

        reconstructed = self.reconstruct(popped, solver.project)

        solver.add_node(reconstructed)
        self.start_n = reconstructed
        prev_nodes, _, _ = solver.get_regions_around(reconstructed.frame_)
        solver.add_edges(prev_nodes, [reconstructed])

        if self.reduced:
            self.chunk_reconnect_(solver)
        else:
            # it is not a chunk anymore
            self.simple_reconnect_(solver)

        return first

    def pop_last(self, solver):
        last = self.end_n

        popped = self.reduced.pop()

        reconstructed = self.reconstruct(popped, solver.project)

        solver.add_node(reconstructed)
        self.end_n = reconstructed
        _, _, next_nodes = solver.get_regions_around(reconstructed.frame_)
        solver.add_edges([reconstructed], next_nodes)

        if self.reduced:
            self.chunk_reconnect_(solver)
        else:
            self.simple_reconnect(solver)

    def if_not_sorted_sort_(self):
        if not self.is_sorted:
            self.reduced = sorted(self.reduced, key=lambda k: k.frame_)
            self.is_sorted = True

    def get_reduced_at(self, t):
        self.if_not_sorted_sort_()

        t -= self.start_t
        if 0 <= t < len(self.reduced):
            return self.reduced[t]

        return None

    def length(self):
        return self.end_t() - self.start_t() + 1

    @ staticmethod
    def reconstruct(r, project):
        if isinstance(r, Reduced):
            return r.reconstruct(project)

        return r

    def split_at_t(self, t, solver):
        # spins off and reconstructs region at frame t and if chunk is long enough it will separate it into 2 chunks
        if self.start_t() < t < self.end_t():
            self.if_not_sorted_sort_()

            node_t = self.get_reduced_at(t)
            pos = self.reduced.index(node_t)

            if pos == 0:
                # it is first in reduced, to spin it off two pop_first are enough
                self.pop_first(solver)
                self.pop_first(solver)
            elif pos == len(self.reduced)-1:
                # it is last in reduced, to spin it off two pop_last are enough
                self.pop_last(solver)
                self.pop_last(solver)
            else:
                # ----- building second chunk -----

                # remove node_t we already have
                self.reduced.pop(pos)
                ch2 = Chunk()
                ch2.start_n = self.reduced.pop(pos)
                ch2.end_n = self.end

                while len(self.reduced) >= pos:
                    ch2.reduced.append(self.reduced.pop(id))

                # ----- adjust first chunk -----
                new_end_n = self.reduced.pop(pos-1)
                reconstructed = self.reconstruct(new_end_n)
                self.end_n = reconstructed

                # ----- reconnect with neighbours ----
                solver.add_node(node_t)
                solver.add_node(self.end_n)
                solver.add_node(ch2.start_n)

                t_minus, t, t_plus = solver.get_regions_around(node_t)
                solver.add_edges_(t_minus, t)
                solver.add_edges_(t, t_plus)

                # ----- test if we still have 2 chunks and reconnect first and last ---
                if self.reduced:
                    self.chunk_reconnect_(solver)
                else:
                    self.simple_reconnect(solver)

                if ch2.reduced:
                    ch2.chunk_reconnect_(solver)
                else:
                    ch2.simple_reconnect(solver)

        else:
            raise Exception("t is out of range of this chunk in chunk.py/split_at_t")











