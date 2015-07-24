__author__ = 'fnaiser'

import numpy as np
from reduced import Reduced
from utils.constants import EDGE_CONFIRMED
from core.log import LogCategories, ActionNames
from core.settings import Settings as S_


class Chunk:
    def __init__(self, start_n=None, end_n=None, solver=None):
        self.reduced = []
        self.is_sorted = False
        self.start_n = None
        self.end_n = None

        if solver:
            self.set_start(start_n, solver)
            self.set_end(end_n, solver)
            self.simple_reconnect_(solver)

    def __str__(self):
        s = "CHUNK --- start_t: "+str(self.start_n.frame_)+" end_t: "+str(self.end_n.frame_)+" reduced_len: "+str(len(self.reduced))+"\n"
        return s

    def start_t(self):
        return self.start_n.frame_

    def end_t(self):
        return self.end_n.frame_

    def set_start(self, n, solver):
        solver.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.CHUNK_SET_START, {'chunk': self, 'old_start_n': self.start_n, 'new_start_n': n})
        self.start_n = n

    def set_end(self, n, solver):
        solver.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.CHUNK_SET_END, {'chunk': self, 'old_end_n': self.end_n, 'new_end_n': n})
        self.end_n = n

    def append_left(self, r, solver, undo_action=False):
        if r.frame_ + 1 != self.start_t():
            raise Exception("DISCONTINUITY in chunk.py/append_left")

        is_ch, t_reversed, ch2 = solver.is_chunk(r)

        first = self.start_n

        self.add_to_reduced_(self.start_n, solver, 0)
        self.set_start(r, solver)

        if not undo_action:
            solver.remove_node(first, False)

        if not undo_action:
            self.chunk_reconnect_(solver)

        # r was already in chunk
        if is_ch:
            ch2.merge(self, solver)

    def append_right(self, r, solver, undo_action=False):
        if r.frame_ != self.end_t() + 1:
            raise Exception("DISCONTINUITY in chunk.py/append_right")

        is_ch, t_reversed, ch2 = solver.is_chunk(r)

        last = self.end_n

        self.add_to_reduced_(self.end_n, solver)
        self.set_end(r, solver)

        if not undo_action:
            solver.remove_node(last, False)

        if not undo_action:
            self.chunk_reconnect_(solver)

        # r was already in chunk
        if is_ch:
            self.merge(ch2, solver)

    def chunk_reconnect_(self, solver):
        solver.add_edge(self.start_n, self.end_n, type=EDGE_CONFIRMED, chunk_ref=self, score=1.0)

    def simple_reconnect_(self, solver):
        solver.add_edge(self.start_n, self.end_n, type=EDGE_CONFIRMED, score=1.0)

    def add_to_reduced_(self, r, solver, i=-1):
        it = Reduced(r)
        try:
            if r.is_virtual:
                # in this case, save whole region, it is much easier...
                it = r
        except:
            pass

        if i == -1:
            self.reduced.append(it)
        else:
            self.reduced.insert(i, it)

        solver.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.CHUNK_ADD_TO_REDUCED, {'chunk': self, 'node': r, 'index': i})
        # self.is_sorted = False

    def remove_from_reduced_(self, i, solver):
        if i < 0:
            r = self.reduced.pop()
        else:
            r = self.reduced.pop(i)
        solver.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.CHUNK_REMOVE_FROM_REDUCED, {'chunk': self, 'node': r, 'index': i})
        return r

    def merge(self, second_chunk, solver, undo_action=False):
        if self.start_t() > second_chunk.start_t():
            second_chunk.merge(self)
            return

        if not undo_action:
            solver.remove_node(self.end_n, False)

        self.add_to_reduced_(self.end_n, solver)

        if S_.general.log_graph_edits:
            while second_chunk.reduced:
                r = second_chunk.remove_from_reduced_(0, solver)
                self.add_to_reduced_(r, solver)
        else:
            self.reduced += second_chunk.reduced
            second_chunk.reduced = []

        # self.is_sorted = False
        self.set_end(second_chunk.end_n, solver)

        if not undo_action:
            self.chunk_reconnect_(solver)

    def first(self):
        if self.start_n:
            return self.start_n
        elif self.end_n:
            return self.end_n

        return None

    def last(self):
        if self.end_n:
            return self.end_n
        elif self.start_n:
            return self.start_n

        return None

    def pop_first(self, solver, reconstructed=None, undo_action=False):
        if not self.reduced:
            if self.start_n:
                first = self.start_n
                self.set_start(None, solver)
                return first
            elif self.end_n:
                first = self.end_n
                self.set_end(None, solver)
                return first
            else:
                return None

        first = self.start_n

        popped = self.remove_from_reduced_(0, solver)
        if reconstructed is None:
            reconstructed = self.reconstruct(popped, solver.project)

        if not undo_action:
            solver.add_node(reconstructed)

        self.set_start(reconstructed, solver)

        if not undo_action:
            solver.remove_edge(first, self.end_n)
            prev_nodes, _, _ = solver.get_regions_around(reconstructed.frame_)
            solver.add_edges_(prev_nodes, [reconstructed])

        if not undo_action:
            if self.reduced:
                self.chunk_reconnect_(solver)
            else:
                # it is not a chunk anymore
                self.simple_reconnect_(solver)

        return first

    def pop_last(self, solver, reconstructed=None, undo_action=False):
        if not self.reduced:
            if self.end_n:
                last = self.end_n
                self.set_end(None, solver)
                return last
            elif self.start_n:
                last = self.start_n
                self.set_start(None, solver)
                return last
            else:
                return None

        last = self.end_n

        popped = self.reduced.pop(len(self.reduced)-1)
        if reconstructed is None:
            reconstructed = self.reconstruct(popped, solver.project)

        if not undo_action:
            solver.add_node(reconstructed)

        self.end_n = reconstructed

        if not undo_action:
            try:
                solver.remove_edge(self.start_n, last)
            except:
                pass

            _, _, next_nodes = solver.get_regions_around(reconstructed.frame_)
            solver.add_edges_([reconstructed], next_nodes)

            if self.reduced:
                self.chunk_reconnect_(solver)
            else:
                self.simple_reconnect_(solver)

        return last

    def if_not_sorted_sort_(self):
        if not self.is_sorted:
            self.reduced = sorted(self.reduced, key=lambda k: k.frame_)
            self.is_sorted = True

    def get_centroid_in_time(self, t):
        if self.start_t() <= t <= self.end_t():
            if t == self.start_t():
                return self.start_n.centroid()
            elif t == self.end_t():
                return self.end_n.centroid()
            else:
                return self.reduced[t-self.start_t()-1].centroid()

        return None

    def get_reduced_at(self, t):
        self.if_not_sorted_sort_()

        t -= self.start_t() + 1
        if 0 <= t < len(self.reduced):
            return self.reduced[t]

        return None

    def length(self):
        if self.start_n and self.end_n:
            return self.end_t() - self.start_t() + 1
        elif self.start_n:
            return 1
        elif self.end_n:
            return 1

        return 0

    @ staticmethod
    def reconstruct(r, project):
        if isinstance(r, Reduced):
            return r.reconstruct(project)

        return r

    def split_at_t(self, t, solver, undo_action=False):
        # spins off and reconstructs region at frame t and if chunk is long enough it will separate it into 2 chunks
        if self.start_t() < t < self.end_t():
            self.if_not_sorted_sort_()

            node_t = self.get_reduced_at(t)
            pos = self.reduced.index(node_t)
            node_t = self.reconstruct(node_t, solver.project)

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
                ch2 = Chunk()
                ch2.start_n = self.reduced.pop(pos+1)
                ch2.start_n = self.reconstruct(ch2.start_n, solver.project)
                ch2.end_n = self.end_n

                while len(self.reduced) > pos+1:
                    ch2.reduced.append(self.reduced.pop(pos+1))

                # ----- adjust first chunk -----
                new_end_n = self.reduced.pop(pos)
                reconstructed = self.reconstruct(new_end_n, solver.project)
                self.end_n = reconstructed

                # ----- reconnect with neighbours ----
                # solver.add_node(node_t)
                if not undo_action:
                    solver.add_node(self.end_n)
                    solver.add_node(ch2.start_n)

                    t_minus, t, t_plus = solver.get_regions_around(self.end_n.frame_)
                    solver.add_edges_(t, t_plus)
                # solver.add_edges_(t, t_plus)

                # ----- test if we still have 2 chunks and reconnect first and last ---
                if not undo_action:
                    if self.reduced:
                        self.chunk_reconnect_(solver)
                    else:
                        self.simple_reconnect_(solver)

                    if ch2.reduced:
                        ch2.chunk_reconnect_(solver)
                    else:
                        ch2.simple_reconnect_(solver)

        else:
            raise Exception("t is out of range of this chunk in chunk.py/split_at_t")











