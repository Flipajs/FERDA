__author__ = 'fnaiser'

import numpy as np
from reduced import Reduced
from utils.constants import EDGE_CONFIRMED
from core.log import LogCategories, ActionNames
from core.settings import Settings as S_
from core.region.region import Region


class Chunk:
    def __init__(self, vertices, id_, graph_manager, region_manager, color=None):
        self.id_ = id_
        self.vertices_ = vertices
        self.color = color
        self.statistics = {}
        self.gm = graph_manager
        self.rm = region_manager

        self.chunk_reconnect_(graph_manager)

    def __str__(self):
        s = "CHUNK --- start_f: "+str(self.start_frame())+" end_f: "+str(self.end_frame())+" length: "+str(len(self.vertices_))+"\n"
        return s

    def append_left(self, vertex, undo_action=False):
        region = self.rm[vertex]
        if region.frame() + 1 != self.start_frame():
            print "DISCONTINUITY in chunk.py/append_left", region.frame(), self.start_frame(), region, self.start_vertex()
            raise Exception("DISCONTINUITY in chunk.py/append_left")

        first = self.start_n

        ch2, _ = self.gm.is_chunk(vertex)
        if ch2:
            ch2.merge(self, undo_action=undo_action)
        else:
            self.vertices_.insert(0, vertex)

        if not undo_action:
            self.gm.remove_node(first, False)
            self.chunk_reconnect_()

    def append_right(self, vertex, solver, undo_action=False):
        region = self.rm[vertex]
        if region.frame() != self.end_t() + 1:
            print "DISCONTINUITY in chunk.py/append_right", region.frame(), self.end_frame(), region, self.end_vertex()
            raise Exception("DISCONTINUITY in chunk.py/append_right")

        last = self.end_vertex()

        ch2, _ = self.gm.is_chunk(vertex)
        if ch2:
            self.merge(ch2, undo_action=undo_action)
        else:
            self.vertices_.append(vertex)

        if not undo_action:
            self.gm.remove_node(last, False)
            self.chunk_reconnect_()

    def pop_first(self, undo_action=False):
        first = self.vertices_.pop(0)
        new_start = self.start_vertex()

        if not undo_action:
            self.gm.add_vertex(new_start)

        if not undo_action:
            self.gm.remove_edge(first, self.end_vertex())
            prev_nodes, _, _ = self.gm.get_vertices_around_t(self.rm[new_start].frame())
            self.gm.add_edges_(prev_nodes, [new_start])

        if not undo_action:
            if len(self.vertices_) > 1:
                self.chunk_reconnect_()

        return first

    def pop_last(self, undo_action=False):
        last = self.vertices_.pop()
        new_end = self.end_vertex()

        if not undo_action:
            self.gm.add_vertex(new_end)

        if not undo_action:
            self.gm.remove_edge(self.start_vertex(), last)

            _, _, next_nodes = self.gm.get_vertices_around_t(self.rm[new_end].frame())
            self.gm.add_edges_([new_end], next_nodes)

            self.chunk_reconnect_()

        return last

    def id(self):
        return self.id_

    def end_vertex(self):
        return self.vertices_[-1]

    def start_vertex(self):
        return self.vertices_[0]

    def start_frame(self):
        return self.rm[self.start_vertex()].frame()

    def end_frame(self):
        return self.rm[self.end_vertex()].frame()

    def length(self):
        return len(self.vertices_)



    def set_start(self, n, solver):
        solver.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.CHUNK_SET_START, {'chunk': self, 'old_start_n': self.start_n, 'new_start_n': n})
        self.start_n = n

    def set_end(self, n, solver):
        solver.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.CHUNK_SET_END, {'chunk': self, 'old_end_n': self.end_n, 'new_end_n': n})
        self.end_n = n

    def chunk_reconnect_(self):
        self.gm.add_edge(self.start_vertex(), self.end_vertex())
        self.gm.g[self.start_vertex()]['chunk_start'] = self.id()
        self.gm.g[self.end_vertex()]['chunk_end'] = self.id()



    def merge(self, second_chunk, solver, undo_action=False):
        if self.start_t() > second_chunk.start_t():
            second_chunk.merge(self, solver)
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

    def merge_and_interpolate(self, second_chunk, solver, undo_action=False):
        if self.end_t() > second_chunk.start_t():
            print self.end_t(), second_chunk.start_t()
            second_chunk.merge_and_interpolate(self, solver, undo_action)
            return

        gap_len = second_chunk.start_t() - self.end_t() - 1
        if gap_len > 0:
            c_diff_part = (second_chunk.start_n.centroid() - self.end_n.centroid()) / gap_len

            i = 1
            for f in range(self.end_t()+1, second_chunk.start_t()):
                r = Region(frame=f)
                r.is_virtual = True
                c = self.end_n.centroid() + np.array(c_diff_part * i)
                r.centroid_ = c.copy()

                # TODO: log...
                self.reduced.append(Reduced(r))

                i += 1

        n1 = self.end_n
        self.merge(second_chunk, solver, undo_action)

        n = second_chunk.start_n
        self.add_to_reduced_(n, solver)
        solver.remove_node(n, False)

        self.is_sorted = False
        self.if_not_sorted_sort_()



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
                ch2 = Chunk(store_area=self.store_area, id=solver.project.solver_parameters.new_chunk_id())
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

                    t_minus, t, t_plus = solver.get_vertices_around_t(self.end_n.frame_)
                    solver.add_edges_(t, t_plus)
                # solver.add_edges_(t, t_plus)

                # ----- test if we still have 2 chunks and reconnect first and last ---
                if not undo_action:
                    self.chunk_reconnect_(solver)
                    ch2.chunk_reconnect_(solver)

        else:
            raise Exception("t is out of range of this chunk in chunk.py/split_at_t")

    def is_virtual_in_time(self, t):
        red = self.get_reduced_at(t)
        if red is not None:
            if isinstance(red, Reduced):
                return False
            else:
                return red.is_virtual

        if self.start_t() == t:
            return self.start_n.is_virtual
        elif self.end_t() == t:
            return self.end_n.is_virtual

