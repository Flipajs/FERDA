__author__ = 'fnaiser'

import numpy as np
from reduced import Reduced
from utils.constants import EDGE_CONFIRMED
from core.log import LogCategories, ActionNames
from core.settings import Settings as S_
from core.region.region import Region


class Chunk:
    def __init__(self, nodes, id_, graph_manager, region_manager, color=None):
        self.id_ = id_
        self.nodes_ = nodes
        self.color = color
        self.statistics = {}
        self.gm = graph_manager
        self.rm = region_manager

        self.chunk_reconnect_(graph_manager)

    def __str__(self):
        s = "CHUNK --- start_f: "+str(self.start_frame())+" end_f: "+str(self.end_frame())+" length: "+str(len(self.nodes_))+"\n"
        return s

    def append_left(self, node, undo_action=False):
        region = self.rm[node]
        if region.frame() + 1 != self.start_frame():
            print "DISCONTINUITY in chunk.py/append_left", region.frame(), self.start_frame(), region, self.start_node()
            raise Exception("DISCONTINUITY in chunk.py/append_left")

        first = self.start_n

        ch2, _ = self.gm.is_chunk(node)
        if ch2:
            ch2.merge(self, undo_action=undo_action)
        else:
            self.nodes_.insert(0, node)

        if not undo_action:
            self.gm.remove_node(first, False)
            self.chunk_reconnect_()

    def append_right(self, node, solver, undo_action=False):
        region = self.rm[node]
        if region.frame() != self.end_t() + 1:
            print "DISCONTINUITY in chunk.py/append_right", region.frame(), self.end_frame(), region, self.end_node()
            raise Exception("DISCONTINUITY in chunk.py/append_right")

        last = self.end_node()

        ch2, _ = self.gm.is_chunk(node)
        if ch2:
            self.merge(ch2, undo_action=undo_action)
        else:
            self.nodes_.append(node)

        if not undo_action:
            self.gm.remove_node(last, False)
            self.chunk_reconnect_()

    def pop_first(self, undo_action=False):
        first = self.nodes_.pop(0)
        new_start = self.start_node()

        if not undo_action:
            self.gm.add_vertex(new_start)

        if not undo_action:
            self.gm.remove_edge(first, self.end_node())
            prev_nodes, _, _ = self.gm.get_vertices_around_t(self.rm[new_start].frame())
            self.gm.add_edges_(prev_nodes, [new_start])

        if not undo_action:
            if len(self.nodes_) > 1:
                self.chunk_reconnect_()

        return first

    def pop_last(self, undo_action=False):
        last = self.nodes_.pop()
        new_end = self.end_node()

        if not undo_action:
            self.gm.add_vertex(new_end)

        if not undo_action:
            self.gm.remove_edge(self.start_node(), last)

            _, _, next_nodes = self.gm.get_vertices_around_t(self.rm[new_end].frame())
            self.gm.add_edges_([new_end], next_nodes)

            self.chunk_reconnect_()

        return last

    def merge(self, ch2, undo_action=False):
        """
        |ch1.start| ... |ch1.end|   |ch2.start|... |ch2.end|
        -> |ch1.start| ... |ch2.end|

        :param second_chunk:
        :param undo_action:
        :return:
        """
        if self.start_frame() > ch2.start_frame():
            ch2.merge(self)
            return

        ch1end = self.end_node()
        ch2start = ch2.start_node()

        if not undo_action:
            self.gm.remove_vertex(ch1end)
            self.gm.remove_vertex(ch2start)

        self.nodes_.extend(ch2.vertices_)

        if not undo_action:
            self.chunk_reconnect_()

    def merge_and_interpolate(self, ch2, undo_action=False):
        if self.end_frame() > ch2.start_frame():
            ch2.merge_and_interpolate(self, undo_action=undo_action)
            return

        gap_len = ch2.start_frame() - self.end_frame() - 1
        if gap_len > 0:
            ch2start_region = self.rm[ch2.start_node()]
            ch1end_region = self.rm[self.end_node()]

            c_diff_part = (ch2start_region.centroid() - ch1end_region.centroid()) / gap_len

            i = 1
            for f in range(self.end_frame()+1, ch2.start_frame()):
                r = Region(frame=f)
                r.is_virtual = True
                c = ch1end_region.centroid() + np.array(c_diff_part * i)
                r.centroid_ = c.copy()

                # TODO: log...
                node = self.gm.add_vertex(r)
                self.append_right(node)

                i += 1

        self.merge(ch2, undo_action)

    def id(self):
        return self.id_

    def end_node(self):
        return self.nodes_[-1]

    def start_node(self):
        return self.nodes_[0]

    def start_frame(self):
        return self.rm[self.start_node()].frame()

    def end_frame(self):
        return self.rm[self.end_node()].frame()

    def length(self):
        return len(self.nodes_)

    def chunk_reconnect_(self):
        self.gm.add_edge(self.start_node(), self.end_node())
        self.gm.g[self.start_node()]['chunk_start_id'] = self.id()
        self.gm.g[self.end_node()]['chunk_end_id'] = self.id()



    # def first(self):
    #     if self.start_n:
    #         return self.start_n
    #     elif self.end_n:
    #         return self.end_n
    #
    #     return None
    #
    # def last(self):
    #     if self.end_n:
    #         return self.end_n
    #     elif self.start_n:
    #         return self.start_n
    #
    #     return None
    #
    # def get_centroid_in_time(self, t):
    #     if self.start_t() <= t <= self.end_t():
    #         if t == self.start_t():
    #             return self.start_n.centroid()
    #         elif t == self.end_t():
    #             return self.end_n.centroid()
    #         else:
    #             return self.reduced[t-self.start_t()-1].centroid()
    #
    #     return None
    #
    # def get_reduced_at(self, t):
    #     self.if_not_sorted_sort_()
    #
    #     t -= self.start_t() + 1
    #     if 0 <= t < len(self.reduced):
    #         return self.reduced[t]
    #
    #     return None
    #
    # @ staticmethod
    # def reconstruct(r, project):
    #     if isinstance(r, Reduced):
    #         return r.reconstruct(project)
    #
    #     return r
    #
    # def split_at_t(self, t, solver, undo_action=False):
    #     # spins off and reconstructs region at frame t and if chunk is long enough it will separate it into 2 chunks
    #     if self.start_t() < t < self.end_t():
    #         self.if_not_sorted_sort_()
    #
    #         node_t = self.get_reduced_at(t)
    #         pos = self.reduced.index(node_t)
    #         node_t = self.reconstruct(node_t, solver.project)
    #
    #         if pos == 0:
    #             # it is first in reduced, to spin it off two pop_first are enough
    #             self.pop_first(solver)
    #             self.pop_first(solver)
    #         elif pos == len(self.reduced)-1:
    #             # it is last in reduced, to spin it off two pop_last are enough
    #             self.pop_last(solver)
    #             self.pop_last(solver)
    #         else:
    #             # ----- building second chunk -----
    #
    #             # remove node_t we already have
    #             ch2 = Chunk(store_area=self.store_area, id=solver.project.solver_parameters.new_chunk_id())
    #             ch2.start_n = self.reduced.pop(pos+1)
    #             ch2.start_n = self.reconstruct(ch2.start_n, solver.project)
    #             ch2.end_n = self.end_n
    #
    #             while len(self.reduced) > pos+1:
    #                 ch2.reduced.append(self.reduced.pop(pos+1))
    #
    #             # ----- adjust first chunk -----
    #             new_end_n = self.reduced.pop(pos)
    #             reconstructed = self.reconstruct(new_end_n, solver.project)
    #             self.end_n = reconstructed
    #
    #             # ----- reconnect with neighbours ----
    #             # solver.add_node(node_t)
    #             if not undo_action:
    #                 solver.add_node(self.end_n)
    #                 solver.add_node(ch2.start_n)
    #
    #                 t_minus, t, t_plus = solver.get_vertices_around_t(self.end_n.frame_)
    #                 solver.add_edges_(t, t_plus)
    #             # solver.add_edges_(t, t_plus)
    #
    #             # ----- test if we still have 2 chunks and reconnect first and last ---
    #             if not undo_action:
    #                 self.chunk_reconnect_(solver)
    #                 ch2.chunk_reconnect_(solver)
    #
    #     else:
    #         raise Exception("t is out of range of this chunk in chunk.py/split_at_t")
    #
    # def is_virtual_in_time(self, t):
    #     red = self.get_reduced_at(t)
    #     if red is not None:
    #         if isinstance(red, Reduced):
    #             return False
    #         else:
    #             return red.is_virtual
    #
    #     if self.start_t() == t:
    #         return self.start_n.is_virtual
    #     elif self.end_t() == t:
    #         return self.end_n.is_virtual

