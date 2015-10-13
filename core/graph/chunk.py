__author__ = 'fnaiser'

import numpy as np
from reduced import Reduced
from utils.constants import EDGE_CONFIRMED
from core.log import LogCategories, ActionNames
from core.settings import Settings as S_
from core.region.region import Region


class Chunk:
    def __init__(self, vertices_ids, id_, project, color=None):
        self.id_ = id_
        self.nodes_ = vertices_ids
        self.color = color
        self.statistics = {}
        self.project = project

        self.chunk_reconnect_()

    def __str__(self):
        s = "CHUNK --- start_f: "+str(self.start_frame())+" end_f: "+str(self.end_frame())+" length: "+str(len(self.nodes_))+"\n"
        return s

    def append_left(self, vertex, undo_action=False):
        vertex_id = int(vertex)
        region = self.project.gm.region(vertex_id)
        if region.frame() + 1 != self.start_frame():
            print "DISCONTINUITY in chunk.py/append_left", region.frame(), self.start_frame(), region, self.project.gm.region(self.start_node())
            raise Exception("DISCONTINUITY in chunk.py/append_left")

        first = self.start_node()

        ch2, _ = self.project.gm.is_chunk(vertex)
        if ch2:
            ch2.merge(self, undo_action=undo_action)
            return
        else:
            self.nodes_.insert(0, vertex_id)

        if not undo_action:
            self.project.gm.remove_vertex(first, False)
            self.chunk_reconnect_()

    def append_right(self, vertex, undo_action=False):
        vertex_id = int(vertex)
        region = self.project.gm.region(vertex_id)
        if region.frame() != self.end_frame() + 1:
            print "DISCONTINUITY in chunk.py/append_right", region.frame(), self.end_frame(), region, self.end_node()
            raise Exception("DISCONTINUITY in chunk.py/append_right")

        last = self.end_node()

        ch2, _ = self.project.gm.is_chunk(vertex)
        if ch2:
            self.merge(ch2, undo_action=undo_action)
            return
        else:
            self.nodes_.append(vertex_id)

        if not undo_action:
            self.project.gm.remove_vertex(last, False)
            self.chunk_reconnect_()

    def pop_first(self, undo_action=False):
        first = self.nodes_.pop(0)
        new_start = self.start_node()

        if not undo_action:
            self.project.gm.add_vertex(new_start)

        if not undo_action:
            self.project.gm.remove_edge(first, self.end_node())
            prev_nodes, _, _ = self.project.gm.get_vertices_around_t(self.project.gm.region(new_start).frame())
            self.project.gm.add_edges_(prev_nodes, [new_start])

        if not undo_action:
            if len(self.nodes_) > 1:
                self.chunk_reconnect_()

        return first

    def pop_last(self, undo_action=False):
        last = self.nodes_.pop()
        new_end = self.end_node()

        if not undo_action:
            self.project.gm.add_vertex(new_end)

        if not undo_action:
            self.project.gm.remove_edge(self.start_node(), last)

            _, _, next_nodes = self.project.gm.get_vertices_around_t(self.project.gm.region(new_end).frame())
            self.project.gm.add_edges_([new_end], next_nodes)

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
            self.project.gm.remove_vertex(ch1end)
            self.project.gm.remove_vertex(ch2start)

        self.nodes_.extend(ch2.nodes_)

        if not undo_action:
            self.chunk_reconnect_()

    def merge_and_interpolate(self, ch2, undo_action=False):
        if self.end_frame() > ch2.start_frame():
            ch2.merge_and_interpolate(self, undo_action=undo_action)
            return

        gap_len = ch2.start_frame() - self.end_frame() - 1
        if gap_len > 0:
            ch2start_region = self.project.gm.region(ch2.start_node())
            ch1end_region = self.project.gm.region(self.end_node())

            c_diff_part = (ch2start_region.centroid() - ch1end_region.centroid()) / gap_len

            i = 1
            for f in range(self.end_frame()+1, ch2.start_frame()):
                r = Region(frame=f)
                r.is_virtual = True
                c = ch1end_region.centroid() + np.array(c_diff_part * i)
                r.centroid_ = c.copy()

                # TODO: log...
                node = self.project.gm.add_vertex(r)
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
        return self.project.gm.region(self.start_node()).frame()

    def end_frame(self):
        return self.project.gm.region(self.end_node()).frame()

    def length(self):
        return len(self.nodes_)

    def chunk_reconnect_(self):
        self.project.gm.add_edge(self.start_node(), self.end_node())
        self.project.gm.g.vp['chunk_start_id'][self.project.gm.g.vertex(self.start_node())] = self.id()
        self.project.gm.g.vp['chunk_end_id'][self.project.gm.g.vertex(self.end_node())] = self.id()