__author__ = 'fnaiser'

import numpy as np
from reduced import Reduced
from utils.constants import EDGE_CONFIRMED
from core.log import LogCategories, ActionNames
from core.settings import Settings as S_
from core.region.region import Region


class Chunk:
    def __init__(self, vertices_ids, id_, gm, color=None):
        # if not isinstance(vertices_ids, list):
        #     raise Exception('vertices_ids must be a list! (in chunk.py)')
        # if len(vertices_ids) < 2:
        #     raise Exception('vertices_ids must be a list with length >= 2 (in chunk.py)')
        self.id_ = id_

        # list of integers. If >= 0 means vertex_id, if < 0 direct link -> region_id
        self.nodes_ = vertices_ids
        self.color = color
        self.statistics = {}
        self.animal_id_ = -1
        self.P = set()
        self.N = set()
        self.segmentation_class = -1

        if len(vertices_ids) > 1:
            if vertices_ids[0] > 0:
                v1 = gm.g.vertex(vertices_ids[0])
                out_edges = [e for e in v1.out_edges()]
                for e in out_edges:
                    gm.remove_edge_(e)

            if vertices_ids[-1] > 0:
                v2 = gm.g.vertex(vertices_ids[-1])

                in_edges = [e for e in v2.in_edges()]
                for e in in_edges:
                    gm.remove_edge_(e)

            if len(vertices_ids) > 2:
                for v in vertices_ids[1:-1]:
                    if v > 0:
                        gm.remove_vertex(v)
                        # v = gm.g.vertex(v)
                        # for e in v.in_edges():
                        #     gm.remove_edge_(e)

        self.chunk_reconnect_(gm)

    def __str__(self):
        s = "CHUNK --- id: "+str(self.id_)+" length: "+str(len(self.nodes_))+"\n"
        return s

    def __len__(self):
        return len(self.nodes_)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self.nodes_)
            return self.nodes_[key]

        ids = []
        if isinstance(key, slice):
            start = key.start
            if start is None:
                start = 0

            stop = key.stop
            if stop is None or stop == 9223372036854775807:
                stop = len(self.nodes_)

            step = key.step
            if step is None:
                step = 1

            ids = range(start, stop, step)
        elif isinstance(key, list):
            ids = key

        items = []

        for i in ids:

            items.append(self.nodes_[i])

        return items

    def print_info(self, gm):
        s = "TRACKLET --- id: "+str(self.id_)+" length: "+str(len(self.nodes_))+"\n"
        s += "\tstarts at: "+str(self.start_frame(gm))+" ends at: "+str(self.end_frame(gm))

        print s

    def append_left(self, vertex, gm, undo_action=False):
        if int(vertex) == 4:
            print vertex

        # test: there cannot be any outgoing edge...
        out_edges = [e for e in vertex.out_edges()]
        for e in out_edges:
            gm.remove_edge_(e)


        vertex_id = int(vertex)
        region = gm.region(vertex_id)
        if region.frame() + 1 != self.start_frame(gm):
            # print ("DISCONTINUITY in chunk.py/append_left region_frame: %d, ch_start_frame: %d", region.frame(), self.start_frame(gm))
            # print "DISCONTINUITY in chunk.py/append_left", region.frame(), self.start_frame(gm), region, self.project.gm.region(self.start_node())
            raise Exception("DISCONTINUITY in chunk.py/append_left")

        first = self.start_node()

        ch2, _ = gm.is_chunk(vertex)
        if ch2:
            ch2.merge(self, gm, undo_action=undo_action)
            return
        else:
            self.nodes_.insert(0, vertex_id)

        if not undo_action:
            gm.remove_vertex(first, False)
            self.chunk_reconnect_(gm)

    def append_right(self, vertex, gm, undo_action=False):
        # test: there cannot be any incomming edge...
        in_edges = [e for e in vertex.in_edges()]
        for e in in_edges:
            gm.remove_edge_(e)

        vertex_id = int(vertex)
        region = gm.region(vertex_id)
        if region.frame() != self.end_frame(gm) + 1:
            # print "DISCONTINUITY in chunk.py/append_right", region.frame(), self.end_frame(gm), region, self.end_node()
            raise Exception("DISCONTINUITY in chunk.py/append_right, frame: {}, r_id: {}".format(region.frame(), region.id()))

        last = self.end_node()

        ch2, _ = gm.is_chunk(vertex)
        if ch2:
            self.merge(ch2, gm, undo_action=undo_action)
            return
        else:
            self.nodes_.append(vertex_id)

        if not undo_action:
            gm.remove_vertex(last, False)
            self.chunk_reconnect_(gm)

    def pop_first(self, gm, undo_action=False):
        first = self.nodes_.pop(0)

        # if last node was popped (e.g. during whole chunk fitting)
        if self.length() > 1:
            new_start = self.start_node()

            if not undo_action:
                new_start = gm.add_vertex(gm.region(new_start))
                # it is necessary to verride vertex_id as the ids inside chunk are not vertices ids but -region_ids
                self.nodes_[0] = int(new_start)

            if not undo_action:
                gm.remove_edge(gm.g.vertex(first), gm.g.vertex(self.end_node()))
                prev_nodes = gm.get_vertices_in_t(gm.region(new_start).frame() - 1)
                gm.add_edges_(prev_nodes, [new_start])

        if not undo_action:
            if len(self.nodes_) > 1:
                self.chunk_reconnect_(gm)

        gm.g.vp['chunk_start_id'][gm.g.vertex(first)] = 0
        gm.g.vp['chunk_end_id'][gm.g.vertex(first)] = 0

        return first

    def pop_last(self, gm, undo_action=False):
        last = self.nodes_.pop()

        # if last node was popped (e.g. during whole chunk fitting)
        new_end = None
        if self.length() > 1:
            new_end = self.end_node()

            if not undo_action:
                new_end = gm.add_vertex(gm.region(new_end))
                # it is necessary to override vertex_id, as it was inside chunk, thus the id was -region_id
                self.nodes_[-1] = int(new_end)

            if not undo_action:
                gm.remove_edge(gm.g.vertex(self.start_node()), gm.g.vertex(last))

                next_nodes = gm.get_vertices_in_t(gm.region(new_end).frame() + 1)
                gm.add_edges_([new_end], next_nodes)

                self.chunk_reconnect_(gm)

        gm.g.vp['chunk_start_id'][gm.g.vertex(last)] = 0
        gm.g.vp['chunk_end_id'][gm.g.vertex(last)] = 0

        return last

    def merge(self, ch2, gm, undo_action=False):
        """
        |ch1.start| ... |ch1.end|   |ch2.start|... |ch2.end|
        -> |ch1.start| ... |ch2.end|

        :param second_chunk:
        :param undo_action:
        :return:
        """
        if self.start_frame(gm) > ch2.start_frame(gm):
            ch2.merge(self, gm)
            return

        ch1end = self.end_node()
        ch2start = ch2.start_node()

        gm.project.chm.remove_chunk(ch2, gm)
        gm.project.chm._try_ch_itree_delete(self, gm)

        if not undo_action:
            if self.length() > 1:
                gm.remove_vertex(ch1end, disassembly=False)
            if ch2.length() > 1:
                gm.remove_vertex(ch2start, disassembly=False)

        self.nodes_.extend(ch2.nodes_)

        if not undo_action:
            self.chunk_reconnect_(gm)

        gm.project.chm._add_ch_itree(self, gm)

    def merge_and_interpolate(self, ch2, gm, undo_action=False):
        if self.end_frame(gm) > ch2.start_frame(gm):
            ch2.merge_and_interpolate(self, gm, undo_action=undo_action)
            return

        gap_len = ch2.start_frame(gm) - self.end_frame(gm) - 1
        if gap_len > 0:
            ch2start_region = gm.region(ch2.start_node())
            ch1end_region = gm.region(self.end_node())

            c_diff_part = (ch2start_region.centroid() - ch1end_region.centroid()) / gap_len

            i = 1
            for f in range(self.end_frame(gm)+1, ch2.start_frame(gm)):
                r = Region(frame=f)
                r.is_virtual = True
                c = ch1end_region.centroid() + np.array(c_diff_part * i)
                r.centroid_ = c.copy()

                # TODO: log...
                node = gm.add_vertex(r)
                self.append_right(node)

                i += 1

        self.merge(ch2, gm, undo_action)

    def split_at(self, frame, gm):
        """
        splits tracklet so the node in t=frame stays in the left tracklet

        Args:
            frame:
            gm:

        Returns:

        """
        start_frame = self.start_frame(gm)

        key = frame - start_frame
        left_nodes = []
        right_nodes = []

        if key >= 0 and key < self.length():
            left_nodes = list(self.nodes_[:key+1])
            right_nodes = self.nodes_[key+1:]

            # TODO: undo action?
            # TODO: what if chunk is of length 2?
            new_end = left_nodes[-1]
            new_end = gm.add_vertex(gm.region(new_end))
            left_nodes[-1] = int(new_end)

            # remove previous edge...
            gm.remove_edge(gm.g.vertex(self.start_node()), gm.g.vertex(right_nodes[-1]))
            next_nodes = gm.get_vertices_in_t(gm.region(new_end).frame() + 1)
            gm.add_edges_([new_end], next_nodes)

            gm.g.vp['chunk_start_id'][gm.g.vertex(right_nodes[-1])] = 0
            gm.g.vp['chunk_end_id'][gm.g.vertex(right_nodes[-1])] = 0

            # not last node of tracklet... because it is already in graph
            if key < self.length() - 1:
                new_start = right_nodes[0]
                new_start = gm.add_vertex(gm.region(new_start))
                right_nodes[0] = int(new_start)

            # self.nodes_ = left_nodes
            # self.chunk_reconnect_(gm)

        return left_nodes, right_nodes

    def id(self):
        return self.id_

    def start_vertex_id(self):
        return self.nodes_[0]

    def end_vertex_id(self):
        return self.nodes_[-1]

    def end_vertex(self, gm):
        return gm.g.vertex(self.end_vertex_id())

    def end_node(self):
        return self.end_vertex_id()

    def start_vertex(self, gm):
        return gm.g.vertex(self.start_vertex_id())

    def start_node(self):
        return self.start_vertex_id()

    def start_frame(self, gm):
        return gm.region(self.start_node()).frame()

    def end_frame(self, gm):
        return gm.region(self.end_node()).frame()

    def length(self):
        return len(self)

    def is_empty(self):
        return True if self.length() == 0 else False

    def chunk_reconnect_(self, gm):
        if len(self.nodes_) > 1:
            if self.start_vertex(gm).out_degree() > 0:
                gm.remove_outgoing_edges(self.start_vertex(gm))

            gm.add_edge(self.start_node(), self.end_node(), 1.0)

        gm.g.vp['chunk_start_id'][gm.g.vertex(self.start_node())] = self.id()
        gm.g.vp['chunk_end_id'][gm.g.vertex(self.start_node())] = 0

        gm.g.vp['chunk_start_id'][gm.g.vertex(self.end_node())] = 0
        gm.g.vp['chunk_end_id'][gm.g.vertex(self.end_node())] = self.id()

    def is_only_one_id_assigned(self, num_animals):
        # if there is one and only one ID assigned to chunk
        return len(self.P) == 1 and \
               len(self.N) == num_animals - 1

    def v_gen(self):
        for v in self.nodes_:
            yield v

    def rid_gen(self, gm):
        for id_ in self.nodes_:
            yield gm.region_id(id_)

    def get_region(self, gm, i):
        return gm.region_id[self.nodes_[i]]

    def r_gen(self, gm, rm):
        for rid in self.rid_gen(gm):
            yield rm[rid]

    def v_id_in_t(self, t, gm):
        t = t - self.start_frame(gm)
        if -1 < t < len(self.nodes_):
            return self.nodes_[t]
        else:
            return None

    def r_id_in_t(self, t, gm):
        return gm.region_id(self.v_id_in_t(t, gm))

    def is_single(self):
        return self.segmentation_class == 0

    def is_multi(self):
        return self.segmentation_class == 1

    def is_noise(self):
        return self.segmentation_class == 2

    def is_part(self):
        return self.segmentation_class == 3

    def is_undefined(self):
        return self.segmentation_class == -1

    def segmentation_class_str(self):
        if self.is_single():
            return "single-ID"
        elif self.is_multi():
            return "multi-ID"
        elif self.is_noise():
            return "no-ID"
        elif self.is_part():
            return "part-of-ID"
        else:
            return "undefined"

    def num_outcoming_edges(self, gm):
        return self.end_vertex(gm).out_degree()

    def num_incoming_edges(self, gm):
        return self.start_vertex(gm).in_degree()

    def get_cardinality(self, gm):
        """
        cardinality = #IDS in given tracklet
        
        Returns: 1 if single, 2, 3, ... when cardinality is known, 0 when cardinality is known and tracklet is noise, 
        -1 when cardinality is not defined

        """

        if self.is_noise():
            return 0

        if self.is_single():
            return 1

        if self.is_multi():
            # first try INcoming...
            cardinality_based_on_in = 0
            for ch in gm.get_incoming_tracklets(self.start_vertex(gm)):
                if ch.is_single() and ch.num_outcoming_edges(gm) == 1:
                    cardinality_based_on_in += 1
                else:
                    cardinality_based_on_in = 0
                    break

            cardinality_based_on_out = 0
            # lets try OUTcoming...
            for ch in gm.get_outcoming_tracklets(self.end_vertex(gm)):
                if ch.is_single() and ch.num_incoming_edges(gm) == 1:
                    cardinality_based_on_out += 1
                else:
                    return -1

            if cardinality_based_on_in == 0 and cardinality_based_on_out:
                return cardinality_based_on_out

            if cardinality_based_on_in and cardinality_based_on_out == 0:
                return cardinality_based_on_in

        return -1

    def entering_tracklets(self, gm):
        return gm.get_incoming_tracklets(self.start_vertex(gm))

    def exiting_tracklets(self, gm):
        return gm.get_outcoming_tracklets(self.end_vertex(gm))

    def solve_interaction(self, detector, gm, rm, im):
        """
        Find tracks in chunks containing two objects.

        :param detector: InteractionDetector() object
        :param gm:
        :param rm:
        :param im:
        :return: pandas.DataFrame - two tracks
        """
        assert self.get_cardinality(gm) == 2
        detections = []
        for r in self.r_gen(gm, rm):
            img = im.get_whole_img(r.frame())
            pred = detector.detect(img, r.centroid()[::-1])
            detections.append(pred)
        tracks, confidence, costs = detector.track(detections)
        return tracks, confidence

