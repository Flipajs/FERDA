import numpy as np
from core.region.region import Region
from random import randint
import warnings
from intervals import IntInterval
import numbers


class Chunk(object):
    """
    Each tracklet has 2 track id sets.
    P - ids are surely present
    N - ids are surely not present

    A - set of all animal ids.
    When P.union(N) == A, tracklet is decided. When len(P) == 1, it is a tracklet with one id.

    When len(P.intersection(N)) > 0 it is a CONFLICT

    """
    def __init__(self, vertices_ids, id_, gm, color=None, origin_interaction=False):
        assert color is None or isinstance(color, np.ndarray)
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
        self.cardinality = None  # estimated number of objects
        self.segmentation_class = -1  # -1 undefined, 0 single, 1 multi, 2 noise, 3 part of object
        self.gm = gm

        self.origin_interaction = origin_interaction

        if not self.origin_interaction:
            if vertices_ids is not None and len(vertices_ids) > 1:
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

            self.chunk_reconnect_()

    def is_consistent(self):
        # first and last node should be positive, the rest negative
        return self.nodes_[0] > 0 and self.nodes_[-1] > 0  #  and all([n < 0 for n in self.nodes_[1:-1]])

    def __str__(self):
        s = "Tracklet --- id: "+str(self.id_)+" length: "+str(len(self.nodes_))+" "+str(self.P)+"\n"
        return s

    def __len__(self):
        return len(self.nodes_)

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
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

            ids = list(range(start, stop, step))
        elif isinstance(key, list):
            ids = key
        else:
            assert False

        items = []

        for i in ids:

            items.append(self.nodes_[i])

        return items

    def __getstate__(self):
        if isinstance(self.color, np.ndarray):
            self.color = self.color.tolist()
        state = self.__dict__.copy()
        del state['gm']
        return state

    def set_random_color(self, low=0, high=255):
        self.color = np.random.randint(low, high, 3)

    def print_info(self):
        s = "TRACKLET --- id: "+str(self.id_)+" length: "+str(len(self.nodes_))+"\n"
        s += "\tstarts at: " + str(self.start_frame()) + " ends at: " + str(self.end_frame())
        print(s)

    def append_left(self, vertex):
        # test: there cannot be any outgoing edge...
        out_edges = [e for e in vertex.out_edges()]
        for e in out_edges:
            self.gm.remove_edge_(e)

        vertex_id = int(vertex)
        region = self.gm.region(vertex_id)
        if region.frame() + 1 != self.start_frame():
            # print("DISCONTINUITY in chunk.py/append_left region_frame: %d, ch_start_frame: %d", region.frame(), self.start_frame(gm))
            # print "DISCONTINUITY in chunk.py/append_left", region.frame(), self.start_frame(gm), region, self.project.gm.region(self.start_node())
            raise Exception("DISCONTINUITY in chunk.py/append_left")

        first = self.start_node()

        ch2, _ = self.gm.is_chunk(vertex)
        if ch2:
            ch2.merge(self)
            return
        else:
            self.nodes_.insert(0, vertex_id)

        self.gm.remove_vertex(first, False)
        self.chunk_reconnect_()

    def append_right(self, vertex):
        # test: there cannot be any incomming edge...
        in_edges = [e for e in vertex.in_edges()]
        for e in in_edges:
            self.gm.remove_edge_(e)

        vertex_id = int(vertex)
        region = self.gm.region(vertex_id)
        if region.frame() != self.end_frame() + 1:
            # print "DISCONTINUITY in chunk.py/append_right", region.frame(), self.end_frame(gm), region, self.end_node()
            raise Exception("DISCONTINUITY in chunk.py/append_right, frame: {}, r_id: {}".format(region.frame(), region.id()))

        last = self.end_node()
        ch2, _ = self.gm.is_chunk(vertex)
        if ch2:
            self.merge(ch2)
            return
        else:
            self.nodes_.append(vertex_id)
            self.gm.remove_vertex(last, False)
            self.chunk_reconnect_()

    def pop_first(self):
        first = self.nodes_.pop(0)

        # if last node was popped (e.g. during whole chunk fitting)
        if self.length() > 1:
            new_start = self.start_node()

            new_start = self.gm.add_vertex(self.gm.region(new_start))
            # it is necessary to verride vertex_id as the ids inside chunk are not vertices ids but -region_ids
            self.nodes_[0] = int(new_start)

            self.gm.remove_edge(self.gm.g.vertex(first), self.gm.g.vertex(self.end_node()))
            prev_nodes = self.gm.get_vertices_in_t(self.gm.region(new_start).frame() - 1)
            self.gm.add_edges_(prev_nodes, [new_start])

        if len(self.nodes_) > 1:
            self.chunk_reconnect_()

        self.gm.g.vp['chunk_start_id'][self.gm.g.vertex(first)] = 0
        self.gm.g.vp['chunk_end_id'][self.gm.g.vertex(first)] = 0

        return first

    def pop_last(self):
        last = self.nodes_.pop()

        # if last node was popped (e.g. during whole chunk fitting)
        if self.length() > 1:
            new_end = self.end_node()

            new_end = self.gm.add_vertex(self.gm.region(new_end))
            # it is necessary to override vertex_id, as it was inside chunk, thus the id was -region_id
            self.nodes_[-1] = int(new_end)

            self.gm.remove_edge(self.gm.g.vertex(self.start_node()), self.gm.g.vertex(last))

            next_nodes = self.gm.get_vertices_in_t(self.gm.region(new_end).frame() + 1)
            self.gm.add_edges_([new_end], next_nodes)

            self.chunk_reconnect_()

        self.gm.g.vp['chunk_start_id'][self.gm.g.vertex(last)] = 0
        self.gm.g.vp['chunk_end_id'][self.gm.g.vertex(last)] = 0

        return last

    def merge(self, ch2):
        """
        |ch1.start| ... |ch1.end|   |ch2.start|... |ch2.end|
        -> |ch1.start| ... |ch2.end|

        :param second_chunk:
        :return:
        """
        if self.start_frame() > ch2.start_frame():
            ch2.merge(self)
            return

        ch1end = self.end_node()
        ch2start = ch2.start_node()

        # TODO: refactor to not expect self.gm.project
        self.gm.project.chm.remove_tracklet(ch2)  # if this fails, see core/graph_assembly.py:215
        self.gm.project.chm._try_ch_itree_delete(self)

        if self.length() > 1:
            self.gm.remove_vertex(ch1end, disassembly=False)
        if ch2.length() > 1:
            self.gm.remove_vertex(ch2start, disassembly=False)

        self.nodes_.extend(ch2.nodes_)

        self.chunk_reconnect_()

        self.gm.project.chm._add_ch_itree(self)

    def merge_and_interpolate(self, ch2):
        if self.end_frame() > ch2.start_frame():
            ch2.merge_and_interpolate(self)
            return

        gap_len = ch2.start_frame() - self.end_frame() - 1
        if gap_len > 0:
            ch2start_region = self.gm.region(ch2.start_node())
            ch1end_region = self.gm.region(self.end_node())

            c_diff_part = (ch2start_region.centroid() - ch1end_region.centroid()) / gap_len

            i = 1
            for f in range(self.end_frame() + 1, ch2.start_frame()):
                r = Region(frame=f)
                r.is_origin_interaction_ = True
                c = ch1end_region.centroid() + np.array(c_diff_part * i)
                r.centroid_ = c.copy()

                # TODO: log...
                node = self.gm.add_vertex(r)
                self.append_right(node)

                i += 1

        self.merge(ch2)

    def split_at(self, frame):
        """
        splits tracklet so the node in t=frame stays in the left tracklet

        Args:
            frame:

        Returns:

        """
        start_frame = self.start_frame()

        key = frame - start_frame
        left_nodes = []
        right_nodes = []

        if 0 <= key < self.length():
            left_nodes = list(self.nodes_[:key+1])
            right_nodes = self.nodes_[key+1:]

            # TODO: what if chunk is of length 2?
            new_end = left_nodes[-1]
            new_end = self.gm.add_vertex(self.gm.region(new_end))
            left_nodes[-1] = int(new_end)

            # remove previous edge...
            self.gm.remove_edge(self.gm.g.vertex(self.start_node()), self.gm.g.vertex(right_nodes[-1]))
            next_nodes = self.gm.get_vertices_in_t(self.gm.region(new_end).frame() + 1)
            self.gm.add_edges_([new_end], next_nodes)

            self.gm.g.vp['chunk_start_id'][self.gm.g.vertex(right_nodes[-1])] = 0
            self.gm.g.vp['chunk_end_id'][self.gm.g.vertex(right_nodes[-1])] = 0

            # not last node of tracklet... because it is already in graph
            if key < self.length() - 1:
                new_start = right_nodes[0]
                new_start = self.gm.add_vertex(self.gm.region(new_start))
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

    def end_vertex(self):
        return self.gm.g.vertex(self.end_vertex_id())

    def end_node(self):
        return self.end_vertex_id()

    def start_vertex(self):
        return self.gm.g.vertex(self.start_vertex_id())

    def start_node(self):
        return self.start_vertex_id()

    def start_frame(self):
        return self.gm.region(self.start_node()).frame()

    def end_frame(self):
        return self.gm.region(self.end_node()).frame()

    def length(self):
        return len(self)

    def is_empty(self):
        return True if self.length() == 0 else False

    def chunk_reconnect_(self):
        if len(self.nodes_) > 1:
            if self.start_vertex().out_degree() > 0:
                self.gm.remove_outgoing_edges(self.start_vertex())

            self.gm.add_edge(self.start_node(), self.end_node(), 1.0)

        self.gm.g.vp['chunk_start_id'][self.gm.g.vertex(self.start_node())] = self.id()
        self.gm.g.vp['chunk_end_id'][self.gm.g.vertex(self.start_node())] = 0

        self.gm.g.vp['chunk_start_id'][self.gm.g.vertex(self.end_node())] = 0
        self.gm.g.vp['chunk_end_id'][self.gm.g.vertex(self.end_node())] = self.id()

    def is_only_one_id_assigned(self, num_objects):
        warnings.warn('is_only_one_id_assigned is possibly bugged, len(P) + len(N) is not always == number of objects')
        # if there is one and only one ID assigned to chunk
        return len(self.P) == 1 and \
               len(self.N) == num_objects - 1

    def v_gen(self):
        for v in self.nodes_:
            yield v

    def rid_gen(self):
        for id_ in self.nodes_:
            yield self.gm.region_id(id_)

    def get_region(self, i):
        return self.gm.region(self.nodes_[i])

    def get_region_in_frame(self, frame):
        sf = self.start_frame()
        try:
            return self.get_region(frame - sf)
        except Exception as e:
            import warnings
            warnings.warn(e.message)
            return None

    def r_gen(self, rm):
        for rid in self.rid_gen():
            yield rm[rid]

    def v_id_in_t(self, t):
        t = t - self.start_frame()
        if -1 < t < len(self.nodes_):
            return self.nodes_[t]
        else:
            return None

    def is_origin_interaction(self):
        try:
            return self.origin_interaction
        except:
            return False

    def r_id_in_t(self, t):
        return self.gm.region_id(self.v_id_in_t(t))

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
            return "single"
        elif self.is_multi():
            return "multi"
        elif self.is_noise():
            return "noise"
        elif self.is_part():
            return "part"
        else:
            return "undefined"

    def is_ghost(self):
        return False

    def is_tracklet(self):
        return True

    def is_track(self):
        return False

    def num_outcoming_edges(self):
        return self.end_vertex().out_degree()

    def num_incoming_edges(self):
        return self.start_vertex().in_degree()

    def get_cardinality(self):
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
            for ch in self.gm.get_incoming_tracklets(self.start_vertex()):
                if ch.is_single() and ch.num_outcoming_edges() == 1:
                    cardinality_based_on_in += 1
                else:
                    cardinality_based_on_in = 0
                    break

            cardinality_based_on_out = 0
            # lets try OUTcoming...
            for ch in self.gm.get_outcoming_tracklets(self.end_vertex()):
                if ch.is_single() and ch.num_incoming_edges() == 1:
                    cardinality_based_on_out += 1
                else:
                    return -1

            if cardinality_based_on_in == 0 and cardinality_based_on_out:
                return cardinality_based_on_out

            if cardinality_based_on_in and cardinality_based_on_out == 0:
                return cardinality_based_on_in

        return -1

    def entering_tracklets(self):
        return self.gm.get_incoming_tracklets(self.start_vertex())

    def exiting_tracklets(self):
        return self.gm.get_outcoming_tracklets(self.end_vertex())

    def solve_interaction(self, detector, rm, im):
        """
        Find tracks in chunks containing two objects.

        :param detector: InteractionDetector() object
        :param rm:
        :param im:
        :return: pandas.DataFrame - two tracks
        """
        assert self.get_cardinality() == 2
        detections = []
        for r in self.r_gen(rm):
            img = im.get_whole_img(r.frame())
            pred = detector.detect_single(img, r.centroid()[::-1])
            detections.append(pred)
        tracks, confidence, costs = detector.track(detections)
        return tracks, confidence

    def is_id_decided(self):
        return len(self.P) == 1

    def get_random_region(self):
        r_frame = randint(self.start_frame(), self.end_frame())
        return self.get_region_in_frame(r_frame)

    def get_track_id(self):
        assert self.is_id_decided()
        return next(iter(self.P))

    def get_interval(self):
        return IntInterval([int(self.start_frame()), int(self.end_frame())])  # int() is needed to convert numpy.int64

    def is_overlapping(self, other):
        return self.get_interval().is_connected(other.get_interval())

    def draw(self, rm, *args, **kwargs):
        if len(self):
            import matplotlib.pylab as plt
            xy = np.array([region.centroid()[::-1] for region in self.r_gen(rm)])
            plt.plot(xy[:, 0], xy[:, 1], *args, **kwargs)
            plt.annotate('{}'.format(self.id()), xy=xy[0], textcoords='offset pixels', xytext=(10, 10), color='w')



