# track is set of tracklets which has the same properties as tracklet (chunk in old nomenclature) but might be
# discontinuous
from .chunk import Chunk
from .ghost_tracklet import GhostTracklet
from warnings import warn
from intervals import IntInterval
import itertools
import numbers


def unfold(tracklets):
    new_tracklets = []

    for t in tracklets:
        if t.is_tracklet():
            new_tracklets.append(t)
        elif t.is_track():
            new_tracklets.extend(unfold(t._data))

    return new_tracklets


class Track(Chunk):
    def __init__(self, tracklets, gm, id_=-1, ghost_tracklets=False):
        super(Track, self).__init__(None, id_, gm)

        # tracklets and spaces...
        self._data = []
        self._num_regions = 0

        self.extend(tracklets, ghost_tracklets)

        self.color = self._data[0].color
        self.segmentation_class = self._data[0].segmentation_class
        self.start = self._data[0].start_frame()
        self.end = self._data[-1].end_frame()

    def __str__(self):
        s = "Track --- id: "+str(self.id_)+" length: "+str(len(self))+"\n"
        return s

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            if key < 0:  # Handle negative indices
                key += len(self)

            offset = 0
            for t in self._data:
                if key - offset < len(t):
                    break
                else:
                    offset += len(t)

            return t[key-offset]
        #
        # ids = []
        # if isinstance(key, slice):
        #     start = key.start
        #     if start is None:
        #         start = 0
        #
        #     stop = key.stop
        #     if stop is None or stop == 9223372036854775807:
        #         stop = len(self.nodes_)
        #
        #     step = key.step
        #     if step is None:
        #         step = 1
        #
        #     ids = range(start, stop, step)
        # elif isinstance(key, list):
        #     ids = key
        #
        # items = []
        #
        # for i in ids:
        #     items.append(self.nodes_[i])

        warn("Not implemented for Track... But can't be too hard to do it...")

    def num_regions(self):
        return self._num_regions

    def start_frame(self):
        return self.start

    def end_frame(self):
        return self.end

    def tracklets(self):
        return self._data

    def _sort_tracklets(self):
        self._data = sorted(self._data, key=lambda x: x.start_frame())
        self.start = self._data[0].start_frame()
        self.end = self._data[-1].end_frame()

    def append(self, tracklet, sort=True, ghost_tracklets=False):
        if ghost_tracklets:
            sf = tracklet.start_frame()
            if len(self._data):
                ef = self._data[-1].end_frame()
                if ef + 1 != sf:
                    self._data.append(GhostTracklet(ef, sf - 1))
        self._data.append(tracklet)
        self.P = self.P.union(tracklet.P)
        self.N = self.N.union(tracklet.N)
        self._num_regions += len(tracklet)
        assert self.is_consistent()
        if sort:
            self._sort_tracklets()

    def extend(self, tracklets, ghost_tracklets=False):
        for t in tracklets:
            self.append(t, sort=False, ghost_tracklets=ghost_tracklets)
        self._sort_tracklets()

    def merge(self, other_track):
        P = self.P.copy()
        N = self.N.copy()
        self.extend(other_track.tracklets())
        self.P = P
        self.N = N

    def __len__(self):
        return self.end - self.start + 1

    def get_temporal_intervals(self):
        return [(t.start_frame(), t.end_frame()) for t in self._data]

    def get_interval(self):
        return [t.get_interval() for t in self._data]

    def is_overlapping(self, other):
        for t1, t2 in itertools.product(self.tracklets(), other.tracklets()):
            if t1.is_overlapping(t2):
                return True
        return False

    def is_consistent(self):
        for t1, t2 in itertools.combinations(self.tracklets(), 2):
            if t1.is_overlapping(t2):
                return False
        return True

    def print_info(self):
        s = "TRACKLET --- id: "+str(self.id_)+" length: "+str(len(self.nodes_))+"\n"
        s += "\tstarts at: " + str(self.start_frame()) + " ends at: " + str(self.end_frame())

        print(s)

    def append_left(self, vertex, undo_action=False):
        warn("Not implemented for Track...")

    def append_right(self, vertex, undo_action=False):
        warn("Not implemented for Track...")

    def pop_first(self, undo_action=False):
        warn("Not implemented for Track...")

    def pop_last(self, undo_action=False):
        warn("Not implemented for Track...")

    def merge_and_interpolate(self, ch2):
        warn("Not implemented for Track...")

    def split_at(self, frame):
        warn("Not implemented for Track...")

    def id(self):
        return self.id_

    def start_vertex_id(self):
        return self._data[0].nodes_[0]

    def end_vertex_id(self):
        return self._data[-1].nodes_[-1]

    def end_vertex(self):
        return self.gm.g.vertex(self.end_vertex_id())

    def end_node(self):
        return self.end_vertex_id()

    def start_vertex(self):
        return self.gm.g.vertex(self.start_vertex_id())

    def start_node(self):
        return self._data[0].start_vertex_id()

    def chunk_reconnect_(self):
        warn("Not implemented for Track")

    def v_gen(self):
        for t in self._data:
            for v in t.nodes_:
                yield v

    def rid_gen(self):
        for t in self._data:
            for id_ in t.nodes_:
                yield self.gm.region_id(id_)

    def v_id_in_t(self, frame):
        tracklet = self._tracklet(frame)

        frame = frame - self.start_frame()
        if -1 < frame < len(tracklet.nodes_):
            return tracklet.nodes_[frame]
        else:
            return None

    def _tracklet(self, frame):
        # TODO: improve...
        for t in self._data:
            if t.start_frame() <= frame <= t.end_frame():
                return t

    def r_id_in_t(self, frame):
        tracklet = self._tracklet(frame)
        # TODO: find proper chunk...
        return self.gm.region_id(tracklet.v_id_in_t(frame))

    def is_tracklet(self):
        return False

    def is_track(self):
        return True

    def is_inside(self, tracklet):
        for t in self._data:
            if t.id() == tracklet.id():
                return True

        return False

    def draw(self, rm, *args, **kwargs):
        import matplotlib.pylab as plt
        for t in self._data:
            t.draw(rm, *args, **kwargs)
        xy = self._data[0].get_region(0).centroid()[::-1]
        plt.annotate('track {}'.format(self.id()), xy=xy, textcoords='offset pixels', xytext=(-10, -10), color='w')

