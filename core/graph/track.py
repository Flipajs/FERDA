# track is set of tracklets which has the same properties as tracklet (chunk in old nomenclature) but might be
# discontinuous
from chunk import Chunk
from ghost_tracklet import GhostTracklet
from warnings import warn

class Track(Chunk):
    def __init__(self, tracklets, gm, id_=-1):
        self.id_ = id_
        self.gm = gm

        # tracklets and spaces...
        self._data=[]
        tracklets = sorted(tracklets, key=lambda x:x.start_frame(gm))

        self.P = set()
        self.N = set()

        self._num_regions = 0
        for t in tracklets:
            sf = t.start_frame(gm)
            if len(self._data):
                ef = self._data[-1].end_frame(gm)
                if ef + 1 != sf:
                    self._data.append(GhostTracklet(ef, sf - 1))

            self.P.union(t.P)
            self.N.union(t.N)

            self._data.append(t)

            self._num_regions += len(t)

        self.color = self._data[0].color
        self.segmentation_class = self._data[0].segmentation_class

        self.start = self._data[0].start_frame(gm)
        self.end = self._data[-1].end_frame(gm)

    def __str__(self):
        s = "Track --- id: "+str(self.id_)+" length: "+str(len(self))+"\n"
        return s

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)

            t = self._tracklet(key+self.start, self.gm)
            return t[key]
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

    # gm is tracklet compability
    def start_frame(self, gm=None):
        return self.start

    # gm is tracklet compability
    def end_frame(self, gm=None):
        return self.end

    def add_tracklet(self, tracklet, gm):
        # TODO: possible speedup
        my_tracklets = [tracklet]
        for t in self._data:
            if isinstance(Chunk):
                my_tracklets.append(t)

        self.__dict__.update(Track(my_tracklets, gm, id_=self.id_).__dict__)

    def __len__(self):
        return self.end - self.start

    def print_info(self, gm):
        s = "TRACKLET --- id: "+str(self.id_)+" length: "+str(len(self.nodes_))+"\n"
        s += "\tstarts at: "+str(self.start_frame(gm))+" ends at: "+str(self.end_frame(gm))

        print s

    def append_left(self, vertex, gm, undo_action=False):
        warn("Not implemented for Track...")

    def append_right(self, vertex, gm, undo_action=False):
        warn("Not implemented for Track...")

    def pop_first(self, gm, undo_action=False):
        warn("Not implemented for Track...")

    def pop_last(self, gm, undo_action=False):
        warn("Not implemented for Track...")

    def merge(self, ch2, gm, undo_action=False):
        warn("Not implemented for Track...")

    def merge_and_interpolate(self, ch2, gm, undo_action=False):
        warn("Not implemented for Track...")

    def split_at(self, frame, gm):
        warn("Not implemented for Track...")

    def id(self):
        return self.id_

    def start_vertex_id(self):
        return self._data[0].nodes_[0]

    def end_vertex_id(self):
        return self._data[-1].nodes_[-1]

    def end_vertex(self, gm):
        return gm.g.vertex(self.end_vertex_id())

    def end_node(self):
        return self.end_vertex_id()

    def start_vertex(self, gm):
        return gm.g.vertex(self.start_vertex_id())

    def start_node(self):
        return self._data[0].start_vertex_id()

    def chunk_reconnect_(self, gm):
        warn("Not implemented for Track")

    def v_gen(self):
        for t in self._data:
            for v in t.nodes_:
                yield v

    def rid_gen(self, gm):
        for t in self._data:
            for id_ in t.nodes_:
                yield gm.region_id(id_)

    def v_id_in_t(self, frame, gm):
        tracklet = self._tracklet(frame, gm)

        frame = frame - self.start_frame(gm)
        if -1 < frame < len(tracklet.nodes_):
            return tracklet.nodes_[frame]
        else:
            return None

    def _tracklet(self, frame, gm):
        # TODO: improve...
        for t in self._data:
            if frame >= t.start_frame(gm) and frame < t.end_frame(gm):
                return t

    def r_id_in_t(self, frame, gm):
        tracklet = self._tracklet(frame, gm)
        # TODO: find proper chunk...
        return gm.region_id(tracklet.v_id_in_t(frame, gm))