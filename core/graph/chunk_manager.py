__author__ = 'flipajs'

from chunk import Chunk
from libs.intervaltree.intervaltree import IntervalTree
from utils.misc import print_progress
from tqdm import tqdm
import warnings

class ChunkManager:
    def __init__(self):
        # default value in graph properties will be 0, so we can easily test...
        self.id_ = 1
        self.chunks_ = {}
        self.itree = IntervalTree()
        self.eps1 = 0.01
        self.eps2 = 0.1
        self.track_refs = {}

    def __getitem__(self, index):
        # if index in self.track_refs:
        #     index = self.track_refs[index]

        return self.chunks_.get(index, None)

    def __len__(self):
        return len(self.chunks_)

    def new_track(self, track, gm):
        track.id_ = self.id_
        self.chunks_[self.id_] = track
        self._add_ch_itree(track, gm)

        for t in track._data:
            # self._try_ch_itree_delete()
            if not t.is_ghost():
                if t.id() in self.chunks_:
                    self.remove_chunk(t, gm)

                self.track_refs[t.id()] = track.id_

            # self.remove_tracklet_from_itree(t, gm)

        self.id_ += 1

        return track, self.id_ - 1

    def new_chunk(self, vertices_ids, gm, assign_color=True, origin_interaction=False):
        ch = Chunk(vertices_ids, self.id_, gm, origin_interaction=origin_interaction)
        self.chunks_[self.id_] = ch

        self._add_ch_itree(ch, gm)

        # assign chunk color
        # r1 = gm.region(vertices_ids[0])
        # rend = gm.region(vertices_ids[-1])
        # if gm.project.color_manager:
        #     ch.color, _ = gm.project.color_manager.new_track(r1.frame_, rend.frame_)

        self.id_ += 1

        return ch, self.id_ - 1

    def chunk_list(self):
        l = []
        for _, ch in self.chunks_.iteritems():
            l.append(ch)

        return l

    def _add_ch_itree(self, ch, gm):
        self.itree.addi(ch.start_frame(gm)-self.eps1, ch.end_frame(gm)+self.eps1, ch)

    def _try_ch_itree_delete(self, ch, gm):
        try:
            self.itree.removei(ch.start_frame(gm)-self.eps1, ch.end_frame(gm)+self.eps1, ch)
        except ValueError:
            # TODO: when is it happening?
            # print "delete failed"
            pass
        except KeyError:
            # print "delete failed"
            pass

    def remove_tracklet_from_itree(self, ch, gm):
        self._try_ch_itree_delete(ch, gm)

    def remove_chunk(self, ch, gm):
        if isinstance(ch, int):
            ch = self.chunks_[ch]

        self._try_ch_itree_delete(ch, gm)
        del self.chunks_[ch.id_]

    def update_chunk(self, ch, gm):
        if isinstance(ch, int):
            ch = self.chunks_[ch]

        self._try_ch_itree_delete(ch, gm)
        self._add_ch_itree(ch, gm)

    def get_chunks_from_intervals_(self, intervals):
        chunks = [i.data for i in intervals]

        return chunks

    def tracklets_in_frame(self, frame):
        intervals = self.itree[frame - self.eps2:frame + self.eps2]

        return self.get_chunks_from_intervals_(intervals)

    def chunks_in_frame(self, frame):
        warnings.warn("Deprecated, use tracklets_in_frame instead.")

        return self.tracklets_in_frame(frame)


    def undecided_singleid_tracklets_in_frame(self, frame):
        return filter(lambda x: len(x.P) == 0 and x.is_single(), self.chunks_in_frame(frame))

    def chunks_in_interval(self, start_frame, end_frame):
        intervals = self.itree[start_frame-self.eps2:end_frame+self.eps2]

        return self.get_chunks_from_intervals_(intervals)

    def tracklets_intersecting_t_gen(self, t, gm):
        for t_ in self.chunks_in_interval(t.start_frame(gm), t.end_frame(gm)):
            if t_ != t:
                yield t_

    def singleid_tracklets_intersecting_t_gen(self, t, gm):
        for t_ in self.tracklets_intersecting_t_gen(t, gm):
            if t_.is_single():
                yield t_

    def chunk_gen(self):
        return self.tracklet_gen()

    def tracklet_gen(self):
        for t in self.chunks_.itervalues():
            yield t

    def reset_itree(self, gm):
        self.itree = IntervalTree()

        chn = len(self)
        for i, ch in tqdm(enumerate(self.chunk_gen()), total=chn, desc='chunk_manager.reset_itree', leave=False):
            self._add_ch_itree(ch, gm)

    def add_single_vertices_chunks(self, p, frames=None):
        self.reset_itree(p.gm)

        nn = p.gm.g.num_vertices()

        for n in tqdm(p.gm.g.vertices(), total=p.gm.g.num_vertices()):
            if frames is None:
                if p.gm.get_chunk(n) is not None:
                    continue
            else:
                r = p.gm.region(n)
                if r.frame() not in frames or p.gm.get_chunk(n) is not None:
                    continue

            if not p.gm.g.vp['active'][n]:
                continue

            self.new_chunk([int(n)], p.gm)

        self.reset_itree(p.gm)

    def reset_PN_sets(self, project):
        full_set = set(range(len(project.animals)))
        for t in self.chunk_gen():
            t.P = set()
            t.N = set()

            if t.is_noise() or t.is_part() or t.is_undefined():
                t.N = set(full_set)

    def update_N_sets(self, project, update_N_callback=None):
        affecting = []
        for t in self.chunk_gen():
            if len(t.P):
                affecting.append((t, set(t.P)))

        self.reset_PN_sets(project)

        all_ids = set(range(len(project.animals)))
        for t, id_set in affecting:
            t.P = id_set
            t.N = all_ids.difference(id_set)
            # self.lp.assign_identity(id_, t)

        if update_N_callback is not None:
            for tracklet, id_set in affecting:
                for t in self.get_affected_undecided_tracklets(tracklet, project):
                        update_N_callback(id_set, t)

    def get_affected_undecided_tracklets(self, tracklet, project):
        num_animals = len(project.animals)
        affected = set(self.chunks_in_interval(tracklet.start_frame(project.gm),
                                                     tracklet.end_frame(project.gm)))

        return filter(lambda x: (x.is_single() or x.is_multi()) and not x.is_id_decided(num_animals), affected)
