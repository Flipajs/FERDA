from core.graph.complete_set import CompleteSet

__author__ = 'flipajs'

from os.path import join
from chunk import Chunk
from libs.intervaltree.intervaltree import IntervalTree
from utils.misc import print_progress
from tqdm import tqdm
import warnings
import numpy as np
import jsonpickle
import utils.load_jsonpickle


class ChunkManager(object):
    def __init__(self):
        # default value in graph properties will be 0, so we can easily test...
        self.id_ = 1
        self.chunks_ = {}
        self.itree = IntervalTree()  # stores chunks with start and end frame as a key
        self.eps1 = 0.01
        self.eps2 = 0.1
        self.track_refs = {}

    def __getitem__(self, index):
        # if index in self.track_refs:
        #     index = self.track_refs[index]

        return self.chunks_.get(index, None)

    def __len__(self):
        return len(self.chunks_)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['itree']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.itree = IntervalTree()

    def __str__(self):
        cardinalities, counts = np.unique([t.segmentation_class_str() for t in self.chunk_gen()], return_counts=True)
        cardinality_report = ' | '.join(['{}x {}'.format(count, cardinality) for cardinality, count in zip(cardinalities, counts)])
        return '{} tracklets, cardinality stats: {}'.format(len(self.chunks_), cardinality_report)

    @classmethod
    def from_dir(cls, directory, gm=None):
        chm = jsonpickle.decode(open(join(directory, 'tracklets.json'), 'r').read(), keys=True)
        if gm is not None:
            chm.set_graph_manager(gm)
        return chm

    def set_graph_manager(self, gm):
        for t in self.tracklet_gen():
            t.gm = gm
        self.reset_itree()

    def save(self, directory):
        open(join(directory, 'tracklets.json'), 'w').write(jsonpickle.encode(self, keys=True, warn=True))

    def new_track(self, track):
        track.id_ = self.id_
        self.chunks_[self.id_] = track
        self._add_ch_itree(track)

        for t in track._data:
            # self._try_ch_itree_delete()
            if not t.is_ghost():
                if t.id() in self.chunks_:
                    self.remove_tracklet(t)

                self.track_refs[t.id()] = track.id_

            # self.remove_tracklet_from_itree(t, gm)

        self.id_ += 1

        return track, self.id_ - 1

    def new_chunk(self, vertices_ids, gm, assign_color=True, origin_interaction=False):
        ch = Chunk(vertices_ids, self.id_, gm, origin_interaction=origin_interaction)
        self.chunks_[self.id_] = ch

        self._add_ch_itree(ch)

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

    def _add_ch_itree(self, ch):
        self.itree.addi(ch.start_frame() - self.eps1, ch.end_frame() + self.eps1, ch)

    def _try_ch_itree_delete(self, ch):
        try:
            self.itree.removei(ch.start_frame() - self.eps1, ch.end_frame() + self.eps1, ch)
        except ValueError:
            # TODO: when is it happening?
            # print "delete failed"
            pass
        except KeyError:
            # print "delete failed"
            pass

    def remove_tracklet_from_itree(self, ch):
        self._try_ch_itree_delete(ch)

    def remove_tracklet(self, t):
        if isinstance(t, int):
            t = self.chunks_[t]

        self._try_ch_itree_delete(t)
        del self.chunks_[t.id_]

    def update_chunk(self, t):
        if isinstance(t, int):
            t = self.chunks_[t]

        self._try_ch_itree_delete(t)
        self._add_ch_itree(t)

    def get_tracklets_from_intervals_(self, intervals):
        return [i.data for i in intervals]

    def tracklets_in_frame(self, frame):
        intervals = self.itree[frame - self.eps2:frame + self.eps2]

        return self.get_tracklets_from_intervals_(intervals)

    def undecided_singleid_tracklets_in_frame(self, frame):
        return filter(lambda x: len(x.P) == 0 and x.is_single(), self.tracklets_in_frame(frame))

    def get_tracklets_in_interval(self, start_frame, end_frame):
        return self.get_tracklets_from_intervals_(self.itree[start_frame-self.eps2:end_frame+self.eps2])

    def tracklets_intersecting_t_gen(self, t):
        for t_ in self.get_tracklets_in_interval(t.start_frame(), t.end_frame()):
            if t_ != t:
                yield t_

    def singleid_tracklets_intersecting_t_gen(self, t):
        for t_ in self.tracklets_intersecting_t_gen(t):
            if t_.is_single():
                yield t_

    def chunk_gen(self):
        return self.tracklet_gen()

    def tracklet_gen(self):
        for t in self.chunks_.itervalues():
            yield t

    def reset_itree(self):
        self.itree = IntervalTree()

        chn = len(self)
        for i, ch in tqdm(enumerate(self.chunk_gen()), total=chn, desc='ChunkManager rebuilding interval tree', leave=False):
            self._add_ch_itree(ch)

    def add_single_vertices_tracklets(self, gm, frames=None):
        self.reset_itree()

        for n in tqdm(gm.g.vertices(), total=gm.g.num_vertices()):
            if frames is None:
                if gm.get_tracklet(n) is not None:
                    continue
            else:
                r = gm.region(n)
                if r.frame() not in frames or gm.get_tracklet(n) is not None:
                    continue

            if not gm.g.vp['active'][n]:
                continue

            self.new_chunk([int(n)], gm)

        self.reset_itree()

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
                for t in self.get_affected_undecided_tracklets(tracklet):
                        update_N_callback(id_set, t)

    def get_affected_undecided_tracklets(self, tracklet):
        affected = set(self.get_tracklets_in_interval(tracklet.start_frame(),
                                                      tracklet.end_frame()))

        return filter(lambda x: (x.is_single() or x.is_multi()) and not x.is_id_decided(), affected)

    def get_conflicts(self, num_objects, verbose=False):
        # pn_sets_lengths = np.array([len(t.P) + len(t.N) for t in self.tracklet_gen()])
        # if np.any(pn_sets_lengths != num_objects) and verbose:
        #     print('{}/{} tracklets have inconsistent PN sets cardinalities.'.format(
        #         np.count_nonzero(pn_sets_lengths != num_objects),
        #         len(self)))

        min_frame = min([t.start_frame() for t in self.tracklet_gen()])
        max_frame = max([t.end_frame() for t in self.tracklet_gen()])
        tracklets = []
        undecided_tracklets = []
        for t in self.tracklet_gen():
            if t.is_only_one_id_assigned(num_objects):
                tracklets.append(t)
            else:
                undecided_tracklets.append(t)
        if verbose:
            print('{} undecided tracklet(s).'.format(len(undecided_tracklets)))

        conflicts = {}
        for frame in xrange(min_frame, max_frame + 1):
            decided = filter(lambda x: x.is_only_one_id_assigned(num_objects), self.tracklets_in_frame(frame))
            obj_ids = [next(iter(t.P)) for t in decided]
            if len(obj_ids) != len(np.unique(obj_ids)):
                if verbose:
                    print('Duplicated object ids at frame {}'.format(frame))
                conflicts[frame] = decided
        return undecided_tracklets, conflicts

    def get_complete_sets(self, project):
        """
        Complete set generator.

        Complete set is set of tracklets where set cardinality is equal to the number of objects in the scene.

        :param project:
        :return: generates list of tracklets
        """
        from sys import maxint

        num_animals = len(project.animals)
        frame = 0
        while frame < project.num_frames():
            s = self.tracklets_in_frame(frame)
            s = filter(lambda x: x.is_single(), s)

            if len(s) == num_animals:
                yield CompleteSet(s)

            min_end_frame = maxint
            for tracklet in s:
                min_end_frame = min(min_end_frame, tracklet.end_frame())

            frame = min_end_frame + 1

    def get_random_regions(self, n, gm):
        import random

        tracklet_ids = self.chunks_.keys()
        regions = []
        vertices = []
        for _ in range(n):
            tracklet = self[random.choice(tracklet_ids)]
            vertex = tracklet[random.randint(0, len(tracklet) - 1)]
            regions.append(gm.region(vertex))
            vertices.append(int(vertex))
        return regions, vertices

    def show_tracklets(self, rm):
        import matplotlib.pylab as plt
        for t in self.chunk_gen():
            yx = np.array([r.centroid() for r in t.r_gen(rm)])
            plt.plot(yx[:, 1], yx[:, 0])
