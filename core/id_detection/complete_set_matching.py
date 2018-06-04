import random
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.misc import imread
from tqdm import tqdm
import pickle
from utils.video_manager import get_auto_video_manager
import logging


logger = logging.getLogger(__name__)


class CompleteSetMatching:
    def __init__(self, project, lp, descriptors, quality_threshold=0.1, quality_threshold2=0.01, tracks={},
                 tracklets_2_tracks={}, prototypes={}):
        self.prototype_distance_threshold = np.inf # ignore
        self.QUALITY_THRESHOLD = quality_threshold
        self.QUALITY_THRESHOLD2 = quality_threshold2
        self.p = project
        self.lp = lp
        self.get_probs = self.lp._get_tracklet_proba
        self.get_p1s = self.lp.get_tracklet_p1s
        if isinstance(descriptors, str):
            with open(descriptors, 'rb') as fr:
                self.descriptors = pickle.load(fr)
        else:
            self.descriptors = descriptors

        self.new_track_id = 0
        self.tracks = tracks
        self.tracklets_2_tracks = tracklets_2_tracks
        self.prototypes = prototypes
        self.tracks_link = {}

        self.update_distances = []
        self.update_weights = []

    def start_matching_process(self):
        track_CSs = self.find_track_cs()

        qualities, track_CSs = self.sequential_matching(track_CSs)

        logger.debug('track complete sets: %s', str([sorted(CS) for CS in track_CSs]))

        # support = {}
        # for t in self.p.chm.chunk_gen():
        #     if len(t.P):
        #         t_identity = list(t.P)[0]
        #         support[t_identity] = support.get(t_identity, 0) + len(t)
        #
        # print support
        #
        # self.remap_ids_from_0(support)

        # self.p.save()
        # import sys
        # sys.exit()

        ##### now do CS of tracks to tracks matching
        self.tracks_CS_matching(track_CSs)

        ##### then do the rest... bruteforce approach
        # 1. find best Track CS
        # 2. for each tracklet, try to find best track, computed probability, sort by probs...

        best_CS = None
        best_support = 0
        for CS in track_CSs:
            val = self.track_support(CS)
            if val > best_support:
                best_CS = CS
                best_support = val

        logger.info('BEGINNING of best_set matching')

        # the problem is that we already have IDs in track.P even thought they are not matched
        # thus we need to reset .P sets
        for t in self.p.chm.tracklet_gen():
            try:
                if list(t.P)[0] not in best_CS:
                    t.P = set()
                    logger.debug('track id {}'.format(t.id()))
            except:
                pass

        # update N sets for unassigned tracklets in relations to best_CS track ids
        for tracklet, track_id in self.tracklets_2_tracks.iteritems():
            self.add_to_N_set(track_id, tracklet)

        # go through tracklets, find biggest set and do matching..
        # 1) choose tracklet
        #       longest?
        #       for now, process as it is in chunk_generator

        num_undecided = 0
        for t in self.p.chm.tracklet_gen():
            # TODO: what about matching unmatched Tracks as well?
            if not t.is_single() or t.is_id_decided() or t.is_origin_interaction():
                continue

            if t not in self.tracklets_2_tracks:
                self.register_tracklet_as_track(t)

            # 2) find biggest set
            best_set = self.find_biggest_undecided_tracklet_set(t)

            # TODO: best_set - replace tracklets with tracks?
            #             / Users / flipajs / Documents / dev / ferda / core / graph / chunk_manager.py:112: UserWarning: Deprecated, use
            #             tracklets_in_frame
            #             instead.
            #             warnings.warn("Deprecated, use tracklets_in_frame instead.")
            #         Traceback(most
            #         recent
            #         call
            #         last):
            #         File
            #         "/Users/flipajs/Documents/dev/ferda/core/id_detection/complete_set_matching.py", line
            #         1458, in < module >
            #         csm.start_matching_process()
            #
            #     File
            #     "/Users/flipajs/Documents/dev/ferda/core/id_detection/complete_set_matching.py", line
            #     122, in start_matching_process
            #     P_a = self.prototypes_distance_probabilities(best_set, best_CS)
            #
            #
            # File
            # "/Users/flipajs/Documents/dev/ferda/core/id_detection/complete_set_matching.py", line
            # 1029, in prototypes_distance_probabilities
            # prob = prob_prototype_represantion_being_same_id_set(self.prototypes[track1], self.prototypes[track2])
            # KeyError: < core.graph.chunk.Chunk
            # instance
            # at
            # 0x11062add0 >

            track_best_set = []
            for tracklet in best_set:
                track_best_set.append(self.tracklets_2_tracks[tracklet])

            best_set = track_best_set

            prohibited_ids = {}
            for t_ in best_set:
                prohibited_ids[t_] = []

                for tracklet in self.tracks[t_]:
                    for test_t in self.p.chm.tracklets_intersecting_t_gen(tracklet, self.p.gm):
                        if test_t.is_single() and self.tracklets_2_tracks[test_t] == t_:
                            continue

                        if len(test_t.P):
                            prohibited_ids[t_].append(list(test_t.P)[0])

            # 3) compute matching
            P_a = self.prototypes_distance_probabilities(best_set, best_CS)

            # TODO: add spatial cost as well
            # invert...
            P = 1 - P_a
            # prohibit already used IDs
            for i, t_ in enumerate(best_set):
                for j, track_id in enumerate(best_CS):
                    if track_id in prohibited_ids[t_]:
                        # ValueError: matrix contains invalid numeric entries was thrown in case of np.inf... so trying huge number instead..
                        P[i, j] = 1000000.0

            from scipy.optimize import linear_sum_assignment
            assert np.sum(P < 0) == 0
            row_ind, col_ind = linear_sum_assignment(P)

            perm = []
            for rid, cid in zip(row_ind, col_ind):
                perm.append((best_set[rid], best_CS[cid]))

            x_ = 1 - P[row_ind, col_ind]
            quality = (x_.min(), x_.sum() / float(len(x_)))

            # np.set_printoptions(precision=3)
            # print P
            # print best_set, best_CS
            # print perm, quality

            # 4) accept?
            if quality[1] > self.QUALITY_THRESHOLD2:
                for (unassigned_track_id, track_id) in perm:
                    logger.debug('{} -> {}'.format(unassigned_track_id, track_id))
                    # print "[{} |{}| (te: {})] -> {}".format(tracklet.id(), len(tracklet), tracklet.end_frame(self.p.gm), track_id)
                    # tracklets_track = self.tracklets_2_tracks[tracklet]
                    for tracklet in self.tracks[unassigned_track_id]:
                        tracklet.id_decision_info = 'best_set_matching'

                    self.merge_tracks(track_id, unassigned_track_id)

                    # # propagate
                    # # TODO: add sanity checks?
                    # for t_ in self.p.chm.singleid_tracklets_intersecting_t_gen(tracklet, self.p.gm):
                    #     if t_ != tracklet:
                    #         self.add_to_N_set(track_id, t_)

            else:
                logger.debug('quality below QUALITY_THRESHOLD2')
                num_undecided += 1

        logger.debug('#UNDECIDED: {}'.format(num_undecided))
        # self.single_track_assignment(best_CS, prototypes, tracklets_2_tracks)

        #### visualize and stats
        from utils.rand_cmap import rand_cmap

        new_cmap = rand_cmap(self.new_track_id+1, type='bright', first_color_black=True, last_color_black=False)
        logger.debug('#IDs: {}'.format(self.new_track_id+1))
        support = {}
        tracks = {}
        tracks_mean_desc = {}
        for t in self.p.chm.chunk_gen():
            if len(t.P):
                t_identity = list(t.P)[0]
                support[t_identity] = support.get(t_identity, 0) + len(t)
                if t_identity not in tracks:
                    tracks[t_identity] = []
                #
                tracks[t_identity].append(t.id())
                #
                t_desc_w = self.get_mean_descriptor(t) * len(t)
                if t_identity not in tracks_mean_desc:
                    tracks_mean_desc[t_identity] = t_desc_w
                else:
                    tracks_mean_desc[t_identity] += t_desc_w
                #
                # plt.scatter(t.start_frame(self.p.gm), t_identity, c=new_cmap[t_identity], edgecolor=[0.,0.,0.,.3])
                # plt.plot([t.start_frame(self.p.gm), t.end_frame(self.p.gm)+0.1], [t_identity, t_identity],
                #          c=new_cmap[t_identity],
                #          path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
            else:
                # if t.is_noise() or len(t) < 5:
                #     continue
                # if t.is_single():
                #     c = [0, 1, 0, .3]
                # else:
                #     c = [0, 0, 1, .3]
                #
                # y = t.id() % self.new_track_id
                # plt.scatter(t.start_frame(self.p.gm), y, c=c, marker='s', edgecolor=[0., 0., 0., .1])
                # plt.plot([t.start_frame(self.p.gm), t.end_frame(self.p.gm) + 0.1], [y, y],
                #          c=c,
                #          linestyle='-')
                pass

        # plt.grid()

        logger.debug("SUPPORT")
        for id in sorted(support.keys()):
            logger.debug("{}: {}, #{} ({})".format(id, support[id], len(tracks[id]), tracks[id]))

        self.remap_ids_from_0(support)

        # qualities = np.array(qualities)
        # plt.figure()
        # plt.plot(qualities[:, 0])
        # plt.grid()
        # plt.figure()
        # plt.plot(qualities[:, 1])
        # plt.grid()
        #
        # plt.figure()
        # plt.show()

        # mean_ds = []
        # for id_, mean in tracks_mean_desc.iteritems():
        #     mean_ds.append(mean/float(support[id]))

        logger.debug("track ids order: {}, length: {}".format(list(tracks_mean_desc.iterkeys()), len(tracks)))
        # from scipy.spatial.distance import pdist, squareform
        # plt.imshow(squareform(pdist(mean_ds)), interpolation='nearest')
        # plt.show()

        # for i in range(50, 60):
        #     print "CS {}, CS {}".format(0, i)
        #     perm, quality = self.cs2cs_matching_ids_unknown(CSs[0], CSs[i])
        #     for (t1, t2) in perm:
        #         print t1.id(), " -> ", t2.id()
        #
        #     print quality

    def remap_ids_from_0(self, support):
        map_ = {}
        for id in range(len(self.p.animals)):
            for t in self.p.chm.chunk_gen():
                if id in t.P:
                    t.P = set([-id])
                    map_[id] = -id

        for new_id, id in enumerate(sorted(support, key=support.get)[-len(self.p.animals):]):
            for t in self.p.chm.chunk_gen():
                if id in map_:
                    id = map_[id]

                if id in t.P:
                    t.P = set([new_id])

    def find_biggest_undecided_tracklet_set(self, t):
        all_intersecting_t = list(self.p.chm.singleid_tracklets_intersecting_t_gen(t, self.p.gm))
        # skip already decided...
        all_intersecting_t = filter(lambda x: len(x.P) == 0, all_intersecting_t)
        t_start = t.start_frame(self.p.gm)
        t_end = t.end_frame(self.p.gm)
        #       for simplicity - find frame with biggest # of intersecting undecided tracklets
        important_frames = {t_start: 1, t_end: 1}
        important_frames_score = {t_start: len(t), t_end: len(t)}
        for t_ in all_intersecting_t:
            ts = t_.start_frame(self.p.gm)
            te = t_.end_frame(self.p.gm)
            if ts >= t_start:
                important_frames.setdefault(ts, 0)
                important_frames_score.setdefault(ts, 0)
                important_frames_score[ts] += len(t_)
                important_frames[ts] += 1

            if te <= t_end:
                important_frames.setdefault(te, 0)
                important_frames_score.setdefault(te, 0)
                important_frames_score[te] += len(t_)
                important_frames[te] += 1
        best_frame = -1
        best_val = 0
        best_score = 0
        for frame, val in important_frames.iteritems():
            if val >= best_val:
                if important_frames_score[frame] > best_score:
                    best_frame = frame
                    best_val = val
                    best_score = important_frames_score[frame]

        return self.p.chm.undecided_singleid_tracklets_in_frame(best_frame)

    def single_track_assignment(self, best_CS, prototypes, tracklets_2_tracks):
        # update N sets for unassigned tracklets in relations to best_CS track ids
        for tracklet, track_id in tracklets_2_tracks.iteritems():
            self.add_to_N_set(track_id, tracklet)
        probabilities = {}
        decisioins = {}
        # moreless a cache...
        tracklets_prototypes = {}
        probs = []
        probs2 = []
        lengths = []
        tracklets = []
        best_track_ids = []
        for t in self.p.chm.chunk_gen():
            # TODO: what about matching unmatched Tracks as well?
            if t in tracklets_2_tracks or not t.is_single():
                continue

            if t not in tracklets_prototypes:
                tracklets_prototypes[t.id()] = self.get_track_prototypes(t)

            best_p, best_track = self.find_best_track_for_tracklet(best_CS, probs2, prototypes, t, tracklets_prototypes)

            probabilities[t] = best_p
            decisioins[t] = best_track

            # probs.append(best_p)
            best_track_ids.append(best_track)
            lengths.append(len(t))
            tracklets.append(t)

        probs2 = np.array(probs2)
        # probs = np.array(probs)
        probs = probs2
        tracklets = np.array(tracklets)
        ids = np.argsort(-probs)
        best_track_ids = np.array(best_track_ids)
        import warnings

        while len(probs):
            id_ = np.argmax(probs)
            if probs[id_] > 0.5:
                probs.remove(id_)
                best_track_ids

                t = tracklets[id_]
                track_id = best_track_ids[id_]
                if track_id in t.N:
                    warnings.warn("IN N ... warning tid: {}, prob: {}".format(t.id()), probs[i])

                logger.debug('{} {}'.format(probs[id_], tracklets[id_]))

                t.P = set([track_id])
                t.id_decision_info = 'single_decision'

                self.add_to_N_set(track_id, t)
                for t_ in self.lp._get_affected_undecided_tracklets(t):
                    pass
                    # TODO: propagate...
                    # update probabilities...

        plt.figure()

        plt.scatter(np.arange(len(probs)), probs, c='r')
        plt.scatter(np.arange(len(probs)), probs2, c='g')

    def find_best_track_for_tracklet(self, best_CS, probs2, prototypes, t, tracklets_prototypes):
        best_p = 0
        best_track = None
        prob_vec = [0] * len(self.p.animals)
        for i, track_id in enumerate(best_CS):
            # skip restricted
            if track_id in t.N:
                continue

            # TODO: certainty?
            prob = self.prototypes_match_probability(prototypes[track_id], tracklets_prototypes[t.id()])
            prob_vec[i] = prob

            if prob > best_p:
                best_p = prob
                best_track = track_id
        prob_vec = np.array(prob_vec) / np.sum(prob_vec)
        probs2.append(max(prob_vec))
        return best_p, best_track

    def sequential_matching(self, CSs):
        """
        Try to match consecutive (nearest in time) complete sets of tracklets.

        :param CSs:
        :return: qualities, track_CSs
        """
        logger.info("beginning of sequential matching")

        track_CSs = [[]]

        for i, track in enumerate(CSs[0]):
            # new_track_id = self.register_tracklet_as_track(t)
            for t in self.tracks[track]:
                t.id_decision_info = 'sequential_matching'

            track_CSs[-1].append(track)

        qualities = []
        for i in tqdm(range(len(CSs) - 1), desc='sequential matching'):
            logger.debug("CS {}, CS {}".format(i, i + 1))

            # done previously...
            # # first create new virtual tracks and their prototypes for CSs[i+1] which are not already in tracks
            # for track in CSs[i + 1]:
            #     if t in self.tracklets_2_tracks:
            #         continue
            #
            #     self.register_tracklet_as_track(t)

            self.update_tracks_from_links(CSs, i)
            self.update_tracks_from_links(CSs, i+1)

            # perm, quality = self.cs2cs_matching_descriptors_and_spatial(CSs[i], CSs[i+1])
            perm, quality = self.cs2cs_matching_prototypes_and_spatial(CSs[i], CSs[i + 1])

            # cs1_max_frame = 0
            # cs2_min_frame = np.inf
            # dividing_frame = 0
            # for (track1, track2) in perm:
            #     if track1 == track2:
            #         break
            #
            #     cs1_max_frame = max(cs1_max_frame, self.track_start_frame(track1))
            #     cs2_min_frame = min(cs2_min_frame, self.track_end_frame(track2))
            #
            #     dividing_frame = max(dividing_frame, self.track_start_frame(track2))
            #
            # print "cs1 max frame: {}, cs2 min frame: {}".format(cs1_max_frame, cs2_min_frame)

            # TODO: threshold 1-

            not_same = 0
            c = [0. + 1 - quality[1], quality[1], 0., 0.2]
            # propagate IDS if quality is good enough:
            if quality[1] > self.QUALITY_THRESHOLD:
                # TODO: transitivity? when t1 -> t2 assignment uncertain, look on ID probs for t2->t3 and validate wtih t1->t3

                for (track1, track2) in perm:
                    # print "[{} |{}| (te: {})] -> {} |{}| (ts: {})".format(track1, len(track1), self.track_end_frame(track1),
                    #                                                       track2, len(track2), self.track_start_frame(track2))
                    logger.debug("[{} -> {}]".format(track1, track2))

                    if track1 != track2:
                        self.merge_tracks(track1, track2, decision_info='sequential_matching')
            else:
                logger.debug('QUALITY BELOW')
                # c = [1., 0.,0.,0.7]

                track_CSs.append([])
                for pair in perm:
                    track2 = pair[1]
                    # if len(t.P) == 0:
                    #     t.P = set([self.new_track_id])
                    #     self.new_track_id += 1

                    track_CSs[-1].append(track2)

                for pair in perm:
                    if pair[0] != pair[1]:
                        not_same += 1

            # plt.plot([dividing_frame, dividing_frame], [-5, -5 + 4.7 * quality[1]], c=c)
            # plt.plot([dividing_frame, dividing_frame], [0, self.new_track_id - 1 + not_same], c=c)

            qualities.append(quality)

        tracks_unassigned_len = 0
        tracks_unassigned_num = 0
        for t in self.p.chm.chunk_gen():
            if t.is_single() and t not in self.tracklets_2_tracks:
                tracks_unassigned_len += len(t)
                tracks_unassigned_num += 1
        num_prototypes = 0
        for prots in self.prototypes.itervalues():
            num_prototypes += len(prots)
        logger.info("sequential CS matching done...")
        logger.debug("#tracks: {}, #tracklets2tracks: {}, unassigned #{} len: {}, #prototypes: {}".format(
                     len(self.tracks), len(self.tracklets_2_tracks), tracks_unassigned_num,
                     tracks_unassigned_len, num_prototypes))

        return qualities, track_CSs

    def update_tracks_from_links(self, CSs, i):
        for j in range(len(CSs[i])):
            track_id = CSs[i][j]
            if track_id not in self.tracks:
                while True:
                    if track_id in self.tracks_link:
                        track_id = self.tracks_link[track_id]
                    else:
                        break

            CSs[i][j] = track_id

    def register_tracklet_as_track(self, t):
        if t not in self.tracklets_2_tracks:
            self.tracks[self.new_track_id] = [t]
            self.tracklets_2_tracks[t] = self.new_track_id
            self.prototypes[self.new_track_id] = self.get_track_prototypes(t)
            t.P = set([self.new_track_id])

            self.new_track_id += 1

        return self.tracklets_2_tracks[t]

    def merge_tracklets(self, t1, t2):
        track1 = list(t1.P)[0]
        track2 = self.tracklets_2_tracks[t2]
        self.update_prototypes(self.prototypes[track1], self.prototypes[track2])

        for t in self.tracks[track2]:
            if t == track2:
                import sys
                import warnings
                warnings.warn("Infinite cycle in Merge tracklets =/")
                sys.exit()
            self.tracks[track1].append(t)
            self.tracklets_2_tracks[t2] = track1

            t.P = set(t1.P)

        del self.tracks[track2]
        del self.prototypes[track2]

    def add_to_N_set(self, track_id, tracklet):
        for t in self.p.chm.chunks_in_interval(tracklet.start_frame(self.p.gm), tracklet.end_frame(self.p.gm)):
            if t.is_single() and t != tracklet:
                t.N.add(track_id)

    def tracks_CS_matching(self, track_CSs):
        # 1. get CS of tracks (we already have them in track_CSs from sequential process.
        # 2. sort CS by sum of track lengths
        # 3. try to match all others to this one (spatio-temporal term might be switched on for close tracks?)
        # 4. if any match accepted update and goto 2.
        # 5. else take second biggest and goto 3.
        # 6. end if only one CS, or # of CS didn't changed...

        logger.info("beginning of global matching")
        updated = True
        with tqdm(total=len(track_CSs), desc='global matching') as pbar:
            while len(track_CSs) > 1 and updated:
                updated = False

                # 2. sort CS by sum of track lengths
                ordered_CSs = self.sort_track_CSs(track_CSs)

                # 3.
                for i in range(len(ordered_CSs)-1):
                    pivot = ordered_CSs[i]

                    best_quality = 0
                    best_perm = None
                    best_CS = None

                    for CS in ordered_CSs[i+1:]:
                        perm, quality = self.cs2cs_matching_prototypes_and_spatial(
                            pivot, CS, use_spatial_probabilities=False
                        )

                        if quality[1] > best_quality:
                            best_quality = quality[1]
                            best_perm = perm
                            best_CS = CS

                    if best_quality > self.QUALITY_THRESHOLD:
                        logger.debug("Best track CS match accepted. {}, {}".format(best_perm, best_quality))
                        self.merge_track_CSs(best_perm)
                        track_CSs.remove(best_CS)
                        self.update_all_track_CSs(best_perm, track_CSs)
                        updated = True
                        pbar.update()
                        break
                    else:
                        logger.debug("Best track CS match rejected. {}, {}".format(perm, quality))

    def update_all_track_CSs(self, pair, track_CSs):
        """

        :param pair: pair of complete sets, list of tuples, e.g. [(1, 416), ...]
        :param track_CSs: list of lists of tracklet ids, e.g. [[141, 404, 1, 93, 6], [141, 93, 1, 404, 6], ...]
        """
        for CS_for_update in track_CSs:
            size_before = len(CS_for_update)
            for track_id1, track_id2 in pair:
                for i, track_id in enumerate(CS_for_update):
                    if track_id == track_id2:
                        CS_for_update[i] = track_id1

            # TODO: this means conflict...
            if len(set(CS_for_update)) != size_before:
                logger.debug("CONFLICT {}".format(CS_for_update))
            assert len(set(CS_for_update)) == size_before

    def merge_track_CSs(self, perm):
        # keep attention, here we have tracks, not tracklets...
        for (track1_id, track2_id) in perm:
            logger.debug("{} -> {}".format(track1_id, track2_id))

            # if merge...
            if track1_id != track2_id:
                self.merge_tracks(track1_id, track2_id)

    def merge_tracks(self, track1_id, track2_id, decision_info='global_matching'):
        self.update_prototypes(self.prototypes[track1_id], self.prototypes[track2_id])
        for tracklet in self.tracks[track2_id]:
            self.tracklets_2_tracks[tracklet] = track1_id
            self.tracks[track1_id].append(tracklet)
            tracklet.P = set([track1_id])
            tracklet.id_decision_info = decision_info

        self.tracks_link[track2_id] = track1_id

        del self.tracks[track2_id]
        del self.prototypes[track2_id]

    def sort_track_CSs(self, track_CSs):
        values = []
        for CS in track_CSs:
            val = self.track_support(CS)

            values.append(val)

        values_i = reversed(sorted(range(len(values)), key=values.__getitem__))
        CS_sorted = []
        for i in values_i:
            logger.debug('%s, %s', str(track_CSs[i]), str(values[i]))
            CS_sorted.append(track_CSs[i])

        return CS_sorted

    def track_support(self, CS):
        val = 0
        for track_id in CS:
            for tracklet in self.tracks[track_id]:
                val += len(tracklet)
        return val

    def find_track_cs(self):
        """
        Find complete sets of tracklets.

        Complete set C is set of tracklets where |C| = number of objects. Then it is guaranteed that no object is
        missing in the set.

        :return: list of lists of tracklet ids, e.g. [[141, 404, 1, 93, 6], [141, 93, 1, 404, 6], ...]
        """
        for t in self.p.chm.tracklet_gen():
            if t.is_single():
                self.register_tracklet_as_track(t)

        unique_tracklets = set()
        CSs = []
        vm = get_auto_video_manager(self.p)
        total_frame_count = vm.total_frame_count()

        frame = 0
        i = 0
        old_frame = 0
        logger.info("analysing project, searching for complete sets")
        with tqdm(total=total_frame_count, desc='searching for complete sets') as pbar:
            while True:
                group = self.p.chm.tracklets_in_frame(frame)
                if len(group) == 0:
                    break

                singles_group = filter(lambda x: x.is_single(), group)

                if len(singles_group) == len(self.p.animals) and min([len(t) for t in singles_group]) >= 1:
                    # tracklets to tracks
                    singles_group = map(lambda x: self.tracklets_2_tracks[x], singles_group)
                    if len(CSs) == 0 or singles_group != CSs[-1]:
                        CSs.append(singles_group)

                    for t in singles_group:
                        unique_tracklets.add(t)

                    frame = min([self.track_end_frame(t) for t in singles_group]) + 1
                else:
                    frame = min([t.end_frame(self.p.gm) for t in group]) + 1

                i += 1
                pbar.update(frame - old_frame)
                old_frame = frame

        return CSs

    def classify_cs(self):
        # matching to IDs, classification but more robust - we want to use each class once
        
        # ? how to deal with 
        
        pass

    def cs2cs_matching_ids_unknown(self, cs1, cs2):
        # TODO: probability is better than cost, easier to interpret
        # get distance costs
        # get ID assignments costs
        # solve matching
        # register matched tracklets to have the same virtual ID
        perm = []

        cs1, cs2, cs_shared = self.remove_straightforward_tracklets(cs1, cs2)
        if len(cs1) == 1:
            perm.append((cs1[0], cs2[0]))
            quality = [1.0, 1.0]
        else:
            P_a = self.appearance_probabilities(cs1, cs2)
            P_s = self.spatial_probabilities(cs1, cs2, lower_bound=0.5)

            # 1 - ... it is minimum weight matching
            P = 1 - np.multiply(P_a, P_s)

            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(P)

            for rid, cid in zip(row_ind, col_ind):
                perm.append((cs1[rid], cs2[cid]))

            x_ = 1 - P[row_ind, col_ind]
            quality = (x_.min(), x_.sum() / float(len(x_)))

        for t in cs_shared:
            perm.append((t, t))

        return perm, quality

    def cs2cs_matching_descriptors_and_spatial(self, cs1, cs2):
        # TODO: probability is better than cost, easier to interpret
        # get distance costs
        # get ID assignments costs
        # solve matching
        # register matched tracklets to have the same virtual ID
        perm = []

        cs1, cs2, cs_shared = self.remove_straightforward_tracklets(cs1, cs2)
        if len(cs1) == 1:
            perm.append((cs1[0], cs2[0]))
            quality = [1.0, 1.0]
        else:
            P_a = self.appearance_distance_probabilities(cs1, cs2)
            P_s = self.spatial_probabilities(cs1, cs2, lower_bound=0.5)

            # 1 - ... it is minimum weight matching
            P = 1 - np.multiply(P_a, P_s)

            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(P)

            for rid, cid in zip(row_ind, col_ind):
                perm.append((cs1[rid], cs2[cid]))

            x_ = 1 - P[row_ind, col_ind]
            quality = (x_.min(), x_.sum() / float(len(x_)))

        for t in cs_shared:
            perm.append((t, t))

        return perm, quality

    def cs2cs_matching_prototypes_and_spatial(self, cs1, cs2, use_spatial_probabilities=True):
        perm = []
        cs1, cs2, cs_shared = self.remove_straightforward_tracklets(cs1, cs2)
        # assert len(cs1) == len(cs2)

        if len(cs1) == 1:
            perm.append((cs1[0], cs2[0]))
            quality = (1.0, 1.0)
        elif len(cs1) == 0:
            quality = (1.0, 1.0)
        else:
            assert len(cs2) > 1
            P_a = self.prototypes_distance_probabilities(cs1, cs2)

            if use_spatial_probabilities:
                P_s = self.spatial_probabilities(cs1, cs2, lower_bound=0.5)

                # 1 - ... it is minimum weight matching
                P = 1 - np.multiply(P_a, P_s)
            else:
                P = 1 - P_a

            from scipy.optimize import linear_sum_assignment

            assert np.sum(P < 0) == 0

            row_ind, col_ind = linear_sum_assignment(P)

            for rid, cid in zip(row_ind, col_ind):
                perm.append((cs1[rid], cs2[cid]))

            x_ = 1 - P[row_ind, col_ind]
            quality = (x_.min(), x_.sum() / float(len(x_)))

        for t in cs_shared:
            perm.append((t, t))

        return perm, quality

    def track_end_frame(self, track):
        return max([t.end_frame(self.p.gm) for t in self.tracks[track]])

    def track_start_frame(self, track):
        return min([t.start_frame(self.p.gm) for t in self.tracks[track]])

    def track_end_node(self, track):
        end_node = None
        end_frame = 0

        for t in self.tracks[track]:
            t_end_f = t.end_frame(self.p.gm)
            if t_end_f > end_frame:
                end_frame = t_end_f
                end_node = t.end_node()

        return end_node

    def track_start_node(self, track):
        start_node = None
        start_frame = np.inf

        for t in self.tracks[track]:
            t_start_f = t.start_frame(self.p.gm)
            if t_start_f < start_frame:
                start_frame = t_start_f
                start_node = t.start_node()

        return start_node

    def spatial_probabilities(self, cs1, cs2, lower_bound=0.5):
        # should be neutral if temporal distance is too big
        # should be restrictive when spatial distance is big
        max_d = self.p.solver_parameters.max_edge_distance_in_ant_length * self.p.stats.major_axis_median
        P = np.zeros((len(cs1), len(cs2)), dtype=np.float)

        for i, track1 in enumerate(cs1):
            t1_ef = self.track_end_frame(track1)
            for j, track2 in enumerate(cs2):
                if track1 == track2:
                    prob = 1.0
                else:
                    # TODO: solve it even for tracks not in sequence
                    temporal_d = self.track_start_frame(track2) - t1_ef

                    if temporal_d < 0:
                        prob = -np.inf
                    else:
                        t1_end_r = self.p.gm.region(self.track_end_node(track1))
                        t2_start_r = self.p.gm.region(self.track_start_node(track2))
                        spatial_d = np.linalg.norm(t1_end_r.centroid() - t2_start_r.centroid())

                        # should be there any weight?
                        spatial_d = spatial_d / float(max_d)

                        # TODO: what if it just makes something strange out of luck? E.G. Two distant CS with one tracklet which has perfect distance thus p~1.0and all others have ~0.5
                        if (1 - spatial_d) < 0:
                            val = 0
                        else:
                            val = (1 - spatial_d) ** temporal_d

                        prob = max(0.0, val)

                P[i, j] = prob


        # it might occur when t1 ends after t2 starts
        invalid = P < 0

        # minimize P_s impact when distance is too big
        P[P<lower_bound] = lower_bound
        P[invalid] = 0

        return P

    def remove_straightforward_tracklets(self, cs1, cs2):
        cs1 = set(cs1)
        cs2 = set(cs2)
        shared = cs1.intersection(cs2)

        return list(cs1.difference(shared)), list(cs2.difference(shared)), list(shared)

    def appearance_probabilities(self, cs1, cs2):
        # ...thoughts...
        # get probabilities for each tracklet
        # ? just probabilities? Or "race conditions term" included ?
        # in my opinion, race condition is already treated by matching
        # thus I suggest using only get_p1, including "homogenity" score

        C = np.zeros((len(cs1), len(cs2)), dtype=np.float)

        for i, t1 in enumerate(cs1):
            p1 = np.mean(self.get_probs(t1), axis=0)
            k1 = np.argmax(p1)
            val1 = p1[k1]

            for j, t2 in enumerate(cs2):
                p2 = np.mean(self.get_probs(t2), axis=0)
                k2 = np.argmax(p2)
                val2 = p2[k2]

                cost1 = val1 * p2[k1]
                cost2 = p1[k2] * val2

                likelihood = max(cost1, cost2)

                C[i, j] = likelihood

        return C

    def get_mean_descriptor(self, tracklet):
        descriptors = []
        for r_id in tracklet.rid_gen(self.p.gm):
            if r_id in self.descriptors:
                descriptors.append(self.descriptors[r_id])

        if len(descriptors) == 0:
            import warnings
            warnings.warn("descriptors missing for t_id: {}, creating zero vector".format(tracklet.id()))

            descriptors.append(np.zeros(32, ))


        descriptors = np.array(descriptors)

        res = np.mean(descriptors, axis=0)

        assert len(res) == 32

        return res

    def appearance_distance_probabilities(self, cs1, cs2):
        # returns distances to mean descriptors
        from scipy.spatial.distance import cdist

        cs1_descriptors = []
        for i, t1 in enumerate(cs1):
            cs1_descriptors.append(self.get_mean_descriptor(t1))

        cs2_descriptors = []
        for i, t2 in enumerate(cs2):
            cs2_descriptors.append(self.get_mean_descriptor(t2))

        C = cdist(cs1_descriptors, cs2_descriptors)

        max_d = 3.0
        C = C / max_d
        C = 1 - C

        return C

    def best_prototype(self, ps, p):
        best_d = np.inf
        best_w = 0
        best_i = 0

        for i, p_ in enumerate(ps):
            d, w = p_.distance_and_weight(p)
            if d < np.inf:
                best_d = d
                best_w = w
                best_i = i

        return best_d, best_w, best_i

    def prototypes_match_probability(self, ps1, ps2):
        # it is not symmetrical, so find best for each from the right (smaller) in left prototypes
        probability = 0

        # TODO: get lambda from siamese network measurements...
        lambda_ = 6.03

        W_ps1 = 0.0
        for p in ps1:
            W_ps1 += p.weight

        W_ps2 = 0.0
        for p in ps2:
            W_ps2 += p.weight

        # W = W_ps1 * W_ps2
        # assert W >= 0

        for p1 in ps1:
            for p2 in ps2:
                d = p1.distance(p2)
                probability += (p1.weight / W_ps1) * (p2.weight / W_ps2) * np.exp(-lambda_ * d)

        # probability /= len(ps1)

        assert 0 <= probability <= 1

        return probability

    def prototypes_distance__deprecated(self, ps1, ps2):
        final_d = 0
        final_w = 0

        # it is not symmetrical, so find best for each from the right (smaller) in left prototypes
        for p2 in ps2:
            best_d, _, _ = self.best_prototype(ps1, p2)

            alpha = final_w / float(final_w + p2.weight)
            final_d = alpha * final_d + (1 - alpha) * best_d
            final_w += p2.weight

        return final_d

    def update_prototypes(self, ps1, ps2):
        for i, p2 in enumerate(ps2):
            d, w, j = self.best_prototype(ps1, p2)

            # self.update_distances.extend([d] * p2.weight)
            # self.update_weights.append(p2.weight)

            if d > self.prototype_distance_threshold:
                # add instead of merging prototypes
                ps1.append(p2)
            else:
                ps1[j].update(p2)

    def prototypes_distance_probabilities(self, cs1, cs2):
        P = np.zeros((len(cs1), len(cs2)))
        # P2 = np.zeros((len(cs1), len(cs2)))

        # assert len(cs1) == len(cs2)

        for j, track2 in enumerate(cs2):
            for i, track1 in enumerate(cs1):
                prob = prob_prototype_represantion_being_same_id_set(self.prototypes[track1], self.prototypes[track2])
                # prob = self.prototypes_match_probability(prototypes[track1], prototypes[track2])

                P[i, j] = prob
                # P2[i, j] = p
                # P2[j, i] = p

        return P

    def desc_clustering_analysis(self):
        from sklearn.cluster import KMeans
        import numpy as np

        Y = []
        X = []
        for y, x in tqdm(self.descriptors.iteritems()):
            Y.append(y)
            X.append(x)

        Y = np.array(Y)

        nbins = 10
        kmeans = KMeans(n_clusters=nbins, random_state=0).fit(X)

        labels = kmeans.labels_


        plt.figure()
        plt.hist(labels, bins=nbins)

        from scipy.spatial.distance import pdist, squareform
        plt.figure()
        plt.imshow(squareform(pdist(kmeans.cluster_centers_)), interpolation='nearest')

        for i in range(nbins):
            xx, yy = 5, 5
            fig, axarr = plt.subplots(xx, yy)
            axarr = axarr.flatten()

            for j, r_id in enumerate(np.random.choice(Y[labels == i], xx*yy)):
                for k in range(6):
                    img = np.random.rand(50, 50, 3)
                    try:
                        img = imread('/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/'+str(k)+'/'+str(r_id)+'.jpg')
                        break
                    except:
                        pass

                axarr[j].imshow(img)
                axarr[j].set_title(str(k))
                axarr[j].axis('off')

            plt.suptitle(str(i))
            plt.show()

        kmeans.cluster_centers_

    def get_track_prototypes(self, tracklet, n=5, debug=False):
        linkages = ['average', 'complete', 'ward']
        linkage = linkages[0]
        connectivity = None

        from sklearn.cluster import AgglomerativeClustering

        # for given GT ID
        # for id_ in range(6):
        X = []
        r_ids = []

        r_ids_arr = tracklet.rid_gen(self.p.gm)

        # r_ids_arr = []
        # import os
        # for r_id in os.listdir('/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/'+str(id_)+'/'):
        #     r_id = int(r_id[:-4])
        #     r_ids_arr.append(r_id)

        for r_id in r_ids_arr:
            if r_id in self.descriptors:
                X.append(self.descriptors[r_id])
                r_ids.append(r_id)
            else:
                logger.debug("descriptor missing for r_id: {}".format(r_id))

        if len(X) == 0:
            logger.warning("missing descriptors for id %d", tracklet.id())

            X = [[0] * 32]

        r_ids = np.array(r_ids)
        X = np.array(X)

        # we need at least 2 samples for aglomerative clustering...
        if X.shape[0] > 1:
            n = min(n, X.shape[0])

            model = AgglomerativeClustering(linkage=linkage,
                                            connectivity=connectivity,
                                            n_clusters=n)
            y = model.fit_predict(X)
        else:
            y = np.array([0])

        prototypes = []
        from track_prototype import TrackPrototype

        # TODO: set this properly!
        std_eps = 1e-6
        for i in range(n):
            ids = y == i
            weight = np.sum(ids)
            if weight:
                desc = np.mean(X[ids, :], axis=0)
                # from sklearn.covariance import LedoitWolf
                # lw = LedoitWolf()
                # lw.fit(X[ids, :])
                # cov = lw.covariance_
                # cov_ = np.cov(X[ids, :].T)

                # this is for case when weight = 1, thus std = 0
                from scipy.spatial.distance import cdist, pdist, squareform
                # TODO: np.mean(cdist([desc], X)**2)**0.5
                d_std = np.mean(cdist([desc], X))
                # std = max(np.mean(np.std(X[ids, :], axis=0)), std_eps)
                prototypes.append(TrackPrototype(desc, d_std, weight))

        if debug:
            # print np.histogram(y, bins=n)

            num_examples = 5
            fig, axarr = plt.subplots(num_examples, n)
            axarr = axarr.flatten()

            for i in range(n):
                for j, r_id in enumerate(np.random.choice(r_ids[y == i], min(num_examples, np.sum(y == i)))):
                    for k in range(6):
                        img = np.random.rand(50, 50, 3)
                        try:
                            img = imread('/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/' + str(k) + '/' + str(
                                r_id) + '.jpg')
                            break
                        except:
                            pass

                    axarr[j * n + i].imshow(img)
                    if j == 0:
                        axarr[j * n + i].set_title(str(np.sum(y == i)))

            for i in range(n*num_examples):
                axarr[i].axis('off')

            plt.suptitle(len(y))
            plt.show()

        return prototypes

    def solve_interactions(self):
        from core.interactions.detect import InteractionDetector
        from core.region.region import Region

        # detector = InteractionDetector('/home/matej/prace/ferda/experiments/171222_0126_batch_36k_random/0.928571428571')
        detector_model_dir = 'data/CNN_models/180222_2253_mobilenet_two_100'
        # TODO: or?
        # detector_model_dir = '../../data/CNN_models/180222_2253_mobilenet_two_100'
        detector = InteractionDetector(detector_model_dir)

        # extract multi tracklets
        multi = [t for t in self.p.chm.tracklet_gen() if t.is_multi()]
        tracklets2 = [t for t in multi if t.get_cardinality(self.p.gm) == 2]

        for t in tqdm(tracklets2, desc='processing 2-interactions'):
            tracks, confidence = t.solve_interaction(detector, self.p.gm, self.p.rm, self.p.img_manager)

            cardinality = 2
            start_frame = t.start_frame(self.p.gm)

            rs = {}
            for id_ in range(cardinality):
                rs[id_] = []

            for i, results in tracks.iterrows():
                for id_ in range(cardinality):
                    r = Region(is_origin_interaction=True, frame=start_frame + i)
                    r.centroid_ = np.array([results["{}_y".format(id_)],
                                            results["{}_x".format(id_)]])
                    r.theta_ = np.deg2rad(results["{}_angle_deg".format(id_)])

                    r.major_axis_ = self.p.stats.major_axis_median
                    r.minor_axis_ = r.major_axis_ / 3

                    rs[id_].append(r)

            # TODO: another threshold...
            conf_threshold = 0.5
            if confidence > conf_threshold:
                used_tracklets = set()

                to_merge = []
                conflict = False

                for id_ in range(cardinality):
                    self.p.rm.add(rs[id_])

                    # for graph manager, when id < 0 means there is no node in graph but it is a direct link to region id*-1
                    rids = [-r.id_ for r in rs[id_]]
                    new_t, _ = self.p.chm.new_chunk(rids, self.p.gm, origin_interaction=True)

                    # Connect...
                    start_r, end_r = self.p.gm.region(new_t.start_vertex_id()), self.p.gm.region(new_t.end_vertex_id())
                    start_frame = start_r.frame()
                    end_frame = end_r.frame()

                    # PRE tracklets
                    pre_tracklets = self.p.chm.tracklets_in_frame(start_frame - 1)
                    # only tracklets which end before interaction are possible options
                    pre_tracklets = filter(lambda x: x.end_frame(self.p.gm) == start_frame - 1 and x.is_single(), pre_tracklets)

                    # TODO: do optimization instead of greedy approach
                    best_start_t = None
                    best_d = np.inf
                    for t in pre_tracklets:
                        t_r = self.p.gm.region(t.end_vertex_id())
                        d = np.linalg.norm(t_r.centroid() - start_r.centroid())

                        if d < best_d:
                            best_d = d
                            best_start_t = t

                    # POST tracklets
                    post_tracklets = self.p.chm.tracklets_in_frame(end_frame + 1)
                    post_tracklets = filter(lambda x: x.start_frame(self.p.gm) == end_frame + 1 and x.is_single(), post_tracklets)

                    best_end_t = None
                    best_d = np.inf
                    for t in post_tracklets:
                        t_r = self.p.gm.region(t.start_vertex_id())
                        d = np.linalg.norm(t_r.centroid() - start_r.centroid())

                        if d < best_d:
                            best_d = d
                            best_end_t = t

                    if best_start_t is not None:
                        if best_start_t not in used_tracklets:
                            self.register_tracklet_as_track(best_start_t)
                            self.register_tracklet_as_track(new_t)

                            to_merge.append((best_start_t, new_t))
                            used_tracklets.add(best_start_t)
                        else:
                            logger.warning("CONFLICT! Race condition during interaction solver best_start")
                            logger.debug("tbest_start_t: {}, t_interaction_origined: {}, best_end_t: {}".format(
                                best_start_t, new_t, best_end_t))

                            conflict = True

                    if best_end_t is not None:
                        if best_end_t not in used_tracklets:
                            self.register_tracklet_as_track(new_t)
                            self.register_tracklet_as_track(best_end_t)
                            to_merge.append((new_t, best_end_t))
                            used_tracklets.add(best_end_t)
                        else:
                            logger.warning("CONFLICT! Race condition during interaction solver best_end")
                            logger.debug("tbest_start_t: {}, t_interaction_origined: {}, best_end_t: {}".format(
                                best_start_t, new_t, best_end_t))

                            conflict = True

                if not conflict:
                    for t1, t2 in to_merge:
                        logger.debug("merging: {} -> {}".format(t1, t2))
                        if t1 != t2:
                            self.merge_tracklets(t1, t2)

        p.save()


def _get_ids_from_folder(wd, n):
    # .DS_Store...
    files = list(filter(lambda x: x[0] != '.', os.listdir(wd)))
    rids = random.sample(files, n)

    rids = map(lambda x: x[:-4], rids)
    return np.array(map(int, rids))


def _get_distances(ids1, ids2, descriptors):
    x = []
    for i, j in zip(ids1, ids2):
        if i not in descriptors or j not in descriptors:
            continue

        x.append(np.linalg.norm(np.array(descriptors[i]) - np.array(descriptors[j])))

    return x


def test_descriptors_distance(descriptors, n=2000):
    WD = '/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/'
    pos_distances = []
    neg_distances = []

    NUM_ANIMALS = 6

    dist_m = np.zeros((NUM_ANIMALS, NUM_ANIMALS))

    for id_ in range(NUM_ANIMALS):
        rids1 = _get_ids_from_folder(WD+str(id_), n)
        rids2 = _get_ids_from_folder(WD+str(id_), n)

        ds = _get_distances(rids1, rids2, descriptors)
        dist_m[id_, id_] = np.mean(ds)
        pos_distances.extend(ds)
        for opponent_id in range(NUM_ANIMALS):
            if id_ == opponent_id:
                continue

            rids3 = _get_ids_from_folder(WD+str(opponent_id), n/NUM_ANIMALS)
            ds = _get_distances(rids1, rids3, descriptors)
            neg_distances.extend(ds)

            dist_m[id_, opponent_id] = np.mean(ds)

    plt.figure()
    # np.set_printoptions(precision=2)
    # print dist_m
    plt.imshow(dist_m, interpolation='nearest')

    bins = 200
    print len(pos_distances)
    print len(neg_distances)
    plt.figure()
    print np.histogram(pos_distances, bins=bins, density=True)
    positive = plt.hist(pos_distances, bins=bins, alpha=0.6, color='g', density=True, label='positive')
    plt.hold(True)
    negative = plt.hist(neg_distances, bins=bins, alpha=0.6, color='r', density=True, label='negative')

    x = np.linspace(0., 3., 100)
    print("lambda: {:.3f}".format(1./np.mean(pos_distances)))
    for lam in [1./np.mean(pos_distances)]:
        y = lam * np.exp(-lam * x)
        pdf, = plt.plot(x, y)
        y = np.exp(-lam * x)
        prob, = plt.plot(x, y)

    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='negative ')
    green_patch = mpatches.Patch(color='green', label='positive')
    # plt.legend([positive, negative, pdf, cdf], ['same', 'different', 'PDF: lambda*e**(-lambda * x), lambda={:.2f}'.format(lam), 'CDF'])
    plt.legend([green_patch, red_patch, pdf, prob], ['same', 'different', 'PDF: \lambda * e^(-\lambda * x), \lambda={:.2f}'.format(lam), 'Probability'])
    # plt.legend([positive, negative], ['same', 'different'])
    plt.xlabel('distance')

    # plt.figure()
    # for lam in [1./np.mean(pos_distances)]:
    #     y = lam * np.exp(-lam * x)
    #     plt.plot(x, y)

    plt.show()



def prob_prototype_represantion_being_same_id_set(prot1, prot2):
    p1 = prototypes_distribution_probability(prot1, prot2)
    p2 = prototypes_distribution_probability(prot2, prot1)

    p = (p1 + p2) / 2.
    return p


def prototypes_distribution_probability(prot1, prot2):
    from scipy.stats import norm

    W1 = float(sum([p1.weight for p1 in prot1]))
    # with sum over mixtures, there is problem, that sum might be > 1...
    # with matching - there should be problem when # prototypes differs len(prot1) != len(prot2)
    p_to_prot2 = 0
    for p1 in prot1:
        best_p = 0
        for p2 in prot2:
            n = norm(0, p1.std)
            # n = multivariate_normal(p1.descriptor, p1.cov, allow_singular=True)
            # p = (p1.weight/W1) * n.pdf(np.linalg.norm(p2.descriptor-p1.descriptor)) / n.pdf(0)
            p = (p1.weight / W1) * 2 * n.cdf(-np.linalg.norm(p2.descriptor - p1.descriptor))

            if p > best_p:
                best_p = p

        p_to_prot2 += best_p

    return p_to_prot2


def do_complete_set_matching(project):
    logger.info('do_complete_set_matching start')
    from core.id_detection.learning_process import LearningProcess
    lp = LearningProcess(project)

    lp._reset_chunk_PN_sets()
    # reset id_decision_info
    for t in project.chm.tracklet_gen():
        # try:
        t.id_decision_info = ''
        # except:
        #     pass

    descriptors_path = os.path.join(project.working_directory, 'descriptors.pkl')
    csm = CompleteSetMatching(project, lp, descriptors_path, quality_threshold=0.2, quality_threshold2=0.01)
    # csm.solve_interactions()
    csm.start_matching_process()
    logger.info('do_complete_set_matching finished')



if __name__ == '__main__':

    # P_WD = '/Users/flipajs/Documents/wd/FERDA/Cam1'
    # P_WD = '/Users/flipajs/Documents/wd/FERDA/april-paper/Cam1_clip_arena_fixed'
    # P_WD = '/Users/flipajs/Documents/wd/FERDA/april-paper/Sowbug3-crop'
    # path = '/Users/flipajs/Documents/wd/FERDA/april-paper/Sowbug3-fixed-segmentation'
    # P_WD = '/Users/flipajs/Documents/wd/FERDA/april-paper/5Zebrafish_nocover_22min'
    # P_WD = '/Users/flipajs/Documents/wd/FERDA/april-paper/Camera3-5min'
    # p.load('/Use/rs/flipajs/Documents/wd/FERDA/Camera3_new')
    path = '../projects/Sowbug_deleteme2'
    from core.project.project import Project
    p = Project(path)
    do_complete_set_matching(p)


