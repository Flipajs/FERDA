import numpy as np
from utils.video_manager import get_auto_video_manager
from tqdm import tqdm
from lazyme.string import color_print
import matplotlib.pyplot as plt
from scipy.misc import imread
import os, random


class CompleteSetMatching:
    def __init__(self, project, lp, descriptors, quality_threshold=0.02):
        self.prototype_distance_threshold = np.inf # ignore
        self.QUALITY_THRESHOLD = quality_threshold
        self.p = project
        self.lp = lp
        self.get_probs = self.lp._get_tracklet_proba
        self.get_p1s = self.lp.get_tracklet_p1s
        self.descriptors = descriptors

        self.update_distances = []
        self.update_weights = []

    def process(self):
        CSs = self.find_cs()

        #####
        # for t in CSs[0]:
        #     self.get_track_prototypes(t, n=10)

        id_ = 0

        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        from matplotlib.widgets import Cursor

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cur = Cursor(ax, horizOn=False)
        # plt.hold(True)

        id_, prototypes, qualities, track_CSs, tracklets_2_tracks, tracks = self.sequential_matching(CSs, id_)

        print("track CSs")
        for CS in track_CSs:
            print(sorted(CS))

        print

        ##### now do CS of tracks to tracks matching
        self.tracks_CS_matching(track_CSs, prototypes, tracklets_2_tracks, tracks)

        ##### then do the rest... bruteforce approach
        # 1. find best Track CS
        # 2. for each tracklet, try to find best track, computed probability, sort by probs...

        best_CS = None
        best_support = 0
        for CS in track_CSs:
            val = self.track_support(CS, tracks)
            if val > best_support:
                best_CS = CS
                best_support = val

        print "BEGINNING of best_set matching"

        # update N sets for unassigned tracklets in relations to best_CS track ids
        for tracklet, track_id in tracklets_2_tracks.iteritems():
            self.add_to_N_set(track_id, tracklet)

        # go through tracklets, find biggest set and do matching..
        # 1) choose tracklet
        #       longest?
        #       for now, process as it is in chunk_generator

        num_undecided = 0
        for t in self.p.chm.tracklet_gen():
            # TODO: what about matching unmatched Tracks as well?
            if t in tracklets_2_tracks or not t.is_single():
                continue

            # 2) find biggest set
            best_set = self.find_biggest_undecided_tracklet_set(t)

            prohibited_ids = {}
            for t_ in best_set:
                if t_ not in prototypes:
                    prototypes[id_] = self.get_track_prototypes(t_)
                    tracklets_2_tracks[t_] = id_
                    id_ += 1

                    prohibited_ids[t_] = []
                    for test_t in self.p.chm.tracklets_intersecting_t_gen(t_, self.p.gm):
                        if test_t == t_:
                            continue

                        if len(test_t.P):
                            prohibited_ids[t_].append(list(test_t.P)[0])

            # 3) compute matching
            P_a = self.prototypes_distance_probabilities(best_set, best_CS, prototypes, tracklets_2_tracks)

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

            np.set_printoptions(precision=3)
            print P
            # print best_set, best_CS
            print perm, quality

            # 4) accept?
            if quality[1] > self.QUALITY_THRESHOLD:
                for (tracklet, track_id) in perm:
                    print "[{} |{}| (te: {})] -> {}".format(tracklet.id(), len(tracklet), tracklet.end_frame(self.p.gm), track_id)
                    tracklets_track = tracklets_2_tracks[tracklet]
                    self.update_prototypes(prototypes[track_id], prototypes[tracklets_track])
                    del prototypes[tracklets_track]

                    tracklets_2_tracks[tracklet] = track_id
                    tracks[track_id].append(tracklet)

                    tracklet.P = set([track_id])
                    tracklet.N = set(range(len(self.p.animals))) - tracklet.P
                    tracklet.id_decision_info = 'best_set_matching'

                    # propagate
                    # TODO: add sanity checks?
                    for t_ in self.p.chm.singleid_tracklets_intersecting_t_gen(tracklet, self.p.gm):
                        if t_ != tracklet:
                            self.add_to_N_set(track_id, t_)

            else:
                color_print('QUALITY BELOW', color='red')
                num_undecided += 1

        print "#UNDECIDED: {}".format(num_undecided)
        print
        # self.single_track_assignment(best_CS, prototypes, tracklets_2_tracks)

        #### visualize and stats
        from utils.rand_cmap import rand_cmap

        new_cmap = rand_cmap(id_+1, type='bright', first_color_black=True, last_color_black=False)
        print "#IDs: {}".format(id_+1)
        support = {}
        tracks = {}
        tracks_mean_desc = {}
        for t in self.p.chm.chunk_gen():
            if len(t.P):
                t_identity = list(t.P)[0]
                support[t_identity] = support.get(t_identity, 0) + len(t)
                if t_identity not in tracks:
                    tracks[t_identity] = []

                tracks[t_identity].append(t.id())

                t_desc_w = self.get_mean_descriptor(t) * len(t)
                if t_identity not in tracks_mean_desc:
                    tracks_mean_desc[t_identity] = t_desc_w
                else:
                    tracks_mean_desc[t_identity] += t_desc_w

                plt.scatter(t.start_frame(self.p.gm), t_identity, c=new_cmap[t_identity], edgecolor=[0.,0.,0.,.3])
                plt.plot([t.start_frame(self.p.gm), t.end_frame(self.p.gm)+0.1], [t_identity, t_identity],
                         c=new_cmap[t_identity],
                         path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
            else:
                if t.is_noise() or len(t) < 5:
                    continue
                if t.is_single():
                    c = [0, 1, 0, .3]
                else:
                    c = [0, 0, 1, .3]

                y = t.id() % id_
                plt.scatter(t.start_frame(self.p.gm), y, c=c, marker='s', edgecolor=[0., 0., 0., .1])
                plt.plot([t.start_frame(self.p.gm), t.end_frame(self.p.gm) + 0.1], [y, y],
                         c=c,
                         linestyle='-')


        plt.grid()

        print "SUPPORT"
        for id in sorted(support.keys()):
            print "{}: {}, #{} ({})".format(id, support[id], len(tracks[id]), tracks[id])

        map_ = {}
        for id in range(6):
            for t in self.p.chm.chunk_gen():
                if id in t.P:
                    t.P = set([id+100])
                    map_[id] = id+100

        for new_id, id in enumerate(sorted(support.keys())[:6]):
            for t in p.chm.chunk_gen():
                if id in map_:
                    id = map_[id]

                if id in t.P:
                    t.P = set([new_id])

        # print "ISOLATED CS GROUPS: {}".format(len(isolated_cs_groups))
        # for cs in isolated_cs_groups:
        #     total_len = 0
        #     s = ""
        #     for t in cs:
        #         s += ", "+str(t.id())
        #         total_len += len(t)
        #
        #     print "total len: {}, IDs: {}".format(total_len, s)
        #     print

        p.save()
        qualities = np.array(qualities)
        plt.figure()
        plt.plot(qualities[:, 0])
        plt.grid()
        plt.figure()
        plt.plot(qualities[:, 1])
        plt.grid()

        plt.figure()
        plt.show()

        # mean_ds = []
        # for id_, mean in tracks_mean_desc.iteritems():
        #     mean_ds.append(mean/float(support[id]))

        print("track ids order: {}\n{}".format(list(tracks_mean_desc.iterkeys()), len(tracks)))
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

                print probs[id_], tracklets[id_]

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

    def sequential_matching(self, CSs, id_):
        print "BEGINNING of SEQUENTIAL MATCHING"
        tracks = {}
        tracklets_2_tracks = {}
        track_CSs = [[]]
        prototypes = {}
        for i, t in enumerate(CSs[0]):
            tracks[i] = [t]
            tracklets_2_tracks[t] = i
            prototypes[i] = self.get_track_prototypes(t)
            t.P = set([id_])
            t.id_decision_info = 'sequential_matching'
            track_CSs[-1].append(id_)
            id_ += 1
        qualities = []
        for i in range(len(CSs) - 1):
            print "CS {}, CS {}".format(i, i + 1)

            # first create new virtual tracks and their prototypes for CSs[i+1] which are not already in tracks
            for t in CSs[i + 1]:
                if t in tracklets_2_tracks:
                    continue

                new_track_id = max(tracks.keys()) + 1
                tracks[new_track_id] = [t]
                tracklets_2_tracks[t] = new_track_id
                prototypes[new_track_id] = self.get_track_prototypes(t)

            # perm, quality = self.cs2cs_matching_descriptors_and_spatial(CSs[i], CSs[i+1])
            perm, quality = self.cs2cs_matching_prototypes_and_spatial(CSs[i], CSs[i + 1], prototypes,
                                                                       tracklets_2_tracks)

            cs1_max_frame = 0
            cs2_min_frame = np.inf
            dividing_frame = 0
            for (t1, t2) in perm:
                if t1 == t2:
                    break

                cs1_max_frame = max(cs1_max_frame, t1.start_frame(self.p.gm))
                cs2_min_frame = min(cs2_min_frame, t2.end_frame(self.p.gm))

                dividing_frame = max(dividing_frame, t2.start_frame(self.p.gm))

            print "cs1 max frame: {}, cs2 min frame: {}".format(cs1_max_frame, cs2_min_frame)

            # TODO: threshold 1-

            not_same = 0
            c = [0. + 1 - quality[1], quality[1], 0., 0.2]
            # propagate IDS if quality is good enough:
            if quality[1] > self.QUALITY_THRESHOLD:
                # TODO: transitivity? when t1 -> t2 assignment uncertain, look on ID probs for t2->t3 and validate wtih t1->t3

                for (t1, t2) in perm:
                    print "[{} |{}| (te: {})] -> {} |{}| (ts: {})".format(t1.id(), len(t1), t1.end_frame(self.p.gm),
                                                                          t2.id(), len(t2), t2.start_frame(self.p.gm))

                    # if merge...
                    if t1 != t2:
                        track1 = list(t1.P)[0]
                        track2 = tracklets_2_tracks[t2]
                        self.update_prototypes(prototypes[track1], prototypes[track2])
                        del tracks[track2]
                        del prototypes[track2]

                        tracklets_2_tracks[t2] = track1
                        tracks[track1].append(t2)

                        t2.P = set(t1.P)
                        t2.N = set(t2.N)
            else:
                color_print('QUALITY BELOW', color='red')
                # c = [1., 0.,0.,0.7]

                track_CSs.append([])
                for pair in perm:
                    t = pair[1]
                    if len(t.P) == 0:
                        t.P = set([id_])
                        id_ += 1

                    track_CSs[-1].append(list(t.P)[0])

                for pair in perm:
                    if pair[0] != pair[1]:
                        not_same += 1

            plt.plot([dividing_frame, dividing_frame], [-5, -5 + 4.7 * quality[1]], c=c)
            plt.plot([dividing_frame, dividing_frame], [0, id_ - 1 + not_same], c=c)

            print quality
            print

            qualities.append(quality)
        print
        tracks_unassigned_len = 0
        tracks_unassigned_num = 0
        for t in self.p.chm.chunk_gen():
            if t.is_single() and t not in tracklets_2_tracks:
                tracks_unassigned_len += len(t)
                tracks_unassigned_num += 1
        num_prototypes = 0
        for prots in prototypes.itervalues():
            num_prototypes += len(prots)
        print("seqeuntial CS matching done...")
        print("#tracks: {}, #tracklets2tracks: {}, unassigned #{} len: {}, #prototypes: {}".format(
            len(tracks), len(tracklets_2_tracks), tracks_unassigned_num, tracks_unassigned_len, num_prototypes))
        return id_, prototypes, qualities, track_CSs, tracklets_2_tracks, tracks

    def add_to_N_set(self, track_id, tracklet):
        for t in self.p.chm.chunks_in_interval(tracklet.start_frame(self.p.gm), tracklet.end_frame(self.p.gm)):
            if t.is_single() and t != tracklet:
                t.N.add(track_id)

    def tracks_CS_matching(self, track_CSs, prototypes, tracklets2tracks, tracks):
        # 1. get CS of tracks (we already have them in track_CSs from sequential process.
        # 2. sort CS by sum of track lengths
        # 3. try to match all others to this one (spatio-temporal term might be switched on for close tracks?)
        # 4. if any match accepted update and goto 2.
        # 5. else take second biggest and goto 3.
        # 6. end if only one CS, or # of CS didn't changed...

        print "BEGINNING of GLOBAL MATCHING"
        updated = True
        while len(track_CSs) > 1 and updated:
            updated = False

            # 2. sort CS by sum of track lengths
            ordered_CSs = self.sort_track_CSs(track_CSs, tracks)

            # 3.
            for i in range(len(ordered_CSs)-1):
                pivot = ordered_CSs[i]

                best_quality = 0
                best_perm = None
                best_CS = None

                for CS in ordered_CSs[i+1:]:
                    perm, quality = self.cs2cs_matching_prototypes_and_spatial(pivot, CS, prototypes, tracklets2tracks,
                                                                               use_spatial_probabilities=False)

                    print CS, quality

                    if quality[1] > best_quality:
                        best_quality = quality[1]
                        best_perm = perm
                        best_CS = CS

                if best_quality > self.QUALITY_THRESHOLD:
                    print("Best track CS match accepted. {}, {}".format(best_perm, best_quality))
                    self.merge_track_CSs(best_perm, prototypes, tracklets2tracks, tracks)
                    track_CSs.remove(best_CS)
                    self.update_all_track_CSs(best_perm, track_CSs)
                    updated = True
                    break
                else:
                    print("Best track CS match rejected. {}, {}".format(perm, quality))

    def update_all_track_CSs(self, perm, track_CSs):
        # update all CS
        for CS_for_update in track_CSs:
            size_before = len(CS_for_update)
            for track_id1, track_id2 in perm:
                for i, track_id in enumerate(CS_for_update):
                    if track_id == track_id2:
                        CS_for_update[i] = track_id1

            # TODO: this means conflict...
            if len(set(CS_for_update)) != size_before:
                print "CONFLICT ", CS_for_update
            # assert len(set(CS_for_update)) == size_before

    def merge_track_CSs(self, perm, prototypes, tracklets2tracks, tracks):
        # keep attention, here we have tracks, not tracklets...
        for (track1_id, track2_id) in perm:
            print "{} -> {}".format(track1_id, track2_id)

            # if merge...
            if track1_id != track2_id:
                self.update_prototypes(prototypes[track1_id], prototypes[track2_id])
                for tracklet in tracks[track2_id]:
                    tracklets2tracks[tracklet] = track1_id
                    tracks[track1_id].append(tracklet)
                    tracklet.P = set([track1_id])
                    tracklet.id_decision_info = 'global_matching'

                del tracks[track2_id]
                del prototypes[track2_id]

    def sort_track_CSs(self, track_CSs, tracks):
        values = []
        for CS in track_CSs:
            val = self.track_support(CS, tracks)

            values.append(val)

        values_i = reversed(sorted(range(len(values)), key=values.__getitem__))
        CS_sorted = []
        for i in values_i:
            print track_CSs[i], values[i]
            CS_sorted.append(track_CSs[i])

        return CS_sorted

    def track_support(self, CS, tracks):
        val = 0
        for track_id in CS:
            for tracklet in tracks[track_id]:
                val += len(tracklet)
        return val

    def find_cs(self):
        unique_tracklets = set()
        CSs = []
        vm = get_auto_video_manager(self.p)
        total_frame_count = vm.total_frame_count()

        frame = 0
        i = 0
        old_frame = 0
        print "analysing project, searching Complete Sets"
        print
        with tqdm(total=total_frame_count) as pbar:
            while True:
                group = self.p.chm.chunks_in_frame(frame)
                if len(group) == 0:
                    break

                singles_group = filter(lambda x: x.is_single(), group)

                if len(singles_group) == len(self.p.animals) and min([len(t) for t in singles_group]) >= 1:
                    CSs.append(singles_group)

                    for t in singles_group:
                        unique_tracklets.add(t)

                    frame = min([t.end_frame(self.p.gm) for t in singles_group]) + 1
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

    def cs2cs_matching_prototypes_and_spatial(self, cs1, cs2, prototypes, tracklets_2_tracks, use_spatial_probabilities=True):
        perm = []
        cs1, cs2, cs_shared = self.remove_straightforward_tracklets(cs1, cs2)
        # assert len(cs1) == len(cs2)

        if len(cs1) == 1:
            perm.append((cs1[0], cs2[0]))
            quality = [1.0, 1.0]
        elif len(cs1) == 0:
            quality = [1.0, 1.0]
        else:
            assert len(cs2) > 1
            P_a = self.prototypes_distance_probabilities(cs1, cs2, prototypes, tracklets_2_tracks)

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



    def spatial_probabilities(self, cs1, cs2, lower_bound=0.5):
        # should be neutral if temporal distance is too big
        # should be restrictive when spatial distance is big
        max_d = self.p.solver_parameters.max_edge_distance_in_ant_length * self.p.stats.major_axis_median
        P = np.zeros((len(cs1), len(cs2)), dtype=np.float)

        for i, t1 in enumerate(cs1):
            t1_ef = t1.end_frame(self.p.gm)
            for j, t2 in enumerate(cs2):
                if t1 == t2:
                    prob = 1.0
                else:
                    temporal_d = t2.start_frame(self.p.gm) - t1_ef

                    if temporal_d < 0:
                        prob = -np.inf
                    else:
                        t1_end_r = self.p.gm.region(t1.end_node())
                        t2_start_r = self.p.gm.region(t2.start_node())
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

    def prototypes_distance_probabilities(self, cs1, cs2, prototypes, tracklets2tracks):
        P = np.zeros((len(cs1), len(cs2)))
        # P2 = np.zeros((len(cs1), len(cs2)))

        # assert len(cs1) == len(cs2)

        for j, t2 in enumerate(cs2):
            for i, t1 in enumerate(cs1):
                # it is already track...
                if isinstance(t1, int):
                    track1 = t1
                else:
                    track1 = tracklets2tracks[t1]

                if isinstance(t2, int):
                    track2 = t2
                else:
                    track2 = tracklets2tracks[t2]

                prob = prob_prototype_represantion_being_same_id_set(prototypes[track1], prototypes[track2])
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
                print r_id

        if len(X) == 0:
            import warnings
            warnings.warn("missing descriptors for id {}".format(tracklet.id()))

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
                import scipy
                from scipy.spatial.distance import cdist, pdist, squareform
                d_std = np.mean(cdist([desc], X))
                # std = max(np.mean(np.std(X[ids, :], axis=0)), std_eps)
                prototypes.append(TrackPrototype(desc, d_std, weight))

        if debug:
            print np.histogram(y, bins=n)

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
    np.set_printoptions(precision=2)
    print dist_m
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


if __name__ == '__main__':
    from core.project.project import Project
    from core.id_detection.learning_process import LearningProcess

    p = Project()
    p.load('/Users/flipajs/Documents/wd/FERDA/Cam1')
    # p.load('/Users/flipajs/Documents/wd/FERDA/Camera3_new')

    lp = LearningProcess(p)
    lp.reset_learning()

    # reset id_decision_info
    for t in p.chm.tracklet_gen():
        try:
            t.id_decision_info = ''
        except:
            pass

    import pickle
    with open('/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/descriptors.pkl') as f:
        descriptors = pickle.load(f)

    from numpy.linalg import norm

    # test_descriptors_distance(descriptors)
    # np.random.seed(13)
    np.random.seed(42)

    csm = CompleteSetMatching(p, lp, descriptors)
    # r1 = csm.get_track_prototypes(p.chm[4188])
    # r2 = csm.get_track_prototypes(p.chm[2475])
    # r3 = csm.get_track_prototypes(p.chm[3856])
    # b1 = csm.get_track_prototypes(p.chm[2715])
    # b2 = csm.get_track_prototypes(p.chm[2137])
    #
    # prob_prototype_represantion_being_same_id_set(r1, r1)
    # prob_prototype_represantion_being_same_id_set(r1, r2)
    # prob_prototype_represantion_being_same_id_set(r1, r3)
    # prob_prototype_represantion_being_same_id_set(r2, r3)
    # prob_prototype_represantion_being_same_id_set(r1, b1)
    # prob_prototype_represantion_being_same_id_set(r1, b2)
    # prob_prototype_represantion_being_same_id_set(b1, b2)

    # csm.desc_clustering_analysis()
    csm.process()
