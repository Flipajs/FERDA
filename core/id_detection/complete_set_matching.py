import numpy as np
from utils.video_manager import get_auto_video_manager
from tqdm import tqdm
from lazyme.string import color_print
import matplotlib.pyplot as plt
from scipy.misc import imread
import os, random


class CompleteSetMatching:
    def __init__(self, project, get_tracklet_probs_callback, get_tracklet_p1s_callback, descriptors, quality_threshold=0.02):
        self.prototype_distance_threshold = np.inf # ignore
        self.QUALITY_THRESHOLD = quality_threshold
        self.p = project
        self.get_probs = get_tracklet_probs_callback
        self.get_p1s = get_tracklet_p1s_callback
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

        tracks = {}
        tracklets_2_tracks = {}
        track_CSs = [[]]
        prototypes = {}

        for i, t in enumerate(CSs[0]):
            tracks[i] = [t]
            tracklets_2_tracks[t] = i
            prototypes[i] = self.get_track_prototypes(t)
            t.P = set([id_])
            track_CSs[-1].append(id_)
            id_ += 1

        qualities = []
        for i in range(len(CSs)-1):
            print "CS {}, CS {}".format(i, i+1)

            # first create new virtual tracks and their prototypes for CSs[i+1] which are not already in tracks
            for t in CSs[i+1]:
                if t in tracklets_2_tracks:
                    continue

                new_track_id = max(tracks.keys()) + 1
                tracks[new_track_id] = [t]
                tracklets_2_tracks[t] = new_track_id
                prototypes[new_track_id] = self.get_track_prototypes(t)

            # perm, quality = self.cs2cs_matching_descriptors_and_spatial(CSs[i], CSs[i+1])
            perm, quality = self.cs2cs_matching_prototypes_and_spatial(CSs[i], CSs[i+1], prototypes, tracklets_2_tracks)

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
            c = [0. + 1-quality[1], quality[1],0., 0.2]
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

            plt.plot([dividing_frame, dividing_frame], [-5, -5 + 4.7*quality[1]], c=c)
            plt.plot([dividing_frame, dividing_frame], [0, id_-1 + not_same], c=c)

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
            if t in tracklets_2_tracks or not t.is_single():
                continue

            if t not in tracklets_prototypes:
                tracklets_prototypes[t.id()] = self.get_track_prototypes(t)


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

            probabilities[t] = best_p
            decisioins[t] = best_track

            probs.append(best_p)
            best_track_ids.append(best_track)
            lengths.append(len(t))
            tracklets.append(t)

        probs2 = np.array(probs2)
        probs = np.array(probs)

        probs = probs2

        tracklets = np.array(tracklets)
        ids = np.argsort(-probs)
        best_track_ids = np.array(best_track_ids)

        import warnings
        for i in ids:
            if probs[i] > 0.5:
                t = tracklets[i]
                track_id = best_track_ids[i]
                if track_id in t.N:
                    warnings.warn("IN N ... warning {}".format(t.id()))

                print probs[i], tracklets[i]
                t.P = set([track_id])

                self.add_to_N_set(track_id, t)

                # TODO: propagate...

        plt.figure()

        plt.scatter(np.arange(len(probs)), probs, c='r')
        plt.scatter(np.arange(len(probs)), probs2, c='g')

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

        mean_ds = []
        for id_, mean in tracks_mean_desc.iteritems():
            mean_ds.append(mean/float(support[id]))

        print("track ids order: {}\n{}".format(list(tracks_mean_desc.iterkeys()), len(tracks)))
        from scipy.spatial.distance import pdist, squareform
        plt.imshow(squareform(pdist(mean_ds)), interpolation='nearest')
        plt.show()

        for i in range(50, 60):
            print "CS {}, CS {}".format(0, i)
            perm, quality = self.cs2cs_matching_ids_unknown(CSs[0], CSs[i])
            for (t1, t2) in perm:
                print t1.id(), " -> ", t2.id()

            print quality

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

        print "trying matching on track CS"
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
                    self.update_all_track_CSs(best_perm, track_CSs)
                    track_CSs.remove(best_CS)
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
        assert len(cs1) == len(cs2)

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
        P2 = np.zeros((len(cs1), len(cs2)))

        assert len(cs1) == len(cs2)

        for j, t2 in enumerate(cs2):
            for i, t1 in enumerate(cs1):
                # it is already track...
                if isinstance(t1, int) and isinstance(t2, int):
                    track1, track2 = t1, t2
                else:
                    track1 = tracklets2tracks[t1]
                    track2 = tracklets2tracks[t2]

                prob = self.prototypes_match_probability(prototypes[track1], prototypes[track2])

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
        for i in range(n):
            ids = y == i
            weight = np.sum(ids)
            if weight:
                desc = np.mean(X[ids, :], axis=0)
                prototypes.append(TrackPrototype(desc, weight))

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
    rids = random.sample(os.listdir(wd), n)

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

    for id_ in range(NUM_ANIMALS):
        rids1 = _get_ids_from_folder(WD+str(id_), n)
        rids2 = _get_ids_from_folder(WD+str(id_), n)

        pos_distances.extend(_get_distances(rids1, rids2, descriptors))
        for opponent_id in range(NUM_ANIMALS):
            if id_ == opponent_id:
                continue

            rids3 = _get_ids_from_folder(WD+str(opponent_id), n/NUM_ANIMALS)
            neg_distances.extend(_get_distances(rids1, rids3, descriptors))

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






if __name__ == '__main__':
    from core.project.project import Project
    from core.id_detection.learning_process import LearningProcess

    p = Project()
    p.load('/Users/flipajs/Documents/wd/FERDA/Cam1')
    # p.load('/Users/flipajs/Documents/wd/FERDA/Camera3_new')

    lp = LearningProcess(p)
    lp.reset_learning()

    import pickle
    with open('/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/descriptors.pkl') as f:
        descriptors = pickle.load(f)


    desc_3599 = []
    for r_id in p.chm[3599].rid_gen(p.gm):
        try:
            desc_3599.append(descriptors[r_id])
        except:
            print r_id

    desc_2670 = []
    for r_id in p.chm[2670].rid_gen(p.gm):
        try:
            desc_2670.append(descriptors[r_id])
        except:
            print r_id

    desc_1710 = []
    for r_id in p.chm[1710].rid_gen(p.gm):
        try:
            desc_1710.append(descriptors[r_id])
        except:
            print r_id

    desc_1951 = []
    for r_id in p.chm[1951].rid_gen(p.gm):
        try:
            desc_1951.append(descriptors[r_id])
        except:
            print r_id

    desc_4035 = []
    for r_id in p.chm[4035].rid_gen(p.gm):
        try:
            desc_4035.append(descriptors[r_id])
        except:
            print r_id

    desc_1727 = []
    for r_id in p.chm[1727].rid_gen(p.gm):
        try:
            desc_1727.append(descriptors[r_id])
        except:
            print r_id


    from numpy.linalg import norm

    # test_descriptors_distance(descriptors)
    # np.random.seed(13)
    np.random.seed(42)

    csm = CompleteSetMatching(p, lp._get_tracklet_proba, lp.get_tracklet_p1s, descriptors)

    # protos = []
    # for i in range(10):
    #     protos.append(csm.get_track_prototypes(p.chm[3599], n=2))
    #
    # for i in range(10):
    #     print csm.prototypes_match_probability(protos[0], protos[i])
    #
    # csm.prototypes_match_probability(protos[0], protos[1])
    # proto3599 = csm.get_track_prototypes(p.chm[3599])
    # proto2670 = csm.get_track_prototypes(p.chm[2670])
    # proto1710 = csm.get_track_prototypes(p.chm[1710])
    # proto1951 = csm.get_track_prototypes(p.chm[1951])
    #
    # print norm(np.mean(desc_3599) - np.mean(desc_2670)), csm.prototypes_match_probability(proto3599, proto2670)
    # print norm(np.mean(desc_3599) - np.mean(desc_1710)), csm.prototypes_match_probability(proto3599, proto1710)
    # print norm(np.mean(desc_1951) - np.mean(desc_2670)), csm.prototypes_match_probability(proto1951, proto2670)
    # print norm(np.mean(desc_1951) - np.mean(desc_1710)), csm.prototypes_match_probability(proto1951, proto1710)

    # csm.desc_clustering_analysis()
    csm.process()
