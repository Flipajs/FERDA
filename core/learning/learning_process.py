import cPickle as pickle
import operator
import sys
import time
import warnings

import numpy as np
from PyQt4 import QtGui
from sklearn.ensemble import RandomForestClassifier

from core.graph.region_chunk import RegionChunk
from core.project.project import Project
from features import get_features_var3, get_features_var5
from gui.learning.ids_names_widget import IdsNamesWidget
from utils.img_manager import ImgManager

import itertools
import math
from utils.img import rotate_img, centered_crop, get_bounding_box, endpoint_rot
from skimage.feature import local_binary_pattern

class LearningProcess:
    """
    each tracklet has 2 id sets.
    P - ids are present for sure
    N - ids are not present for sure

    A - set of all animal ids.
    When P.union(N) == A, tracklet is decided. When len(P) == 1, it is a tracklet with one id.

    When len(P.intersection(N)) > 0 it is a CONFLICT

    I(t) - set of all tracklets having intersection with tracklet t

    t ~ u means has the same id set
    knowledge[t.id()] - list of all additional information obtained from user tracklet t e.g.
    - t is decided (P, N is known)
    - t ~ u (has the same id set)
    - t.P has id
    - t.N has id (same as t.P has not id)
    - t !~ u (has the different id set to u)

    basic operations:
        update_P
        update_N
        before each basic operation check knowledge[t.id()]
    """
    def __init__(self, p, use_feature_cache=False, use_rf_cache=False, question_callback=None, update_callback=None, ghost=False):
        if use_rf_cache:
            warnings.warn("use_rf_cache is Deprecated!", DeprecationWarning)

        self.p = p

        self.min_new_samples_to_retrain = 10000

        # TODO: add whole knowledge class...
        self.tracklet_knowledge = {}

        self.question_callback = question_callback
        self.update_callback = update_callback

        self._eps_certainty = 0.3

        # TODO: make standalone feature extractor...
        self.get_features = get_features_var5

        # to solve uncertainty about head orientation... Add both
        self.features_fliplr_hack = True

        # TODO: global parameter!!!
        self.k_ = 50.0

        self.X = []
        self.y = []
        self.old_x_size = 0

        self.collision_chunks = {}

        self.p.img_manager = ImgManager(self.p, max_num_of_instances=700)

        self.all_ids = set(range(len(self.p.animals)))

        self.undecided_tracklets = set()
        self.tracklet_certainty = {}
        self.tracklet_measurements = {}

        self.user_decisions = []

        self.mistakes = []

        self.features = {}

        self.load_learning()

        # when creating without feature data... e.g. in main_tab_widget
        if ghost:
            return

        if not use_feature_cache:
            # TODO: do better... Idealy chunks should already have labels
            self.get_candidate_chunks()

            self.features = self.precompute_features_()

            with open(p.working_directory+'/features.pkl', 'wb') as f:
                d = {'features': self.features,
                     'collision_chunks': self.collision_chunks}
                # pickle.dump(d, f, -1)
                # withou -1, compression, faster?
                pickle.dump(d, f)
        else:
            print "LOADING features..."

            with open(p.working_directory+'/features.pkl', 'rb') as f:
                d = pickle.load(f)
                self.features = d['features']
                self.collision_chunks = d['collision_chunks']

            print "LOADED"

        print "precompute avalability"
        self.__reset_chunk_PN_sets()

        # TODO: remove this
        self.class_frequences = []

        print "undecided tracklets..."
        self.fill_undecided_tracklets()

        print "Init data..."
        self.X = []
        self.y = []

        # TODO: wait for all necesarry examples, then finish init.
        # np.random.seed(13)
        self.__train_rfc()
        print "TRAINED"

        self.GT = None
        # try:
        #     # with open(self.p.working_directory+'/GT_sparse.pkl', 'rb') as f:
        #     with open('/Users/flipajs/Dropbox/dev/ferda/data/GT/Cam1_sparse.pkl', 'rb') as f:
        #         self.GT = pickle.load(f)
        # except IOError:
        #     pass

    def set_eps_certainty(self, eps):
        self._eps_certainty = eps

    def compute_distinguishability(self):
        num_a = len(self.p.animals)

        min_weighted_difs = np.zeros((num_a, num_a))
        total_len = np.zeros((num_a, 1))

        for id_ in self.tracklet_measurements:
            t = self.p.chm[id_]
            l_ = t.length()

            total_len += l_
            m = self.tracklet_measurements[id_]

            i = np.argmax(m)
            difs = (m[i] - m) * l_
            total_len[i] += l_
            min_weighted_difs[i, :] += difs

        for i in range(num_a):
            print i, min_weighted_difs[i, :] / total_len[i]

        print self.rfc.feature_importances_

    def load_learning(self):
        try:
            with open(self.p.working_directory+'/learning.pkl', 'rb') as f:
                d = pickle.load(f)
                self.user_decisions = d['user_decisions']
                self.undecided_tracklets = d['undecided_tracklets']

                self.__train_rfc()
        except IOError:
            pass

    def save_learning(self):
        with open(self.p.working_directory+'/learning.pkl', 'wb') as f:
            print "SAVING learning.pkl"
            d = {'user_decisions': self.user_decisions, 'undecided_tracklets': self.undecided_tracklets}
            pickle.dump(d, f)

    def fill_undecided_tracklets(self):
        for t in self.p.chm.chunk_gen():
            if t.id() in self.collision_chunks:
                continue

            self.undecided_tracklets.add(t.id())

    def update_undecided_tracklets(self):
        self.undecided_tracklets = set()
        for t in self.p.chm.chunk_gen():
            # if t.id() in self.collision_chunks:
            #     continue
            if not self.__tracklet_is_decided(t.P, t.N):
                self.undecided_tracklets.add(t.id())

        # TODO: remove in future, this is to fix already labeled data...
        for t in self.p.chm.chunk_gen():
            if t.id() in self.undecided_tracklets:
                if self.__only_one_P_possibility(t):
                    id_ = self.__get_one_possible_P(t)
                    self.__if_possible_update_P(t, id_)

    def run_learning(self):
        while len(self.undecided_tracklets):
            self.next_step()

    def __human_in_the_loop_request(self):
        # TODO: raise question

        # TODO:
        try:
            best_candidate_tracklet = self.__get_best_question()

            id_ = -1
            if not self.question_callback:
                # to speed up testing - simulate Human in the Loop by asking GT
                id_ = self.__DEBUG_get_answer_from_GT(best_candidate_tracklet)
            else:
                # id_ = self.question_callback(best_candidate_tracklet)
                self.question_callback(best_candidate_tracklet)

            if id_ > -1:
                print 'Human in the loop says: tracklet id: {} is animal ID: {}'.format(best_candidate_tracklet.id(), id_)
                self.assign_identity(id_, best_candidate_tracklet)
                return True
            else:
                print "DEBUG_get_answer didn't returned an ID"
        except:
            print "human in the loop failed"

        return False

    def __DEBUG_GT_test(self, id_, tracklet):
        gt_id_ = self.__DEBUG_get_answer_from_GT(tracklet)

        if gt_id_ == -1:
            print "DEBUG: GT info not available. Tracklet id: ", tracklet.id()
            return True

        return gt_id_ == id_

    def __DEBUG_get_answer_from_GT(self, tracklet):
        best_id = -1

        if self.GT is None:
            return best_id

        values = None
        region = None
        for frame in range(tracklet.start_frame(self.p.gm), tracklet.end_frame(self.p.gm) + 1):
            if frame in self.GT:
                values = self.GT[frame]
                rch = RegionChunk(tracklet, self.p.gm, self.p.rm)
                region = rch.region_in_t(frame)
                break

        if values is not None:
            for id_, val in enumerate(values):
                # TODO: global parameter
                best_dist = 20

                gt_pt = np.array(val)

                try:
                    d = np.linalg.norm(gt_pt - region.centroid())
                except:
                    return -1

                if d < best_dist:
                    if gt_pt[0] > -1:
                        best_dist = d
                        best_id = id_

        return best_id

    def __get_best_question(self, strategy='default'):
        best_tracklet = None
        best_value = -1
        for t_id in self.undecided_tracklets:
            tracklet = self.p.chm[t_id]
            in_time = set(self.p.chm.chunks_in_interval(tracklet.start_frame(self.p.gm),
                                                        tracklet.end_frame(self.p.gm)))
            # find tracklet which overlaps with max number of other tracklets
            if strategy == 'max_tracklets':
                val = len(in_time)

            elif strategy == 'longest_impact':
                min_t = np.inf
                max_t = 0
                for t in in_time:
                    if t.start_frame(self.p.gm) < min_t:
                        min_t = t.start_frame(self.p.gm)

                    if t.end_frame(self.p.gm) > max_t:
                        max_t = t.start_frame(self.p.gm)

                val = max_t - min_t

            elif strategy == 'default' or strategy == 'longest_tracklet':
                # DEFAULT - find longest tracklet...
                val = tracklet.length()

            if best_value < val:
                best_value = val
                best_tracklet = tracklet

        return best_tracklet

    def __train_rfc(self):
        self.rfc = RandomForestClassifier(class_weight='balanced_subsample')
        if len(self.X):
            print "TRAINING RFC"
            self.rfc.fit(self.X, self.y)
            self.__precompute_measurements()

    def __precompute_measurements(self):
        for t_id in self.undecided_tracklets:
            tracklet = self.p.chm[t_id]
            x, t_length = self.__get_tracklet_proba(tracklet)

            # c
            # print x

            self.tracklet_measurements[tracklet.id()] = x
            self.__update_certainty(tracklet)

    def precompute_features_(self):
        features = {}
        i = 0
        for ch in self.p.chm.chunk_gen():
            if ch in self.collision_chunks:
                continue

            X = self.get_data(ch)

            i += 1
            features[ch.id()] = X

            print i

        return features

    def get_candidate_chunks(self):
        # TODO: do it better... What if first chunks are merged...

        # from core.graph.region_chunk import RegionChunk
        #
        # avg_areas = []
        # areas = []
        # for ch in project.chm.chunk_gen():
        #     rch = RegionChunk(ch, project.gm, project.rm)
        #
        #     areas_sum = 0
        #     for r in rch.regions_gen():
        #         areas.append(r.area())
        #         areas_sum += r.area()
        #
        #     avg_areas.append(areas_sum / rch.chunk_.length())
        #
        # import numpy as np
        # import matplotlib.mlab as mlab
        # import matplotlib.pyplot as plt
        #
        # n, bins, patches = plt.hist(areas, 50, normed=1, facecolor='green', alpha=0.75)
        # n, bins, patches = plt.hist(avg_areas, 50, normed=1, facecolor='red', alpha=0.75)
        # # l = plt.plot(bins)
        # plt.show()
        #

        vertices = self.p.gm.get_vertices_in_t(0)

        areas = []
        for v in vertices:
            t = self.p.chm[self.p.gm.g.vp['chunk_start_id'][self.p.gm.g.vertex(v)]]

            for r in RegionChunk(t, self.p.gm, self.p.rm):
                areas.append(r.area())

        areas = np.array(areas)
        area_mean_thr = np.mean(areas) + 2*np.std(areas)

        print "MEAN: {} STD: {} ".format(np.mean(areas), np.std(areas))
        print "THRESHOLD = ", area_mean_thr


        print "ALL CHUNKS:", len(self.p.chm)
        # filtered = []

        # TODO: remvoe
        i = 0
        for ch in self.p.chm.chunk_gen():
            # if i > 1:
            #     continue

            i += 1

            # if ch.start_frame(self.p.gm) > 500:
            #     continue
            # else:
            # filtered.append(ch)

            if ch.length() > 0:
                ch_start_vertex = self.p.gm.g.vertex(ch.start_node())

                # ignore chunks of merged regions
                # is_merged = False
                # for e in ch_start_vertex.in_edges():
                #     if self.p.gm.g.ep['score'][e] == 0 and ch_start_vertex.in_degree() > 1:
                #         # is_merged = True
                #         self.collision_chunks[ch.id()] = True
                #         break

                rch = RegionChunk(ch, self.p.gm, self.p.rm)

                if ch.length() > 0:
                    sum = 0
                    for r in rch.regions_gen():
                        sum += r.area()

                    # area_mean = sum/float(ch.length())

                    area_mean = sum/float(ch.length())
                    c = 'C' if ch.id() in self.collision_chunks else ' '

                    p = 'C' if area_mean > area_mean_thr else ' '
                    print "%s %s %s area: %.2f, id:%d, length:%d" % (p==c, c, p, area_mean, ch.id(), ch.length())

                    if area_mean > area_mean_thr:
                        self.collision_chunks[ch.id()] = True


                # if not is_merged:
                #     filtered.append(ch)

        # print "FILTERED: ", len(filtered)
        #
        # filtered = sorted(filtered, key=lambda x: x.start_frame(self.p.gm))
        # return filtered

        # return filtered

    def __learn(self, ch, id_):
        if len(self.features) == 0:
            return

        print "LEARNING ", id_
        if ch.id() not in self.features:
            print "cached features are missing. COMPUTING..."
            X = self.get_data(ch)
            print "Done"
        else:
            X = self.features[ch.id()]

        # if empty, create... else there is a problem with vstack...
        if len(self.y) == 0:
            self.X = np.array(X)
            self.y = np.array([id_] * len(X))
        else:
            self.X = np.vstack([self.X, np.array(X)])
            y = [id_] * len(X)
            self.y = np.append(self.y, np.array(y))

    def __get_ids(self):
        """
        TODO: improve

        returns set of ids based on number of vertices in first frame. It is not good solution.
        Problems occurs when
         1) there is a noise chunk in frame 0
         2) there is an ant missing in frame 0

        Returns:

        """

        vertices = map(self.p.gm.g.vertex, self.p.gm.get_vertices_in_t(0))
        return set(range(len(vertices)))

    def __reset_chunk_PN_sets(self):
        for ch in self.p.chm.chunk_gen():
            ch.P = set()
            ch.N = set()


    def next_step(self):
        eps_certainty_learning = self._eps_certainty / 2

        # if enough new data, retrain
        if len(self.X) - self.old_x_size > self.min_new_samples_to_retrain:
            t = time.time()
            self.__train_rfc()
            print "RETRAIN t:", time.time() - t
            self.old_x_size = len(self.X)
            self.next_step()
            return True

        # pick one with best certainty
        # TODO: it is possible to improve speed (if necessary) implementing dynamic priority queue

        best_tracklet_id = max(self.tracklet_certainty.iteritems(), key=operator.itemgetter(1))[0]
        best_tracklet = self.p.chm[best_tracklet_id]

        # if not good enough, raise question
        # different strategies... 1) pick the longest tracklet, 2) tracklet with the longest intersection impact
        certainty = self.tracklet_certainty[best_tracklet.id()]
        if certainty >= 1 - self._eps_certainty:
            learn = False
            # if good enough, use for learning
            if certainty >= 1 - eps_certainty_learning:
                learn = True

            x = self.tracklet_measurements[best_tracklet_id]

            # compute certainty
            x_ = np.copy(x)
            for id_ in best_tracklet.N:
                x_[id_] = 0

            # TODO: test conflict in other tracklet with high certainity for given ID

            id_ = np.argmax(x_)
            self.assign_identity(id_, best_tracklet, learn=learn)
        else:
            # if new training data, retrain
            if len(self.X) - self.old_x_size > self.min_new_samples_to_retrain:
                t = time.time()
                self.__train_rfc()
                print "RETRAIN t:", time.time() - t
                self.old_x_size = len(self.X)
                self.next_step()
                return True
            else:
                if not self.__human_in_the_loop_request():
                    return False

        self.update_callback()

        return True

    def get_frequence_vector_(self):
        return float(np.sum(self.class_frequences)) / self.class_frequences

    def __get_tracklet_proba(self, ch):
        X = self.features[ch.id()]
        if len(X) == 0:
            return None, 0

        probs = self.rfc.predict_proba(np.array(X))
        probs = np.mean(probs, 0)

        # probs = self.apply_consistency_rule(ch, probs)
        #
        # # normalise
        # if np.sum(probs) > 0:
        #     probs /= float(np.sum(probs))

        return probs, len(X)

    def apply_consistency_rule(self, ch, probs):
        mask = np.zeros(probs.shape)
        for id_ in self.chunk_available_ids[ch.id()]:
            mask[id_] = 1

        probs *= mask

        return probs

    def get_data(self, ch):
        X = []
        r_ch = RegionChunk(ch, self.p.gm, self.p.rm)
        i = 0
        for r in r_ch.regions_gen():
            if not r.is_virtual:
                if self.features_fliplr_hack:
                    f1_, f2_ = self.get_features(r, self.p, fliplr=True)
                    X.append(f2_)
                else:
                    f1_, f2_ = self.get_features(r, self.p, fliplr=False)

                X.append(f1_)

                i += 1

        return X

    def get_init_data(self):
        vertices = self.p.gm.get_vertices_in_t(0)

        tracklets = []
        for v in vertices:
            tracklets.append(self.p.chm[self.p.gm.g.vp['chunk_start_id'][self.p.gm.g.vertex(v)]])

        # X = []
        # y = []

        # id_ = 0
        # for t in tracklets:
        #     ch_data = self.get_data(t)
        #     X.extend(ch_data)
        #
        #     self.class_frequences.append(len(ch_data))
        #
        #     y.extend([id_] * len(ch_data))
        #
        #     id_ += 1
        #
        # self.class_frequences = np.array(self.class_frequences)

        # self.X = np.array(X)
        # self.y = np.array(y)

        # it is in this position, because we need self.X, self.y to be ready for the case when we solve something by conservation rules -> thus we will be learning -> updating self.X...
        for id_, t in enumerate(tracklets):
            self.assign_identity(id_, t)

            self.user_decisions.append({'tracklet_id': t.id(), 'type': 'P', 'ids': [id_]})

        return set(range(len(tracklets)))

    def __if_possible_update_P(self, tracklet, id_, is_in_intersection_Ps=False):
        """

        Args:
            tracklet:
            id_:
            is_in_intersection_Ps: - for strict test, whether

        Returns:

        """

        for t in self.__get_affected_undecided_tracklets(tracklet):
            if t == tracklet:
                continue

            # stronger test, but maybe not always necessary...
            # maybe it should be if there is a frame, where every tracklet has id_ in t.N ?
            if is_in_intersection_Ps:
                if id_ not in t.N:
                    return False
            else:
                # Test if there is an intersection with tracklet which has also only one possible id == id_
                if self.__only_one_P_possibility(t):
                    if id_ == self.__get_one_possible_P(t):
                        # TODO: possible CONFLICT ? At least we cannot decide this tracklet now.
                        return False

            # Test if there is an intersection with tracklet that already has id == id_
            if id_ in t.P:
                # TODO: possible CONFLICT ? At least we cannot decide this tracklet now.
                return False

        return self.assign_identity(id_, tracklet)

    def __update_P(self, id_, tracklet):
        """
        updates set P (definitelyPresent) as follows P = P.union(ids)
        then tries to add ids into N (definitelyNotPresent) in others tracklets if possible.

        Args:
            id_: list
            tracklet: class Tracklet (Chunk)

        Returns:

        """

        warnings.warn("not tested yet", UserWarning)

        # TODO: check knowledge base:
        # for k in self.tracklet_knowledge[tracklet.id()] .......

        P = tracklet.P
        N = tracklet.N

        P = P.union(id_)
        N = N.remove(id_)

        # consistency check
        if not self.__consistency_check_PN(P, N):
            # TODO: CONFLICT
            return False

        # moved from the end... I think it is better to have the decision here
        tracklet.P = P

        # if the tracklet labelling is fully decided
        if self.__tracklet_is_decided(P, N):
            self.undecided_tracklets.remove(tracklet.id())

        # update affected
        affected_tracklets = self.__get_affected_undecided_tracklets(tracklet)
        for t in affected_tracklets:
            # there was self.__if_possible_update_N(t, tracklet, ids), but it was too slow..
            self.__update_N(id_, t)

        # everything is OK
        return True

    def __tracklet_is_decided(self, P, N):
        return P.union(N) == self.all_ids

    def __update_certainty(self, tracklet):
        if len(self.tracklet_measurements) == 0:
            # print "tracklet_measurements is empty"
            return

        # ignore collision tracklets because there are no measurements etc...
        if tracklet.id() in self.collision_chunks:
            return

        P = tracklet.P
        N = tracklet.N

        import math

        # skip the oversegmented regions
        if len(P) == 0:
            x = self.tracklet_measurements[tracklet.id()]

            uni_probs = np.ones((len(x),)) / float(len(x))
            # TODO: why 0.99 and not 1.0? Maybe to give a small chance for each option independently on classifier
            alpha = (min((tracklet.length()/self.k_)**2, 0.99))

            # if it is not obvious e.g. (1.0, 0, 0, 0, 0)...
            # if 0 < np.max(x) < 1.0:
            # reduce the certainty by alpha factor depending on tracklet length
            # print x, alpha, t_length
            # x = (1-alpha) * uni_probs + alpha*x

            # compute certainty
            x_ = np.copy(x)
            for id_ in N:
                x_[id_] = 0

            i1 = np.argmax(x_)
            p1 = x_[i1]
            x_[i1] = 0
            p2 = np.max(x_)

            # take maximum probability from rest after removing definitely not present ids and the second max and compute certainty
            # for 0.6 and 0.4 it is 0.6 but for 0.6 and 0.01 it is ~ 0.98
            div = p1 + p2
            if div == 0:
                certainty = 0.0
            else:
                certainty = p1 / div
                certainty = (1-alpha) * 0.5 + alpha*certainty

            if math.isnan(certainty):
                certainty = 0.0
                # means conflict or noise tracklet
                print 'is NaN', tracklet, p1, p2, x, P, N

            self.tracklet_certainty[tracklet.id()] = certainty

    def __consistency_check_PN(self, P, N):
        if len(P.intersection(N)) > 0:
            print "WARNING: inconsistency in learning_process.", P, N
            print "Intersection of DefinitelyPresent and DefinitelyNotPresent is NOT empty!!!"

            return False

        return True

    def __find_conflict(self, tracklet, id_=None):
        if id_ is None:
            id_ = self.__DEBUG_get_answer_from_GT(tracklet)

        in_time = set(self.p.chm.chunks_in_interval(tracklet.start_frame(self.p.gm),
                                                    tracklet.end_frame(self.p.gm)))

        conflicts = []
        for t in in_time:
            if id_ in t.P:
                conflicts.append(t)

        return conflicts

    def __print_conflicts(self, conflicts, tracklet, depth=0):
        if depth > 5:
            return

        print "DEPTH ", depth

        self.print_tracklet(tracklet)
        print "CONFLICTS with", conflicts
        for t in conflicts:
            self.print_tracklet(t)

            new_conflicts = self.__find_conflict(t)
            self.__print_conflicts(new_conflicts, t, depth=depth+1)

    def __get_in_v_N_union(self, v):
        N = None

        for v_in in v.in_neighbours():
            t_ = self.p.gm.get_chunk(v_in)

            if N is None:
                N = set(t_.N)
            else:
                N = N.intersection(t_.N)

        return N

    def __get_out_v_N_union(self, v):
        N = None

        for v_out in v.out_neighbours():
            t_ = self.p.gm.get_chunk(v_out)

            if N is None:
                N = set(t_.N)
            else:
                N = N.intersection(t_.N)

        return N

    def __update_N(self, ids, tracklet, skip_in=False, skip_out=False):
        # TODO: knowledge base check

        P = tracklet.P
        N = tracklet.N

        old_len = len(N)

        N = N.union(ids)

        if len(N) == old_len:
            # nothing happened
            return True

        # consistency check
        if not self.__consistency_check_PN(P, N):
            # TODO: CONFLICT
            return False

        # propagate changes
        tracklet.N = N

        # TODO: gather all and update_certainty at the end, it is possible that it will be called multiple times
        self.__update_certainty(tracklet)

        if not skip_out:
            # update all outcoming
            for v_out in tracklet.end_vertex(self.p.gm).out_neighbours():
                t_ = self.p.gm.get_chunk(v_out)

                new_N = self.__get_in_v_N_union(v_out)

                if not new_N.issubset(t_.N):
                    # print "UPDATING OUTCOMING", tracklet, t_, t_.N, new_N
                    self.__update_N(new_N, t_)

        if not skip_in:
            # update all incoming
            for v_in in tracklet.start_vertex(self.p.gm).in_neighbours():
                t_ = self.p.gm.get_chunk(v_in)

                new_N = self.__get_out_v_N_union(v_in)

                if not new_N.issubset(t_.N):
                    # print "UPDATING INCOMING", tracklet, t_, t_.N, new_N
                    self.__update_N(new_N, t_)

        if self.__only_one_P_possibility(tracklet):
            id_ = self.__get_one_possible_P(tracklet)

            self.__if_possible_update_P(tracklet, id_, is_in_intersection_Ps=True)

        return True

    def __get_one_possible_P(self, tracklet):
        return (self.all_ids - tracklet.N).pop()

    def __only_one_P_possibility(self, tracklet):
        # if only one id possible for P
        return len(self.all_ids) - 1 == len(tracklet.N)

    def print_tracklet(self, tracklet):
        tracklet.print_info(self.p.gm)
        try:
            print "\tGT id: ", self.__DEBUG_get_answer_from_GT(tracklet)
        except:
            pass
        print "\tP:", tracklet.P, " N: ", tracklet.N

    def __if_possible_update_N(self, tracklet, present_tracklet, ids):
        """
        check whether there is a risk of tracklet being a second part of present_tracklet (undersegmentation). If not
        add it to definitelyNotPresent

        test is simple... If there is a time when the distance between tracklets is bigger then ant length, it is not
        undersegmentation. It is possible to have first check - if those tracklets don't overlap (frame range) then it
        is not a undersegmantation.
        Args:
            tracklet:
            present_tracklet:
            ids:

        Returns:

        """
        oversegmented = False

        # oversegmented = True
        # if tracklet.start_frame(self.p.gm) != present_tracklet.start_frame(self.p.gm) \
        #         or tracklet.end_frame(self.p.gm) != present_tracklet.end_frame(self.p.gm):
        #     oversegmented = False
        # else:
        #     rch1 = RegionChunk(tracklet, self.p.gm, self.p.rm)
        #     rch2 = RegionChunk(present_tracklet, self.p.gm, self.p.rm)
        #
        #     for r1, r2 in itertools.izip(rch1.regions_gen(), rch2.regions_gen()):
        #         if np.linalg.norm(r1.centroid() - r2.centroid()) > self.p.stats.major_axis_median:
        #             oversegmented = False
        #             break

        if not oversegmented:
            self.__update_N(ids, tracklet)

        return not oversegmented

    def reset_learning(self):
        self.undecided_tracklets = set()
        self.tracklet_certainty = {}
        self.tracklet_measurements = {}
        self.fill_undecided_tracklets()

        self.__reset_chunk_PN_sets()

        self.X = []
        self.y = []

        for d in self.user_decisions:
            tracklet_id = d['tracklet_id']
            tracklet = self.p.chm[tracklet_id]
            ids = d['ids']
            type = d['type']

            if len(ids) == 1:
                id_ = ids[0]
            else:
                # TODO: multi ID decisions
                warnings.warn("Multiple ids in tracklet. Not supported yet")
                return

            if type == 'P':
                self.assign_identity(id_, tracklet, learn=True, user=False, gt=True)
            elif type == 'N':
                self.__update_N(set([id_]), tracklet)

        self.__train_rfc()

    def assign_identity(self, id_, tracklet, learn=True, not_affecting=False, oversegmented=False, user=False, gt=False):
        """
        Sets set definitelyPresent (P) = ids
        and set definitelyNotPresent (N) = complement of ids in all IDS

        if learn is True, it will use tracklet as an training example

        then calls update on all affected
        Args:
            ids:
            tracklet:

        Returns:

        """

        if not isinstance(id_, int):
            print "FAIL in learning_process.py assign_identity, id is not a number"

        if user:
            if tracklet.id() in self.collision_chunks:
                del self.collision_chunks[tracklet.id()]
                print "Fixing tracklet wrongly labeled as OVERSEGMENTED"

            self.user_decisions.append({'tracklet_id': tracklet.id(), 'type': 'P', 'ids': [id_]})

        # # conflict test
        # conflicts = self.__find_conflict(tracklet, id_=id_)
        # if len(conflicts):
        #     print
        #     print "------------------- CONFLICT ------------"
        #     print tracklet, tracklet.start_frame(self.p.gm), id_
        #     print "WITH:"
        #     for c in conflicts:
        #         print c
        #
        #     print
        #     print
        #     if not gt:
        #         return

        # if not self.__DEBUG_GT_test(id_, tracklet):
        #     self.mistakes.append(tracklet)
        #
        #     try:
        #         print "\nMISTAKE ", tracklet, " P:", tracklet.P, "N: ", tracklet.N,\
        #             "MEASUREMENTS", self.tracklet_measurements[tracklet.id()], "Certainty: ", self.tracklet_certainty[tracklet.id()], "tracklet id:",  tracklet.id(), " ID: ", id_
        #     except:
        #         pass
        #
        #     # TODO: remove in future...
        #     # self.assign_identity(self.__DEBUG_get_answer_from_GT(tracklet), tracklet, learn=True)
        #
        #     # return

        print "ASSIGNING ID: ", id_, " to tracklet: ", tracklet.id(), "length: ", tracklet.length(), "start: ", tracklet.start_frame(self.p.gm), tracklet.end_frame(self.p.gm)
        try:
            print "\t\tcertainty: ", self.tracklet_certainty[tracklet.id()], " measurements: ", self.tracklet_measurements[tracklet.id()]
        except:
            pass

        # finalize
        # TODO: test collision chunk, if yes and learn - compute features...
        try:
            self.undecided_tracklets.remove(tracklet.id())
            del self.tracklet_certainty[tracklet.id()]
        except KeyError:
            # it might means it is already in user_decisions
            warnings.warn("tracklet.id(): "+str(tracklet.id())+" not in self.undecided_tracklets")

        if learn:
            self.__learn(tracklet, id_)

        id_set = set([id_])
        tracklet.P = id_set
        # and the rest of ids goes to not_present
        # we want to call this function, so the information is propagated...
        self.__update_N(self.all_ids.difference(id_set), tracklet)

        for t in self.__get_affected_undecided_tracklets(tracklet):
            self.__update_N(id_set, t)

    def __get_affected_undecided_tracklets(self, tracklet):
        """
        Returns all tracklets overlapping range <tracklet.startFrame, tracklet.endFrame>
        which ids are in self.undecided_tracklets
        """

        affected = set(self.p.chm.chunks_in_interval(tracklet.start_frame(self.p.gm),
                                                     tracklet.end_frame(self.p.gm)))

        # ignore already decided chunks...
        return filter(lambda x: x.id() in self.undecided_tracklets, affected)

    def set_min_new_samples_to_retrain(self, val):
        self.min_new_samples_to_retrain = val

if __name__ == '__main__':
    p = Project()
    p.load('/Users/flipajs/Documents/wd/zebrafish')

    p.img_manager = ImgManager(p)

    learn_proc = LearningProcess(p, use_feature_cache=False, use_rf_cache=False)

