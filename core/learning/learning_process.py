from sklearn.ensemble import RandomForestClassifier
from core.project.project import Project
from core.graph.region_chunk import RegionChunk
from skimage.measure import moments_central, moments_hu, moments_normalized, moments
from utils.img_manager import ImgManager
import cv2
from utils.img import get_img_around_pts, replace_everything_but_pts
import cPickle as pickle
import numpy as np
from libs.intervaltree.intervaltree import IntervalTree
from gui.learning.ids_names_widget import IdsNamesWidget
from PyQt4 import QtGui
import sys
import operator
import time
import itertools
import math
from utils.img import rotate_img, centered_crop, get_bounding_box, endpoint_rot
import warnings
from features import *

class LearningProcess:
    def __init__(self, p, use_feature_cache=False, use_rf_cache=False, question_callback=None, update_callback=None, ghost=False):
        if use_rf_cache:
            warnings.warn("use_rf_cache is Deprecated!", DeprecationWarning)

        self.p = p

        self.question_callback = question_callback
        self.update_callback = update_callback

        self._eps1 = 0.01

        # TODO: global parameter
        self.eps_certainty = 0.3

        self.get_features = get_features_var3
        # to solve uncertainty about head orientation... Add both
        self.features_fliplr_hack = True

        # TODO: global parameter!!!
        self.k_ = 50.0
        if self.features_fliplr_hack:
            self.k_ *= 2

        self.X = []
        self.y = []
        self.old_x_size = 0

        self.collision_chunks = {}

        # TODO: remove these
        # TODO: saving chunks info...
        self.ids_present_in_tracklet = {}
        self.ids_not_present_in_tracklet = {}

        self.p.img_manager = ImgManager(self.p, max_num_of_instances=700)

        self.all_ids = set(range(len(self.p.animals)))

        self.undecided_tracklets = set()
        self.tracklet_certainty = {}
        self.tracklet_measurements = {}

        self.user_decisions = []

        self.mistakes = []

        self.features = {}

        self.load_learning()

        if ghost:
            return

        if not use_feature_cache:
            # TODO:
            self.get_candidate_chunks()

            self.features = self.precompute_features_()

            with open(p.working_directory+'/features.pkl', 'wb') as f:
                d = {'features': self.features,
                     'collision_chunks': self.collision_chunks}
                pickle.dump(d, f, -1)
        else:
            print "LOADING features..."

            with open(p.working_directory+'/features.pkl', 'rb') as f:
                d = pickle.load(f)
                self.features = d['features']
                self.collision_chunks = d['collision_chunks']

            print "LOADED"


        print "precompute avalability"
        # basically set every chunk with full set of possible ids
        self.__precompute_availability()

        # TODO: remove this
        self.class_frequences = []

        print "undecided tracklets..."
        self.fill_undecided_tracklets()

        print "Init data..."
        self.X = []
        self.y = []
        # self.get_init_data()

        # TODO: wait for all necesarry examples, then finish init.
        # np.random.seed(13)
        self.__train_rfc()
        print "TRAINED"

        with open(p.working_directory+'/rfc.pkl', 'wb') as f:
            d = {'rfc': self.rfc, 'X': self.X, 'y': self.y, 'ids': self.all_ids,
                 'class_frequences': self.class_frequences,
                 'ids_present_in_tracklet': self.ids_present_in_tracklet,
                 'ids_not_present_in_tracklet': self.ids_not_present_in_tracklet,
                 'undecided_tracklets': self.undecided_tracklets,
                 'old_x_size': self.old_x_size,
                 'tracklet_certainty': self.tracklet_certainty,
                 'tracklet_measurements': self.tracklet_measurements}
            pickle.dump(d, f, -1)

        self.GT = None
        try:
            # with open(self.p.working_directory+'/GT_sparse.pkl', 'rb') as f:
            with open('/Users/flipajs/Dropbox/dev/ferda/data/GT/Cam1_sparse.pkl', 'rb') as f:
                self.GT = pickle.load(f)
        except IOError:
            pass

        # self.save_ids_()
        # self.reset_learning()

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

    def set_ids_(self):
        app = QtGui.QApplication(sys.argv)
        ex = IdsNamesWidget()
        ex.show()

        app.exec_()
        app.deleteLater()

    def precompute_features_(self):
        features = {}
        i = 0
        for ch in self.p.chm.chunk_gen():
            if ch in self.collision_chunks:
                continue

            # if i > 20:
            #     break
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
        for ch in self.p.chm.chunk_gen():
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

    def __solve_if_clear(self, vertices1, vertices2):
        score = np.zeros((len(vertices1), len(vertices2)))

        for i, v1 in enumerate(vertices1):
            for j, v2 in enumerate(vertices2):
                s, _, _, _ = self.p.solver.assignment_score(self.p.gm.region(v1), self.p.gm.region(v2))

                score[i, j] = s

        confirmed = []

        for i in range(len(vertices1)):
            j = np.argmax(score[i, :])
            if i == np.argmax(score[:, j]):
                confirmed.append([i, j])

        assign_new_ids = []

        if len(confirmed) == len(vertices1):
            for i, j in confirmed:
                v1 = vertices1[i]
                v2 = vertices2[j]

                ch1, _ = self.p.gm.is_chunk(v1)
                ch2, _ = self.p.gm.is_chunk(v2)

                ids1 = self.chunk_available_ids.get(ch1.id(), [])
                id1 = -1 if len(ids1) != 1 else ids1[0]
                ids2 = self.chunk_available_ids.get(ch2.id(), [])
                id2 = -1 if len(ids2) != 1 else ids2[0]

                if id1 > -1 and id2 > -1 and id1 != id2:
                    assign_new_ids = []
                    break

                if id1 == id2:
                    continue

                if id1 > -1:
                    assign_new_ids.append((id1, ch2))
                elif id2 > -1:
                    assign_new_ids.append((id2, ch1))

        b = False
        for id_, ch in assign_new_ids:
            if len(self.chunk_available_ids[ch.id()]) > 1:
                if ch.id() in self.features:
                    self.__learn(ch, id_)

                if self.__assign_id(ch, id_):
                    b = True

        return b

    def test_connected_with_merged(self, ch):
        s_vertex = ch.start_vertex(self.p.gm)
        e_vertext = ch.end_vertex(self.p.gm)

        # test previous...
        if s_vertex.in_degree() == 1:
            for v in s_vertex.in_neighbours():
                pass

            ch, _ = self.p.gm.is_chunk(v)

            ch_s_v = ch.start_vertex(self.p.gm)
            ch_e_v = ch.end_vertex(self.p.gm)

            if ch.length() <= 5 and ch_s_v.in_degree() == ch_e_v.out_degree():
                vertices1 = [v for v in ch_s_v.in_neighbours()]
                vertices2 = [v for v in ch_e_v.out_neighbours()]

                if self.__solve_if_clear(vertices1, vertices2):
                    return True

        return False

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

    def __assign_id(self, ch, id_):
        if len(self.chunk_available_ids[ch.id()]) <= 1:
            try:
                del self.undecided_chunks[ch.id()]
            except:
                pass

            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "WARNING: Strange behaviour occured attempting to assign id to already resolved chunk in __assign_id/learning_process.py"
            return False

        try:
            del self.undecided_chunks[ch.id()]
        except:
            print "PROBLEMATIC CHUNK", ch.id(),  ch.start_frame(self.p.gm), ch.end_frame(self.p.gm), ch, "A_ID: ", id_

        self.chunk_available_ids[ch.id()] = [id_]
        self.update_after_hard_decision(ch, id_)

        return True

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

    def __precompute_availability(self):
        for ch in self.p.chm.chunk_gen():
            ch.P = set()
            ch.N = set()

    def __propagate_availability(self, ch, remove_id=[]):
        S_in = set()
        affected = []
        for u in ch.start_vertex(self.p.gm).in_neighbours():
            ch_, _ = self.p.gm.is_chunk(u)
            # if ch_ in self.chunks:
            affected.append(ch_)
            S_in.update(self.chunk_available_ids[ch_.id()])

        S_out = set()
        for u in ch.end_vertex(self.p.gm).out_neighbours():
            ch_, _ = self.p.gm.is_chunk(u)
            # if ch_ in self.chunks:
            affected.append(ch_)
            S_out.update(self.chunk_available_ids[ch_.id()])

        S_self = set(self.chunk_available_ids[ch.id()])

        # first chunks
        if not S_in:
            if not S_out:
                new_S_self = S_self
            else:
                new_S_self = S_self.intersection(S_out)
        else:
            if not S_out:
                new_S_self = S_self.intersection(S_in)
            else:
                new_S_self = S_self.intersection(S_in).intersection(S_out)

        if ch.start_frame(self.p.gm) > 0:
            for id_ in S_self.difference(new_S_self):
                if id_ not in S_in:
                    in_chunks = self.p.chm.chunks_in_frame(ch.start_frame(self.p.gm)-1)
                    ids_test = set()
                    for ch_ in in_chunks:
                        ids_test.update(self.chunk_available_ids[ch_.id()])

                    # Id is lost
                    if id_ not in ids_test:
                        new_S_self.add(id_)

                if id_ not in S_out:
                    out_chunks = self.p.chm.chunks_in_frame(ch.end_frame(self.p.gm) + 1)
                    ids_test = set()
                    for ch_ in out_chunks:
                        ids_test.update(self.chunk_available_ids[ch_.id()])

                    # Id is lost
                    if id_ not in ids_test:
                        new_S_self.add(id_)

        for id_ in remove_id:
            new_S_self.discard(id_)

        if S_self == new_S_self:
            return []

        if len(S_self) < len(new_S_self):
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "WARNING: S_self < new_S_self!"

        if not new_S_self:
            print "ZERO available IDs set", ch.id(),  ch.start_frame(self.p.gm), ch.end_frame(self.p.gm), ch

        new_S_self = list(new_S_self)
        if len(new_S_self) == 1:
            self.__assign_id(ch, new_S_self[0])
            # self.update_availability(ch, new_S_self[0], learn=True)
            # in_time = set(self.p.chm.chunks_in_interval(ch.start_frame(self.p.gm), ch.end_frame(self.p.gm)))
            # in_time.remove(ch)
            # affected.extend(list(in_time))

            print "Chunk solved by ID conservation rules", ch.id(),  ch.start_frame(self.p.gm), ch.end_frame(self.p.gm), ch, "AID: ", new_S_self[0]
        else:
            self.chunk_available_ids[ch.id()] = new_S_self
            if not new_S_self:
                try:
                    del self.undecided_chunks[ch.id()]
                except:
                    pass

        return affected

    def update_availability(self, ch, id_, learn=False):
        # TODO: remove this function
        if len(self.chunk_available_ids[ch.id()]) <= 1:
            return

        if ch.id() in self.collision_chunks:
            # try:
            #     del self.undecided_chunks[ch.id()]
            # except:
            #     pass

            print "CANNOT DECIDE COLLISION CHUNK!!!"
            return

        if learn:
            self.__learn(ch, id_)

        # self.save_ids_()

        if not self.__assign_id(ch, id_):
            return

        print "Ch.id: %d assigned animal id: %d. Ch.start: %d, Ch.end: %d" % (ch.id(), id_, ch.start_frame(self.p.gm), ch.end_frame(self.p.gm))


    def update_after_hard_decision(self, ch, id_):
        queue = [self.p.gm.get_chunk(u) for u in ch.start_vertex(self.p.gm).in_neighbours()] + \
                [self.p.gm.get_chunk(u) for u in ch.end_vertex(self.p.gm).out_neighbours()]

        # remove from all chunks in same time
        in_time = set(self.p.chm.chunks_in_interval(ch.start_frame(self.p.gm), ch.end_frame(self.p.gm)))
        in_time.remove(ch)
        in_time = list(in_time)
        for ch in in_time:
            queue.extend(self.__propagate_availability(ch, remove_id=[id_]))

        while queue:
            ch = queue.pop(0)
            self.tracklet_certainty
            queue.extend(self.__propagate_availability(ch))

        self.update_callback()

    def next_step(self):
        eps_certainty_learning = self.eps_certainty / 2
        min_new_samples_to_retrain = 50

        # if enough new data, retrain
        if len(self.X) - self.old_x_size > 10000:
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
        if certainty >= 1 - self.eps_certainty:
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
            # self.save_ids_()
        else:
            # if new training data, retrain
            if len(self.X) - self.old_x_size > min_new_samples_to_retrain:
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

    def save_ids_(self):
        with open(self.p.working_directory + '/temp/chunk_available_ids.pkl', 'wb') as f_:
            d_ = {'ids_present_in_tracklet': self.ids_present_in_tracklet,
                  'ids_not_present_in_tracklet': self.ids_not_present_in_tracklet,
                  'probabilities': self.tracklet_measurements,
                  'rfc': self.rfc,
                  'X': self.X,
                  'y': self.y}
            pickle.dump(d_, f_, -1)

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

    def classify_chunk(self, ch, proba):
        pass

    def recompute_rfc(self):
        pass

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

    def __update_definitely_present(self, ids, tracklet):
        """
        updates set P (definitelyPresent) as follows P = P.union(ids)
        then tries to add ids into N (definitelyNotPresent) in others tracklets if possible.

        Args:
            ids: list
            tracklet: class Tracklet (Chunk)

        Returns:

        """

        P = tracklet.P
        N = tracklet.N

        P = P.union(ids)
        N = N.remove(ids)

        # consistency check
        if not self.__consistency_check_PN(P, N):
            return False

        # if the tracklet labellign is fully decided
        if P.union(N) == self.all_ids:
            self.undecided_tracklets.remove(tracklet.id())
            del self.tracklet_certainty[tracklet.id()]
            # TODO: commented so we can save the measurements for visualisation
            # del self.tracklet_measurements[tracklet.id()]

        # update affected
        affected_tracklets = self.__get_affected_tracklets(tracklet)
        for t in affected_tracklets:
            self.__if_possible_add_definitely_not_present(t, tracklet, ids)

        tracklet.P = P

        # everything is OK
        return True

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

        # TODO: remove this
        # # repairing conflict...
        # tracklet.P = set([gt_id])
        # tracklet.N = self.all_ids - set([gt_id])

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

    def __update_definitely_not_present(self, ids, tracklet, skip_in=False, skip_out=False):
        P = tracklet.P
        N = tracklet.N

        N = N.union(ids)

        # consistency check
        if not self.__consistency_check_PN(P, N):
            print "MISTAKES: ", self.mistakes, "\n\n"

            conflicts = self.__find_conflict(tracklet)
            self.__print_conflicts(conflicts, tracklet)

            return False

        # propagate changes
        tracklet.N = N

        self.__update_certainty(tracklet)

        if not skip_out:
            # update all outcoming
            for v_out in tracklet.end_vertex(self.p.gm).out_neighbours():
                t_ = self.p.gm.get_chunk(v_out)

                new_N = self.__get_in_v_N_union(v_out)

                if not new_N.issubset(t_.N):
                    # print "UPDATING OUTCOMING", tracklet, t_, t_.N, new_N
                    self.__update_definitely_not_present(new_N, t_)

        if not skip_in:
            # update all incoming
            for v_in in tracklet.start_vertex(self.p.gm).in_neighbours():
                t_ = self.p.gm.get_chunk(v_in)

                new_N = self.__get_out_v_N_union(v_in)

                if not new_N.issubset(t_.N):
                    # print "UPDATING INCOMING", tracklet, t_, t_.N, new_N
                    self.__update_definitely_not_present(new_N, t_)

        return True

    def print_tracklet(self, tracklet):
        tracklet.print_info(self.p.gm)
        try:
            print "\tGT id: ", self.__DEBUG_get_answer_from_GT(tracklet)
        except:
            pass
        print "\tP:", tracklet.P, " N: ", tracklet.N

    def __if_possible_add_definitely_not_present(self, tracklet, present_tracklet, ids):
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
            self.__update_definitely_not_present(ids, tracklet)

        return not oversegmented

    def reset_learning(self):
        self.undecided_tracklets = set()
        self.tracklet_certainty = {}
        self.tracklet_measurements = {}
        self.fill_undecided_tracklets()

        self.__precompute_availability()

        # TODO: fill self.X and self.Y only with tracklets decided by user (self.user_decisions)
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
                return

            if type == 'P':
                self.assign_identity(id_, tracklet, learn=True, user=False, gt=True)
            elif type == 'N':
                self.__update_definitely_not_present(set([id_]), tracklet)

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

        if tracklet.id() in self.collision_chunks:
            del self.collision_chunks[tracklet.id()]
            print "Fixing tracklet wrongly labeled as OVERSEGMENTED"

        if user:
            self.user_decisions.append({'tracklet_id': tracklet.id(), 'type': 'P', 'ids': [id_]})

        # conflict test
        conflicts = self.__find_conflict(tracklet, id_=id_)
        if len(conflicts):
            print
            print "------------------- CONFLICT ------------"
            print tracklet, tracklet.start_frame(self.p.gm), id_
            print "WITH:"
            for c in conflicts:
                print c

            print
            print
            if not gt:
                return

        if tracklet.id() == 612:
            print 612

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
        except KeyError:
            pass

        if len(self.tracklet_measurements):
            try:
                del self.tracklet_certainty[tracklet.id()]
            except KeyError:
                pass

            # TODO: commented so we can save it for visualisation
            #     del self.tracklet_measurements[tracklet.id()]

        if learn:
            self.__learn(tracklet, id_)

        id_set = set([id_])
        tracklet.P = id_set
        # and the rest of ids goes to not_present
        # we want to call this function, so the information is propagated...
        self.__update_definitely_not_present(self.all_ids.difference(id_set), tracklet)

        # if affecting:
        affected_tracklets = self.__get_affected_tracklets(tracklet)

        for t in affected_tracklets:
            self.__if_possible_add_definitely_not_present(t, tracklet, id_set)


    def __get_affected_tracklets(self, tracklet):
        """
        Returns all tracklets overlapping range <tracklet.startFrame, tracklet.endFrame>

        Args:
            tracklet:

        Returns:

        """

        affected = set(self.p.chm.chunks_in_interval(tracklet.start_frame(self.p.gm),
                                                     tracklet.end_frame(self.p.gm)))
        affected.remove(tracklet)
        affected = list(affected)

        return affected



if __name__ == '__main__':
    p = Project()
    p.load('/Users/flipajs/Documents/wd/zebrafish')
    p.img_manager = ImgManager(p)

    learn_proc = LearningProcess(p, use_feature_cache=False, use_rf_cache=False)

