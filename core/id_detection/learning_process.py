import cPickle as pickle
import operator
import sys
import time
import warnings

from itertools import izip
from tqdm import tqdm, trange
import numpy as np
import os
import psutil
from PyQt4 import QtGui
from sklearn.ensemble import RandomForestClassifier

from core.graph.region_chunk import RegionChunk
from core.project.project import Project
from features import get_basic_properties, get_colornames_hists, get_colornames_and_basic
from gui.learning.ids_names_widget import IdsNamesWidget
from utils.img_manager import ImgManager
from utils.video_manager import get_auto_video_manager
import itertools
import math
from utils.img import rotate_img, centered_crop, get_bounding_box, endpoint_rot

from utils.misc import print_progress

CNN_SOFTMAX = 1
RFC = 2

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
    def __init__(self, p, use_feature_cache=False, use_rf_cache=False, question_callback=None, update_callback=None,
                 ghost=False, verbose=0, id_N_propagate=True, id_N_f=True, progressbar_callback=None):
        if use_rf_cache:
            warnings.warn("use_rf_cache is Deprecated!", DeprecationWarning)

        self.p = p
        self.progressbar_callback = progressbar_callback
        self.verbose = verbose
        self.show_key_error_warnings = False

        self.classifier_name = CNN_SOFTMAX

        self.min_new_samples_to_retrain = 10000000
        self.rf_retrain_up_to_min = np.inf

        # TODO: add whole knowledge class...
        self.tracklet_knowledge = {}

        self.question_callback = question_callback
        self.update_callback = update_callback

        self._eps_certainty = 0.3

        self.id_N_propagate = id_N_propagate
        self.id_N_f = id_N_f

        # TODO: make standalone feature extractor...
        self.get_features = get_colornames_hists

        # to solve uncertainty about head orientation... Add both
        self.features_fliplr_hack = True

        # TODO: global parameter!!!
        self.min_tracklet_len = 50.0

        self.X = []
        self.y = []
        self.old_x_size = 0

        self.rf_max_features = 'auto'
        self.rf_n_estimators = 10
        self.rf_min_samples_leafs = 1
        self.rf_max_depth = None

        # TODO:
        self.force = True

        self.collision_chunks = {}

        self.p.img_manager = ImgManager(self.p, max_num_of_instances=700)

        self.all_ids = set(range(len(self.p.animals)))

        self.undecided_tracklets = set()
        self.tracklet_certainty = {}
        self.tracklet_measurements = {}
        self.tracklet_stds = {}

        self.user_decisions = []

        self.mistakes = []

        self.features = {}

        self.load_learning()

        self.separated_frame = -1

        self.consistency_violated = False
        self.last_id = -1

        self.ignore_inconsistency = False

        self.map_decisions = False

        self.human_in_the_loop = False

        # tracklet agglomeration links
        self.links = None

        # when creating without feature data... e.g. in main_tab_widget
        if ghost:
            return

        # False mean - don't ask
        self.human_in_the_loop = True
        self.GT_in_the_loop = False

        if self.GT_in_the_loop:
            from utils.gt.gt import GT
            self.GT = GT()
            self.GT.load(self.p.GT_file)

        try:
            # with open(self.p.working_directory+'/GT_sparse.pkl', 'rb') as f:
            with open('/Users/flipajs/Dropbox/dev/ferda/data/GT/Cam1_.pkl', 'rb') as f:
                self.GT = pickle.load(f)
        except IOError:
            pass


    def load_features(self, db_names='fm.sqlite3'):
        from core.id_detection.features import FeatureManager

        self.progressbar_callback(True)
        # TODO: speedup loading... What about loading 10% longest tracklets... When solved, load more?
        self.collision_chunks = set()
        self.features = {}

        self.fms = []
        if isinstance(db_names, str):
            db_names = [db_names]

        for n in db_names:
            self.fms.append(FeatureManager(self.p.working_directory, db_name=n, use_cache=False))

        expected_sum = len(self.p.animals) * self.p.img_manager.vid.total_frame_count()
        t_sum = 0
        for i, t in enumerate(self.p.chm.chunk_gen()):
            # if t.is_single():
            #     if len(t) < 10:
            #         continue

                r_ids = [id_ for id_ in t.rid_gen(self.p.gm)]

                ff = []
                for fm in self.fms:
                    _, f = fm[r_ids]

                    ff.append(f)

                F = []
                #merge
                for j in range(len(ff[0])):
                    X = []
                    for fi in range(len(self.fms)):
                        f = ff[fi][j]
                        if f is None:
                            X = []
                            break

                        X.extend(f)
                    if len(X) > 0:
                        F.append(X)

                if len(F):
                    self.features[t.id()] = F
                else:
                    self.collision_chunks.add(t.id())

                # Debug info...
                if i % 500 == 0:
                    process = psutil.Process(os.getpid())
                    # print
                    # print "Memory usage: {:.2f}Mb".format((process.memory_info().rss) / 1e6)
                    # print

                t_sum += len(t)
                print_progress(t_sum, expected_sum)
            # elif t.is_multi():
            #     self.collision_chunks.add(t.id())

        print_progress(expected_sum, expected_sum, "", "LOADED")
        print len(self.features), len(self.collision_chunks)
        self.update_undecided_tracklets()
        self.progressbar_callback(False)

    def compute_features(self):
        self.compute_features_mp()

        # print "computing features..."
        # time_ = time.time()
        # from core.id_detection.features import get_colornames_hists
        # from core.id_detection.feature_manager import FeatureManager
        #
        # # TODO:
        # fm = FeatureManager(self.p.working_directory, db_name='fm_sp.sqlite3')
        #
        # len_sum = 0
        # expected_sum = len(self.p.animals) * self.p.img_manager.vid.total_frame_count()
        # for i, t in enumerate(self.p.chm.chunk_gen()):
        #     len_sum += len(t)
        #
        #     print_progress(min(len_sum, expected_sum), expected_sum)
        #     if not t.is_single():
        #         continue
        #
        #     for r in RegionChunk(t, self.p.gm, self.p.rm).regions_gen():
        #         if fm[r.id()][1] == [None]:
        #             f = get_colornames_hists(r, self.p, saturated=True, lvls=1)
        #             fm.add(r.id(), f)
        #
        # print "total time: ", time.time() - time_

    def compute_features_mp(self):
        print "computing features... (MP)"
        t = time.time()
        # set num cores to 1, so parallelisation works
        import cv2
        from utils.mp_counter import Counter
        from multiprocessing import Queue, cpu_count, Process, Lock
        default_num_threads = cv2.getNumThreads()
        cv2.setNumThreads(0)

        q_tasks = Queue()
        counter = Counter(0)
        num_cpus = max(1, cpu_count() - 1)
        lock = Lock()

        num_frames = self.p.img_manager.vid.total_frame_count()
        children = []
        for i in range(num_cpus):
            p = Process(target=compute_features_process, args=(counter, lock, q_tasks, self.p.working_directory, num_frames))
            p.start()
            children.append(p)

        step = 100

        for frame in range(0, num_frames, step):
            img = self.p.img_manager.get_whole_img(frame)
            h, w, c = img.shape
            img.shape = (h*w*c)
            q_tasks.put((frame, min(frame+step, num_frames)))

        q_tasks.put(None)

        for p in children:
            p.join()

        print_progress(num_frames, num_frames, "features computation in progress:", "FINIHSED")
        print "all joined"

        # set back to default
        cv2.setNumThreads(default_num_threads)
        print "total time: ", time.time() - t

    def set_eps_certainty(self, eps):
        self._eps_certainty = eps

    def set_tracklet_length_k(self, k):
        self.min_tracklet_len = k
        try:
            self.update_undecided_tracklets()
            try:
                self.__precompute_measurements(only_unknown=True)
            except Exception as e:
                print e

        except AttributeError:
            pass
        except IndexError:
            pass

    def compute_distinguishability(self):
        num_a = len(self.p.animals)

        min_weighted_difs = np.zeros((num_a, num_a))
        total_len = np.zeros((num_a, 1))

        for id_ in self.tracklet_measurements:
            t = self.p.chm[id_]
            l_ = t.length()

            total_len += l_
            m = self.tracklet_measurements[id_]


            ids = self.GT.tracklet_id_set_without_checks(t, self.p)
            # i = np.argmax(m)
            i = ids[0]
            # difs = (m[i] - m) * l_
            difs = (m) * l_
            total_len[i] += l_
            min_weighted_difs[i, :] += difs

        for i in range(num_a):
            print i, min_weighted_difs[i, :] / total_len[i]

        print self.classifier.feature_importances_

    def load_learning(self):
        try:
            with open(self.p.working_directory+'/learning.pkl', 'rb') as f:
                d = pickle.load(f)
                self.user_decisions = d['user_decisions']
                self.undecided_tracklets = d['undecided_tracklets']
                self.tracklet_measurements = d['measurements']
                self.tracklet_stds = d['stds']
                self.tracklet_certainty = d['certainty']
                self.classifier = d['rfc']
                self.IF_region_anomaly = d['IF_region_anomaly']
                self.LR_region_anomaly = d['LR_region_anomaly']

                self.load_features()

        except Exception as e:
            print e

    def save_learning(self):
        with open(self.p.working_directory+'/learning.pkl', 'wb') as f:
            print "SAVING learning.pkl"
            d = {'user_decisions': self.user_decisions,
                 'undecided_tracklets': self.undecided_tracklets,
                 'measurements': self.tracklet_measurements,
                 'stds': self.tracklet_stds,
                 'certainty': self.tracklet_certainty,
                 'rfc': self.classifier,
                 'IF_region_anomaly': self.IF_region_anomaly,
                 'LR_region_anomaly': self.LR_region_anomaly
                 }

            pickle.dump(d, f)

    def fill_undecided_tracklets(self):
        for t in self.p.chm.chunk_gen():
            # if t.id() in self.collision_chunks or t.is_multi():
            if not t.is_single() or t.length() < self.min_tracklet_len:
                continue

            self.undecided_tracklets.add(t.id())

    def update_undecided_tracklets(self):
        print "Updating undecided tracklets..."
        self.undecided_tracklets = set()
        for t in tqdm(self.p.chm.chunk_gen()):
            if not t.is_single() or t.length() < self.min_tracklet_len:
                continue

            # # skip all non root tracklets
            # if self.links is not None:
            #     if t.id() in self.links and t.id() != self.links[t.id()]:
            #         continue

            if not self.__tracklet_is_decided(t.P, t.N):
                self.undecided_tracklets.add(t.id())

    def run_learning(self):
        while len(self.undecided_tracklets):
            self.next_step()

    def __human_in_the_loop_request(self):
        # TODO: raise question

        # TODO:
        try:
            best_candidate_tracklet = self.get_best_question()

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

    def get_best_question(self, strategy='default'):
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

    def __tid2features(self):
        X = []
        for t_id in self.X:
            x = self.features[t_id]
            if len(X) == 0:
                X = np.array(x)
            else:
                X = np.vstack([X, np.array(x)])

        return X

    def train(self, init=False, use_xgboost=False):
        if self.classifier_name == RFC:
            self.__train_rfc(init=init, use_xgboost=False)
        elif self.classifier_name == CNN_SOFTMAX:
            with open(self.p.working_directory+'/softmax_results_map.pkl') as f:
                self.cnn_results_map = pickle.load(f)

            self.__precompute_measurements()

    def __train_rfc(self, init=False, use_xgboost=False):
        if use_xgboost:
            from xgboost import XGBClassifier
            self.classifier = XGBClassifier()
        else:
            self.classifier = RandomForestClassifier(class_weight='balanced_subsample',
                                                     max_features=self.rf_max_features,
                                                     n_estimators=self.rf_n_estimators,
                                                     min_samples_leaf=self.rf_min_samples_leafs,
                                                     max_depth=self.rf_max_depth)
        if len(self.X):
            y = []
            for i in range(len(self.p.animals)):
                y.append(np.sum(np.array(self.y) == i))

            # and - allow it if it is a first training.
            if min(y) >= self.rf_retrain_up_to_min and not init:
                return False

            print "TRAINING RFC", y
            t = time.time()
            self.classifier.fit(self.__tid2features(), self.y)
            print "t: {:.2f}s".format(time.time() - t)
            self.__precompute_measurements()

            return y

        return False

    def __precompute_measurements(self, only_unknown=False):
        from tqdm import tqdm
        print "Computing ID probability distributions..."
        for t_id in tqdm(self.undecided_tracklets):
            tracklet = self.p.chm[t_id]
            if only_unknown:
                if t_id in self.tracklet_certainty:
                    continue

            # # find root tracklet and set of all connected tracklets
            # t_set = []
            # if t_id not in self.links:
            #     t_set = [t_id]
            # else:
            #     if t_id
            #
            # root_t_id = self.links[t_id]
            # t_set = [v for k, v in self.links.iteritems() if v == root_t_id]
            # t_set.append(root_t_id)

            try:
                x, t_length, stds = self._get_tracklet_proba(tracklet)
            except KeyError:
                warnings.warn("used random class probability distribution for tracklet it: {}".format(t_id))
                # TODO: remove this, instead compute features...
                x = np.random.rand(len(self.p.animals)) * 0
                stds = np.random.rand(len(self.p.animals))
                if self.verbose > 2:
                    print "features missing for ", tracklet.id()

            self.tracklet_measurements[tracklet.id()] = x
            self.tracklet_stds[tracklet.id()] = stds

        for t_id in tqdm(self.undecided_tracklets):
            if only_unknown:
                if t_id in self.tracklet_certainty:
                    continue

            self._update_certainty(self.p.chm[t_id])

    def precompute_features_(self):
        from utils.misc import print_progress

        features = {}
        i = 0

        print "COMPUTING FEATURES..."
        ch_num = len(self.p.chm)

        for ch in self.p.chm.chunk_gen():
            if ch.id() in self.collision_chunks:
                continue

            X = self.get_data(ch)

            i += 1
            features[ch.id()] = X

            print_progress(i, ch_num)
        print_progress(ch_num, ch_num)
        print "DONE"

        return features

    def get_candidate_chunks(self):
        vertices = self.p.gm.get_vertices_in_t(self.separated_frame)

        areas = []
        for v in vertices:
            t = self.p.chm[self.p.gm.g.vp['chunk_start_id'][self.p.gm.g.vertex(v)]]

            for r in RegionChunk(t, self.p.gm, self.p.rm):
                areas.append(r.area())

        areas = np.array(areas)
        area_mean_thr = np.mean(areas) + 2*np.std(areas)

        print "SINGLE / MERGED human_iloop_classification..."
        print "MEAN: {} STD: {} ".format(np.mean(areas), np.std(areas))
        print "THRESHOLD = ", area_mean_thr


        print "ALL CHUNKS:", len(self.p.chm)
        # filtered = []

        i = 0
        num_chunks = len(self.p.chm)
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
                    # print "%s %s %s area: %.2f, id:%d, length:%d" % (p==c, c, p, area_mean, ch.id(), ch.length())

                    if area_mean > area_mean_thr:
                        self.collision_chunks.add(ch.id())

                print_progress(i, num_chunks)

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

        # X = [r_id for r_id in ch.rid_gen(self.p.gm)]

        # print "LEARNING ", id_
        # if ch.id() not in self.features:
        #     print "cached features are missing. COMPUTING..."
        #     X = self.get_data(ch)
        #     print "Done"
        # else:
        #     X = self.features[ch.id()]

        # if empty, create... else there is a problem with vstack...
        # if len(self.y) == 0:
        #     self.X = np.array(X)
        #     self.y = np.array([id_] * len(X))
        # else:
        #     self.X = np.vstack([self.X, np.array(X)])
        y = [id_] * len(self.features[ch.id()])
        # TODO: performance
        if ch.id() not in self.X:
            self.X.append(ch.id())
            self.y.extend(y)

        # self.y = np.append(self.y, np.array(y))

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

    def _reset_chunk_PN_sets(self):
        full_set = set(range(len(self.p.animals)))
        for ch in self.p.chm.chunk_gen():
            ch.P = set()
            ch.N = set()

            if ch.is_noise() or ch.is_part() or ch.is_undefined():
                ch.N = set(full_set)

            if self.map_decisions:
                try:
                    del ch.decision_certainty
                    del ch.measurements
                except:
                    pass

    def next_step(self, update_gui=True):
        if len(self.tracklet_certainty) == 0:
            # print "ALL is done"
            return True

        eps_certainty_learning = self._eps_certainty / 2

        # if enough new data, retrain
        if len(self.X) - self.old_x_size > self.min_new_samples_to_retrain:
            t = time.time()
            if self.train():
                print "RETRAIN t:", time.time() - t
                self.old_x_size = len(self.X)
                self.next_step()
                return True

        # pick one with best certainty
        # TODO: it is possible to improve speed (if necessary) implementing dynamic priority queue
        try:
            best_tracklet_id = max(self.tracklet_certainty.iteritems(), key=operator.itemgetter(1))[0]
        except ValueError:
            print len(self.tracklet_certainty), len(self.undecided_tracklets)

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
            x_ = np.sum(x, axis=0)
            for id_ in best_tracklet.N:
                x_[id_] = 0

            # TODO: test conflict in other tracklet with high certainity for given ID

            id_ = np.argmax(x_)
            self.last_id = id_
            self.assign_identity(id_, best_tracklet, learn=learn)
        else:
            # if new training data, retrain
            if len(self.X) - self.old_x_size > self.min_new_samples_to_retrain:
                t = time.time()
                self.train()
                print "RETRAIN t:", time.time() - t
                self.old_x_size = len(self.X)
                self.next_step()
                return True
            else:
                if self.human_in_the_loop:
                    if not self.__human_in_the_loop_request():
                        return False
                else:
                    self.last_id = best_tracklet_id
                    self.consistency_violated = True

        if self.update_callback is not None and update_gui:
            self.update_callback()

        return True

    def get_frequence_vector_(self):
        return float(np.sum(self.class_frequences)) / self.class_frequences

    def _get_tracklet_proba(self, ch, debug=False):
        if self.classifier_name == RFC:
            X = self.features[ch.id()]
            if len(X) == 0:
                return None, 0

            anomaly_probs = self.get_tracklet_anomaly_probs(ch)

        try:
            if self.classifier_name == CNN_SOFTMAX:
                probs = []
                for r_id in ch.rid_gen(self.p.gm):
                    probs.append(self.cnn_results_map[r_id])

                probs = np.array(probs)
            else:
                probs = self.classifier.predict_proba(np.array(X))
            if debug:
                print "Probs:"
                print probs

                wprobs = np.copy(probs)
                for i in range(probs.shape[0]):
                    wprobs[i] *= anomaly_probs[i]

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ind = np.arange(probs.shape[1])
                width = 0.35
                ax.bar(ind, np.mean(probs, 0), width, yerr=np.std(probs, 0, ddof=1))
                ax.bar(ind + width, np.mean(wprobs, 0), width, yerr=np.std(wprobs, 0, ddof=1))
                plt.show()
                plt.ion()
        except:
            print X

        if debug:
            print "anomaly probs:"
            print anomaly_probs

        # TODO:
        if self.classifier_name == RFC:
            # todo: mat multiply
            for i in range(probs.shape[0]):
                probs[i] *= anomaly_probs[i]

        stds = np.std(probs, 0, ddof=1)
        # probs = np.sum(probs, 0)
        #
        # probs /= float(len(X))

        if debug:
            print "Probs: ", probs
            print "STDS: ", stds

        # # normalise
        # if np.sum(probs) > 0:
        #     probs /= float(np.sum(probs))


        # TODO: refactor
        # second value on return is some old relict...
        return probs, probs.shape[0], stds

    # def apply_consistency_rule(self, ch, probs):
    #     mask = np.zeros(probs.shape)
    #     for id_ in self.chunk_available_ids[ch.id()]:
    #         mask[id_] = 1
    #
    #     probs *= mask
    #
    #     return probs

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

            self.user_decisions.append({'tracklet_id_set': t.id(), 'type': 'P', 'ids': [id_]})

        return set(range(len(tracklets)))

    def __if_possible_update_P(self, tracklet, id_, is_in_intersection_Ps=False):
        """

        Args:
            tracklet:
            id_:
            is_in_intersection_Ps: - for strict test, whether

        Returns:

        """

        for t in self._get_affected_undecided_tracklets(tracklet):
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

        self.last_id = tracklet.id()
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
        affected_tracklets = self._get_affected_undecided_tracklets(tracklet)
        for t in affected_tracklets:
            # there was self.__if_possible_update_N(t, tracklet, ids), but it was too slow..

            self.last_id = t.id()
            if not self._update_N(id_, t):
                return

        # everything is OK
        return True

    def __tracklet_is_decided(self, P, N):
        return P.union(N) == self.all_ids

    def _get_p1(self, X, i):
        div = np.sum(np.power(2, X))

        return 2**X[i] / div

    def _get_p2(self, X, i):
        a = 1
        for r in X:
            a *= (1 - self._get_p1(r, i))

        div = 0
        for j in range(X.shape[1]):
            b = 1
            for r in X:
                b *= (1 - self._get_p1(r, i))

            div += self._get_p1(np.sum(X, 0), j) * b

        return (self._get_p1(np.sum(X, 0), i) * a) / div

    def get_p1(self, x, i):
        x__ = np.sum(x, axis=0)
        sum1 = np.sum([2 ** a for a in x__])
        p1 = 2 ** x__[i] / sum1

        return p1


    def _update_certainty(self, tracklet):
        if len(self.tracklet_measurements) == 0:
            # print "tracklet_measurements is empty"
            return

        # ignore collision tracklets because there are no measurements etc...
        if not tracklet.is_single():
        # if tracklet.id() in self.collision_chunks:
            return

        P = tracklet.P
        N = tracklet.N

        import math

        # skip the oversegmented regions
        if len(P) == 0:
            x = self.tracklet_measurements[tracklet.id()]

            uni_probs = np.ones((len(x),)) / float(len(x))
            # TODO: why 0.99 and not 1.0? Maybe to give a small chance for each option independently on classifier
            # alpha = (min((tracklet.length() / self.k_) ** 2, 0.99))
            # x = (1 - alpha) * uni_probs + alpha * x


            x_ = np.copy(x)
            for id_ in N:
                x_[:, id_] = 0

            # TODO: try P2 computation for all potential k values?
            # k is best predicted ID
            k = np.argmax(np.sum(x_, axis=0))
            p1 = self.get_p1(x_, k)

            # set of all other relevant regions (single regions in tracklet timespan overlap)
            C = self.p.chm.chunks_in_interval(tracklet.start_frame(self.p.gm), tracklet.end_frame(self.p.gm))

            term1 = 1
            term2 = 0

            for t in C:
                if not t.is_single() or t == tracklet or len(t.P):
                    continue

                try:
                    xx = self.tracklet_measurements[t.id()]
                    term1 *= 1 - self.get_p1(xx, k)
                except KeyError:
                    pass

            for i in range(len(self.p.animals)):
                term3 = 1
                for t in C:
                    if not t.is_single() or t == tracklet or len(t.P):
                        continue

                    try:
                        xx = self.tracklet_measurements[t.id()]
                        term3 *= 1 - self.get_p1(xx, i)
                    except KeyError:
                        pass

                term2 += self.get_p1(x_, i) * term3

            p2 = (p1 * term1) / term2
            self.tracklet_certainty[tracklet.id()] = p2

            return

            std = self.tracklet_stds[tracklet.id()]

            id1 = np.argmax(x_)
            m1 = max(0, x_[id1] - std[id1])
            x_[id1] = 0

            id2 = np.argmax(x_)
            m2 = min(1, x_[id2] + std[id2])

            certainty = m1 / (m1 + m2 + 1e-6)

            self.tracklet_certainty[tracklet.id()] = certainty

            # id_ = np.argmax(x)
            # c = self._get_p1(x, id_)
            # self.tracklet_certainty[tracklet.id()] = x[id_]

            #
            # uni_probs = np.ones((len(x),)) / float(len(x))
            # # TODO: why 0.99 and not 1.0? Maybe to give a small chance for each option independently on classifier
            # alpha = (min((tracklet.length()/self.k_)**2, 0.99))
            # # alpha = (min((tracklet.length()/self.k_)**1.1, 0.99))
            #
            # # if it is not obvious e.g. (1.0, 0, 0, 0, 0)...
            # # if 0 < np.max(x) < 1.0:
            # # reduce the certainty by alpha factor depending on tracklet length
            # # print x, alpha, t_length
            # # x = (1-alpha) * uni_probs + alpha*x
            #
            # # compute certainty
            # x_ = np.copy(x)
            # for id_ in N:
            #     x_[id_] = 0
            #
            # i1 = np.argmax(x_)
            # p1 = x_[i1]
            # x_[i1] = 0
            # p2 = np.max(x_)
            #
            # # take maximum probability from rest after removing definitely not present ids and the second max and compute certainty
            # # for 0.6 and 0.4 it is 0.6 but for 0.6 and 0.01 it is ~ 0.98
            # div = p1 + p2
            #
            # # if prob is too low...
            # if div < 0.3 and tracklet.length() < 10:
            #     div = 1.0
            #
            # if div == 0:
            #     certainty = 0.0
            # else:
            #     certainty = p1 / div
            #     certainty = (1-alpha) * 0.5 + alpha*certainty
            #
            # if math.isnan(certainty):
            #     certainty = 0.0
            #     # means conflict or noise tracklet
            #     print 'is NaN', tracklet, p1, p2, x, P, N
            #
            # self.tracklet_certainty[tracklet.id()] = certainty

    def __consistency_check_PN(self, P, N):
        if self.ignore_inconsistency:
            return True

        if len(P.intersection(N)) > 0:
            self.consistency_violated = True
            print "WARNING: inconsistency in learning_process.", P, N
            # print "Intersection of DefinitelyPresent and DefinitelyNotPresent is NOT empty!!!"

            return False

        return True

    def __find_sconflict(self, tracklet, id_=None):
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

    def __get_in_v_N_union(self, v, ignore_noise=False):
        N = None

        for v_in in v.in_neighbours():
            t_ = self.p.gm.get_chunk(v_in)

            if ignore_noise and t_.is_noise():
                continue

            if N is None:
                N = set(t_.N)
            else:
                N = N.intersection(t_.N)

        return N

    def __get_out_v_N_union(self, v, ignore_noise=False):
        N = None

        for v_out in v.out_neighbours():
            t_ = self.p.gm.get_chunk(v_out)

            if ignore_noise and t_.is_noise():
                continue

            if N is None:
                N = set(t_.N)
            else:
                N = N.intersection(t_.N)

        return N

    def _update_N(self, ids, tracklet, skip_in=False, skip_out=False):
        if not self.id_N_f:
            return

        # TODO: knowledge base check

        P = tracklet.P
        N = tracklet.N

        old_len = len(N)

        N = N.union(ids)

        if len(N) == old_len:
            # nothing happened
            return True

        self.last_id = tracklet.id()
        # consistency check
        if not self.__consistency_check_PN(P, N):
            print "ID P, N CONFLICT: ", tracklet.id(), P, N
            print "IN CONFLICT WITH: "
            # for tc in self.__find_conflict(tracklet, list(tracklet.P)[0]):
            #     print tc.id(), tc, tc.segmentation_class
            # TODO: CONFLICT
            if not self.force:
                return False

        # propagate changes
        if tracklet.is_single() and len(N) == len(self.p.animals):
            print "CONFLICT in update_N, tracklet-id: {}".format(tracklet.id())

        tracklet.N = N

        # TODO: gather all and update_certainty at the end, it is possible that it will be called multiple times
        if tracklet.id() in self.tracklet_measurements:
            self._update_certainty(tracklet)

        if self.id_N_propagate:
            if not skip_out:
                # update all outcoming
                for v_out in tracklet.end_vertex(self.p.gm).out_neighbours():
                    t_ = self.p.gm.get_chunk(v_out)

                    if len(t_.P) > 0:
                        continue

                    new_N = self.__get_in_v_N_union(v_out, ignore_noise=True)

                    if new_N is not None:
                        if not new_N.issubset(t_.N):
                            # print "UPDATING OUTCOMING", tracklet, t_, t_.N, new_N
                            if not self._update_N(new_N, t_):
                                return

            if not skip_in:
                # update all incoming
                for v_in in tracklet.start_vertex(self.p.gm).in_neighbours():
                    t_ = self.p.gm.get_chunk(v_in)

                    if len(t_.P) > 0:
                        continue

                    new_N = self.__get_out_v_N_union(v_in, ignore_noise=True)

                    if new_N is not None:
                        if not new_N.issubset(t_.N):
                            # print "UPDATING INCOMING", tracklet, t_, t_.N, new_N
                            if not self._update_N(new_N, t_):
                                return

        if self.__only_one_P_possibility(tracklet) and tracklet.is_single():
            id_ = self.__get_one_possible_P(tracklet)

            self.assign_identity(id_, tracklet, learn=False)
            # TODO: ...
            # self.__if_possible_update_P(tracklet, id_, is_in_intersection_Ps=False)

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
            if not self._update_N(ids, tracklet):
                return False

        return not oversegmented

    def get_appearance_features(self, r):
        return [
            r.area(),
            r.a_,
            r.b_,
            (r.a_ / r.b_),
            r.sxx_ ,
            r.syy_,
            r.sxy_ ,
            r.min_intensity_
        ]

    def get_tracklet_anomaly_probs(self, t):
        rch = RegionChunk(t, self.p.gm, self.p.rm)

        X = []
        for r in rch.regions_gen():
            X.append(self.get_appearance_features(r))

        vals = self.IF_region_anomaly.decision_function(X)
        probs = self.LR_region_anomaly.predict_proba(vals.reshape((len(vals), 1)))[:, 0]

        return probs

    def reset_learning(self, use_xgboost=False):
        self.undecided_tracklets = set()
        self.tracklet_certainty = {}
        self.tracklet_measurements = {}
        self.fill_undecided_tracklets()

        self._reset_chunk_PN_sets()

        self.old_x_size = 0

        self.X = []
        self.y = []

        self.consistency_violated = False
        self.last_id = -1

        from sklearn.ensemble import IsolationForest
        self.IF_region_anomaly = IsolationForest()
        region_X = []

        full_set = set(range(len(self.p.animals)))
        test_set = set()
        for d in self.user_decisions:
            if d['type'] == 'P' and len(d['ids']) == 1:
                test_set.add(d['ids'][0])

        if len(full_set.intersection(test_set)) != len(full_set) and self.classifier_name == RFC:
            QtGui.QMessageBox.information(None, '',
                                          'There are not examples for all classes. Did you use auto initialisation? Missing ids: '+str(full_set-test_set))
            return

        for d in self.user_decisions:
            tracklet_id = d['tracklet_id_set']
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

                for r_id in tracklet.rid_gen(self.p.gm):
                    region_X.append(self.get_appearance_features(self.p.rm[r_id]))

            elif type == 'N':
                self._update_N(set([id_]), tracklet)
        try:
            self.IF_region_anomaly.fit(region_X)
            vals = self.IF_region_anomaly.decision_function(region_X)
            vals_sorted = sorted(vals)

            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression()
            use_for_learning = 0.1

            part_len = int(len(vals) * use_for_learning)
            part1 = np.array(vals_sorted[:part_len])
            part2 = np.array(vals_sorted[-part_len:])

            X = np.hstack((part1, part2))
            X.shape = ((X.shape[0], 1))

            y = np.array([1 if i < len(part1) else 0 for i in range(X.shape[0])])
            lr.fit(X, y)
            self.LR_region_anomaly = lr
        except Exception as e:
            print e

        # for t in self.p.chm.chunk_gen():
        #     if not t.is_single():
        #         continue
        #     probs = self.get_tracklet_anomaly_probs(t)
        #
        #     print t.id(), np.mean(probs), np.median(probs), len(t)

        ret = self.train(init=True, use_xgboost=use_xgboost)
        print "TRAINING FINISHED"

        return ret

    def assign_identity(self, id_, tracklet, learn=True, not_affecting=False, oversegmented=False, user=False,
                        gt=False):
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
                self.collision_chunks.discard(tracklet.id())
                print "Fixing tracklet wrongly labeled as OVERSEGMENTED"

            self.user_decisions.append({'tracklet_id_set': tracklet.id(), 'type': 'P', 'ids': [id_]})

        # TODO: debug reasons:
        if self.map_decisions:
            try:
                tracklet.decision_cert = self.tracklet_certainty[tracklet.id()]
                tracklet.measurements = self.tracklet_measurements[tracklet.id()]
            except:
                pass

        if self.verbose > 2:
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
            # it might means it is already in user_decisions
            if self.show_key_error_warnings:
                warnings.warn("tracklet.id(): "+str(tracklet.id())+" not in self.undecided_tracklets")

        try:
            del self.tracklet_certainty[tracklet.id()]
        except KeyError:
            if self.show_key_error_warnings:
                warnings.warn("tracklet.id(): "+str(tracklet.id())+" not in self.tracklet_certainty")

        if tracklet.id() in self.collision_chunks:
            if self.verbose > 2:
                print tracklet.id(), "in collision chunks"
            return

        if learn:
            self.__learn(tracklet, id_)

        id_set = set([id_])
        tracklet.P = id_set

        # tracklet.N = self.all_ids.difference(id_set)

        self.last_id = id_

        # and the rest of ids goes to not_present
        # we want to call this function, so the information is propagated...
        self._update_N(self.all_ids.difference(id_set), tracklet)

        for t in self._get_affected_undecided_tracklets(tracklet):
            self._update_N(id_set, t)

    def _get_affected_undecided_tracklets(self, tracklet):
        """
        Returns all tracklets overlapping range <tracklet.startFrame, tracklet.endFrame>
        which ids are in self.undecided_tracklets
        """

        affected = set(self.p.chm.chunks_in_interval(tracklet.start_frame(self.p.gm),
                                                     tracklet.end_frame(self.p.gm)))

        # ignore already decided chunks...
        return filter(lambda x: (x.is_single() or x.is_multi()) and not self.__tracklet_is_decided(x.P, x.N), affected)
        # return filter(lambda x: x.id() in self.undecided_tracklets, affected)

    def set_min_new_samples_to_retrain(self, val):
        self.min_new_samples_to_retrain = val

    def edit_tracklet(self, tracklet, new_P, new_N, method='fix_tracklet_only'):
        old_P = set(tracklet.P)
        old_N = set(tracklet.N)

        print "edit tracklet old P", old_P, "old N", old_N, "new P", new_P, "new N", new_N, method, tracklet

        if method == 'fix_affected':
            for t in self._get_affected_undecided_tracklets(tracklet):
                print t
        elif method == 'fix_tracklet_only':
            # TODO: check conflict?
            tracklet.N = new_N
            tracklet.P = new_P

    def _get_and_train_rfc(self, g, use_xgboost=False):
        X = []
        y = []
        for i, t in enumerate(g):
            x = self.features[t.id()]
            y.extend([i] * len(x))
            if len(X) == 0:
                X = np.array(x)
            else:
                X = np.vstack([X, np.array(x)])

        if use_xgboost:
            from xgboost import XGBClassifier
            rfc = XGBClassifier()
        else:
            rfc = RandomForestClassifier()

        rfc.fit(X, y)
        return rfc

    def _get_and_train_rfc_tracklets_groups(self, tracklets_groups, use_xgboost=False):
        X = []
        y = []
        for i, g in enumerate(tracklets_groups.values()):
            y_len = 0
            for t_id in g:
                x = self.features[t_id]
                y_len += len(x)

                if len(X) == 0:
                    X = np.array(x)
                else:
                    X = np.vstack([X, np.array(x)])

            y.extend([i] * y_len)

        if use_xgboost:
            from xgboost import XGBClassifier
            rfc = XGBClassifier()
        else:
            rfc = RandomForestClassifier()

        rfc.fit(X, y)
        return rfc

    def predict_permutation(self, rfc, g):
        from scipy import stats
        results = []
        for t in g:
            results.append(int(stats.mode(rfc.predict(self.features[t.id()]))[0]))

        return results

    def _test_stable_marriage(self, p1, p2):
        for i, val in enumerate(p1):
            if p2[val] != i:
                return False

        return True

    def _presort_css(self, cs1, cs2):
        # presort... so the same tracklets have the same indices..
        inter = set(cs1.keys()).intersection(cs2.keys())
        intersection = {}
        for root in inter:
            intersection[root] = cs1[root]

        cs1_ = {}
        cs2_ = {}

        for root_id in cs1.keys():
            if root_id in intersection:
                continue
            cs1_[root_id] = cs1[root_id]

        for root_id in cs2.keys():
            if root_id in intersection:
                continue
            cs2_[root_id] = cs2[root_id]

        return cs1_, cs2_, intersection

    def get_cs_pair_price(self, cs1, cs2, use_xgboost=False, links={}):
        # returns only unknown... mutual tracklets are excluded and returned in intersection list

        ### arrange tracklets into groups
        cs1_groups = {}
        for t in cs1:
            id_ = self._find_update_link(t.id(), links)

            if id_ in cs1_groups:
                cs1_groups[id_].append(t.id())
            else:
                cs1_groups[id_] = [t.id()]

        cs2_groups = {}
        for t in cs2:
            id_ = self._find_update_link(t.id(), links)

            if id_ in cs2_groups:
                cs2_groups[id_].append(t.id())
            else:
                cs2_groups[id_] = [t.id()]

        cs1, cs2, intersection = self._presort_css(cs1_groups, cs2_groups)
        if len(cs1) == 0:
            return [], np.inf

        if len(cs1) == 1:
            matching = [[cs1[cs1.keys()[0]], cs2[cs2.keys()[0]]]]

            print "\t 1on1", cs1.keys(), cs2.keys()

            return matching, 0

        md = self.p.solver_parameters.max_edge_distance_in_ant_length * self.p.stats.major_axis_median
        C = np.zeros((len(cs1), len(cs2)), dtype=np.float)

        # TODO: if cs2 is bigger, then do it the opposite way
        rfc1 = self._get_and_train_rfc_tracklets_groups(cs1, use_xgboost)
        for i, g in enumerate(cs1.values()):
            X = []
            for t_id in g:
                x = self.features[t_id]

                if len(X) == 0:
                    X = np.array(x)
                else:
                    X = np.vstack([X, np.array(x)])

            # TODO: idTracker like probs
            # for each ID, compute ID tracker probs
            res = rfc1.predict_proba(X)
            C[i, :] = np.mean(res, axis=0)

        # TODO: nearest tracklet (position) based on min over all t_distances
        for i, repr_id1 in enumerate(cs1.keys()):
            for j, repr_id2 in enumerate(cs2.keys()):
                # find best pair
                # TODO: can be optimized...
                best_ti = None
                best_tj = None
                best_d = np.inf

                for t_id_i in cs1[repr_id1]:
                    for t_id_j in cs2[repr_id2]:
                        ti = self.p.chm[t_id_i]
                        tj = self.p.chm[t_id_j]

                        ef = ti.end_frame(self.p.gm)
                        sf = tj.start_frame(self.p.gm)

                        # something smaller than np.inf to guarantee at least one best_ti, best_tj
                        d = 10000000
                        if ef < sf:
                            d = sf - ef

                        if d < best_d:
                            best_d = d
                            best_ti = ti
                            best_tj = tj

                t1 = best_ti
                t2 = best_tj

                # same track
                # let, probability is 1
                prob = 1
                if t1 != t2:
                    t1_end_f = t1.end_frame(self.p.gm)
                    t2_start_f = t2.start_frame(self.p.gm)

                    # not allowed, probability is zero
                    if t1_end_f >= t2_start_f:
                        prob = 0
                    else:
                        t1_end_r = self.p.gm.region(t1.end_node())
                        t2_start_r = self.p.gm.region(t2.start_node())

                        frame_d = t2_start_f - t1_end_f
                        # d = np.linalg.norm(t1_end_r.centroid() - t2_start_r.centroid()) / float(frame_d * md)
                        d = np.linalg.norm(t1_end_r.centroid() - t2_start_r.centroid()) / float(md)
                        prob = max(0, 1 - d)

                        # TODO: raise uncertainty with frame_d
                        prob -= (frame_d - 1) * 0.05




                # probability complement... something like cost
                # TODO: -log(P) ?
                C[i, j] = 1 - C[i, j] * prob

        # TODO: what to do with too short CS, this will stop aglomerattive clustering

        from scipy.optimize import linear_sum_assignment

        # use hungarian (nonnegative matrix)
        row_ind, col_ind = linear_sum_assignment(C)
        price = C[row_ind, col_ind].sum()
        price_norm = price / float(len(cs1))

        # TODO price - max
        # print price, price_norm, C[row_ind, col_ind].max()
        np.set_printoptions(precision=3)
        # print C

        matching = []
        for rid, cid in izip(row_ind, col_ind):
            matching.append([cs1[cs1.keys()[rid]], cs2[cs2.keys()[cid]]])

        return matching, price_norm

    def _find_update_link(self, t_id, links):
        id_ = t_id

        while id_ is not None:
            if id_ not in links:
                break

            id_ = links[id_]

        if t_id != id_:
            links[t_id] = id_

        return id_

    def full_csosit_analysis(self, use_xgboost=False):
        from tracklet_complete_set import TrackletCompleteSet

        groups = []
        unique_tracklets = set()
        frames = []

        vm = get_auto_video_manager(self.p)
        total_frame_count = vm.total_frame_count()
        expected_t_len = len(self.p.animals) * total_frame_count
        total_single_t_len = 0
        for t in self.p.chm.chunk_gen():
            if t.is_single():
                total_single_t_len += t.length()

        overlap_sum = 0
        frame = 0
        i = 0
        while True:
            group = self.p.chm.chunks_in_frame(frame)
            if len(group) == 0:
                break

            singles_group = filter(lambda x: x.is_single(), group)

            # if len(singles_group) == len(self.p.animals) and min([len(t) for t in singles_group]) >= self.min_tracklet_len:
            if len(singles_group) == len(self.p.animals) and min([len(t) for t in singles_group]) >= 1:
                groups.append(singles_group)

                overlap = min([t.end_frame(self.p.gm) for t in singles_group]) \
                          - max([t.start_frame(self.p.gm) for t in singles_group])

                overlap_sum += overlap

                frames.append((frame, overlap, ))
                for t in singles_group:
                    unique_tracklets.add(t)

            frame = min([t.end_frame(self.p.gm) for t in group]) + 1

            if i % 100:
                print_progress(frame, total_frame_count, "searching for CSoSIT...")

            i += 1
        print_progress(total_frame_count, total_frame_count, "searching for CSoSIT...")

        import matplotlib.pyplot as plt

        used = {}
        positions = {}

        print "# groups: {}".format(len(groups))
        g1 = groups[0]
        g2 = groups[1]

        print "G1"
        for t in g1:
            print t.start_frame(self.p.gm)
            print t.end_frame(self.p.gm)

        print "G2"
        for t in g2:
            print t.start_frame(self.p.gm)
            print t.end_frame(self.p.gm)

        # filter CS in the same frame:
        print "Filtering suspicious CS (single-ID false positive)"
        frames = {}
        # suspicious = set()
        for g in groups:
            frame = max([t.start_frame(self.p.gm) for t in g])
            if frame in frames:
                print frame
                # suspicious.add(frames[frame])
                # suspicious.add(g)
            else:
                frames[frame] = g

        print ""

        TCS = {}
        left_neighbour = None

        tcs_id = 0

        # key is most left CS
        results = []

        for i in range(len(groups) - 1):
            cs = TrackletCompleteSet(groups[i], id=tcs_id,left_neighbour=left_neighbour)

            if left_neighbour:
                left_neighbour.right_neighbour = cs

            TCS[tcs_id] = cs

            left_neighbour = cs

            matching, price = self.get_cs_pair_price(groups[i], groups[i + 1])
            results.append([price, matching, (tcs_id, tcs_id+1)])

            # don't forget to increment, so we have unique numbers...
            tcs_id += 1

        # so the best result is on the last index, ready for .pop()
        # results, idx = sorted((e, i) for i, e in enumerate(results))
        sorted_results = sorted(results, key=lambda x: -x[0])
        # results = sorted(results, key=lambda x: -x[0])

        links = {}
        invalid_TCS = set()


        # load gt...
        path = '/Users/flipajs/Documents/dev/ferda/data/GT/Cam1_.pkl'
        from utils.gt.gt import GT
        self.GT = GT()
        self.GT.load(path)
        self.GT.set_offset(y=self.p.video_crop_model['y1'],
                           x=self.p.video_crop_model['x1'],
                           frames=self.p.video_start_t)


        while len(sorted_results):
            price, matching, (tcs1_id, tcs2_id) = sorted_results.pop()

            tcs_left = TCS[tcs1_id]
            tcs_right = TCS[tcs2_id]

            if tcs_left in invalid_TCS:
                print "DROPING: ", tcs1_id
                continue

            if tcs_right in invalid_TCS:
                print "DROPING: ", tcs2_id
                continue

            if np.isinf(price):
                print "INFINITE price... Ending with {} CS left".format(len(TCS) - len(invalid_TCS))
                break

            if price > 0.6:
                print "Price is too big: {}, Ending with {} CS left".format(price, len(TCS) - len(invalid_TCS))
                break

            print price, tcs1_id, tcs2_id

            # check...
            for t_pair in matching:
                t0_id = t_pair[0][0]
                id0 = self.GT.tracklet_id_set_without_checks(self.p.chm[t0_id], self.p)
                for t1_id in t_pair[1]:
                    id1 = self.GT.tracklet_id_set_without_checks(self.p.chm[t1_id], self.p)

                    if id0 != id1:
                        print "GT doesn't agree", t0_id, t1_id

            new_group = list(TCS[tcs1_id].tracklets)

            for t_pair in matching:
                for t1_id in t_pair[1]:
                    links[t1_id] = t_pair[0][0]

                    t = self.p.chm[t1_id]
                    new_group.append(t)

            # nomenclature A -- tcs_left - tcs_right -- B (tcs_left merges with tcs_right into tcs)
            tcs_A = tcs_left.left_neighbour
            tcs_B = tcs_right.right_neighbour

            ### Create new TCS
            tcs = TrackletCompleteSet(new_group, tcs_id, left_neighbour=tcs_A, right_neighbour=tcs_B)
            TCS[tcs_id] = tcs
            # update global ID counter
            tcs_id += 1

            ################################################


            invalid_TCS.add(tcs_left)
            # update references
            if tcs_A:
                # invalidate outdated results...

                tcs_A.right_neighbour = tcs

                # compute new results:
                new_matching, new_price = self.get_cs_pair_price(tcs_A.tracklets, tcs.tracklets, links=links)
                for i, (price, _, _) in enumerate(sorted_results):
                    if new_price > price:
                        break

                print "New price", new_price, i
                sorted_results.insert(i, [new_price, new_matching, (tcs_A.id, tcs.id)])

            # invalidate outdated results...
            invalid_TCS.add(tcs_right)

            if tcs_B:
                tcs_B.left_neighbour = tcs

                new_matching, new_price = self.get_cs_pair_price(tcs.tracklets, tcs_B.tracklets, links=links)
                for i, (price, _, _) in enumerate(sorted_results):
                    if new_price > price:
                        break

                print "new price", new_price, i
                sorted_results.insert(i, [new_price, new_matching, (tcs.id, tcs_B.id)])



        for key in links.keys():
            links[key] = self._find_update_link(key, links)

        lk = links.values()
        print "#links: {}, #UNIQUE keys: {}, keys: {}".format(len(lk), len(set(lk)), set(lk))

        # TODO: add own support per root tracklets...

        # find biggest support:
        support = {}
        for t_id, t_root_id in links.iteritems():
            if t_root_id in support:
                support[t_root_id] += len(self.p.chm[t_id])
            else:
                support[t_root_id] = len(self.p.chm[t_root_id])

        best_tcs = None
        best_support = 0
        for tcs in TCS.itervalues():
            if tcs in invalid_TCS:
                continue

            # tcs_supp = 0
            # for t in tcs.tracklets:
            #     if t.id() in support:
            #         tcs_supp += support[t.id()]

            # max min
            tcs_supp = 0
            tcs_supports = {}

            for t in tcs.tracklets:
                t_root_id = self._find_update_link(t.id(), links)
                if t_root_id not in tcs_supports:
                    tcs_supports[t_root_id] = len(t)
                else:
                    tcs_supports[t_root_id] += len(t)

            tcs_supp = min(tcs_supports.values())
            # if t.id() in support:
            #     tcs_supp = max(tcs_supp, support[t.id()])

            if tcs_supp > best_support:
                best_tcs = tcs
                best_support = tcs_supp

        tid_set = set()
        for t in best_tcs.tracklets:
            tid_set.add(self._find_update_link(t.id(), links))


        print "#TCS: {}".format(len(TCS) - len(invalid_TCS))
        print "final TID set: ", tid_set, best_support
        IDs_mapping = {}
        IDs = list(tid_set)

        for id_, key in enumerate(IDs):
            IDs_mapping[key] = id_

        fullset = set(range(len(IDs)))

        self.user_decisions = []

        for t in self.p.chm.chunk_gen():
            t.P = set()
            t.N = set()

            if t.id() in links:
                root_id = links[t.id()]
                if root_id in IDs_mapping:
                    id_ = IDs_mapping[root_id]
                    t.P = set([id_])
                    t.N = fullset - t.P

                    self.user_decisions.append({'tracklet_id_set': t.id(), 'type': 'P', 'ids': [id_]})

            if t.id() in IDs_mapping:
                id_ = IDs_mapping[t.id()]
                t.P = set([id_])
                t.N = fullset - t.P

                self.user_decisions.append({'tracklet_id_set': t.id(), 'type': 'P', 'ids': [id_]})


        vals = TCS.values()
        for i, tcs in enumerate(vals):
            if tcs in invalid_TCS:
                continue

            subset = True
            for tcs2 in vals[i:]:
                for t in tcs.tracklets:
                    if t in tcs2.tracklets:
                        subset = False
                        break

                if subset:
                    print tcs.id, "is SUBSET"

            sum_length = 0

            min_frame = np.inf
            max_frame = 0

            for t in tcs.tracklets:
                sf = t.start_frame(self.p.gm)
                ef = t.end_frame(self.p.gm)

                min_frame = min(sf, min_frame)
                max_frame = max(ef, max_frame)

                sum_length += len(t)

            t_ids = [t.id() for t in tcs.tracklets]

            print "id: {}, sum: {}, min frame: {} max frame: {}\n\t#{} {}\n\n".format(tcs.id, sum_length, min_frame, max_frame, len(t_ids), t_ids)

        # Train RFC on biggest CS
        # self.train(init=True)

        self.tcs = TCS
        self.links = links

        print "num undecided before: ", len(self.undecided_tracklets)
        self.update_undecided_tracklets()
        print "num undecided after: ", len(self.undecided_tracklets)

        for cs in TCS.itervalues():
            # g = sorted(g, key=lambda x: x.id())
            c = np.random.rand(3, 1)

            frame = max([t.start_frame(self.p.gm) for t in cs.tracklets])
            plt.plot([frame, frame], [0, len(self.p.animals)], c=c)
            plt.hold(True)

            free_pos = range(len(self.p.animals))
            for i, t in enumerate(g):
                if t in used:
                    used[t] += 1
                    i = positions[t]

                    free_pos.remove(i)

            for t in cs.tracklets:
                if t not in used:
                    positions[t] = free_pos[0]
                    free_pos.pop(0)
                    used[t] = 1

                offset = (used[t]-1) / 10.
                pos = positions[t]

                # offset = 0
                plt.plot([t.start_frame(self.p.gm), t.end_frame(self.p.gm)], [pos+offset, pos+offset], c=c)

        plt.grid()
        plt.show()




        # # continue classifying only using ID classification
        #
        # for tcs in TCS.itervalues():
        #     if tcs in invalid_TCS or tcs == best_tcs:
        #         continue
        #
        #     score = tcs x best_tcs

        return

        ii = 0
        for g in groups:
            ii += 1
            if ii > 5:
                break

            g = sorted(g, key=lambda x: x.id())
            c = np.random.rand(3, 1)

            frame = max([t.start_frame(self.p.gm) for t in g])
            plt.plot([frame, frame], [0, len(self.p.animals)], c=c)
            plt.hold(True)

            free_pos = range(len(self.p.animals))
            for i, t in enumerate(g):
                if t in used:
                    used[t] += 1
                    i = positions[t]

                    free_pos.remove(i)

            for t in g:
                if t not in used:
                    positions[t] = free_pos[0]
                    free_pos.pop(0)
                    used[t] = 1

                offset = (used[t]-1) / 10.
                pos = positions[t]

                # offset = 0
                plt.plot([t.start_frame(self.p.gm), t.end_frame(self.p.gm)], [pos+offset, pos+offset], c=c)

        plt.grid()
        plt.show()


        best_g_i = -1
        best_g_val = 0
        for i, g in enumerate(groups):
            val = min(len(t) for t in g)
            if val > best_g_val:
                best_g_i = i
                best_g_val = val

        if best_g_i < 0:
            # TODO: manual init, documentation...
            print "No CSoSIT was found. This means, there is no frame satisfying that #single-ID tracklets = #animals and minimum of tracklet lengths > 'tracklet min len' parameter."
            print "This usually happens when test video with short length is used... You can still initialise manually"
            return

        tracklet_ids = {}
        for id_, t in enumerate(groups[best_g_i]):
            tracklet_ids[t.id()] = id_
            self.user_decisions.append({'tracklet_id_set': t.id(), 'type': 'P', 'ids': [id_]})

        max_best_frame = max(t.start_frame(self.p.gm) for t in groups[best_g_i])

        try:
            # path = '/Users/flipajs/Documents/dev/ferda/data/GT/Cam1_.pkl'
            path = '/Users/flipajs/Documents/dev/ferda/data/GT/rep1-cam2.pkl'
            from utils.gt.gt import GT
            self.GT = GT()
            self.GT.load(path)
            self.GT.set_offset(y=self.p.video_crop_model['y1'],
                                x=self.p.video_crop_model['x1'],
                                frames=self.p.video_start_t)

            permutation_data = []
            for id_, t in enumerate(self.p.chm.chunks_in_frame(max_best_frame)):
                if not t.is_single():
                    continue

                # id_ = list(t.P)[0]
                y, x = RegionChunk(t, self.p.gm, self.p.rm).centroid_in_t(max_best_frame)
                permutation_data.append((max_best_frame, id_, y, x))

            self.GT.set_permutation_reversed(permutation_data)

            print "GT permutation set in frame {}".format(max_best_frame)
        except IOError:
            pass

        from math import floor
        half = int(floor(len(groups)/2))

        ok_min_sum = min(len(t) for t in groups[best_g_i])
        ok_total_sum = sum(len(t) for t in groups[best_g_i])

        g1 = groups[best_g_i]
        rfc1 = self._get_and_train_rfc(g1, use_xgboost)
        from tqdm import trange
        for i in trange(len(groups)):
            if i == best_g_i:
                continue

            g2 = groups[i]
            rfc2 = self._get_and_train_rfc(g2, use_xgboost)

            perm1 = self.predict_permutation(rfc1, g2)
            perm2 = self.predict_permutation(rfc2, g1)

            if self._test_stable_marriage(perm1, perm2):
                for id_, t in izip(perm1, g2):
                    if t.id() in tracklet_ids:
                        if id_ != tracklet_ids[t.id()]:
                            warnings.warn("auto init ids doesn't match. T_ID: {} firstly assigned with: {} now: {}".format(t.id(), tracklet_ids[t.id()], id_))
                        else:
                            print t.id(), "duplicate..."

                        continue
                    else:
                        tracklet_ids[t.id()] = id_

                    self.user_decisions.append({'tracklet_id_set': t.id(), 'type': 'P', 'ids': [id_]})
                    # self.assign_identity(id_, t)
                    try:
                        ids = self.GT.tracklet_id_set_without_checks(t, self.p)
                        if ids[0] != id_:
                            print "GT differs ", id_, ids[0], t.id()
                    except:
                        pass

                ok_min_sum += min(len(t) for t in g2)
                ok_total_sum += sum(len(t) for t in g2)
                print "OK", min([len(t) for t in g1]), min([len(t) for t in g2])
            # else:
                # print "fail", min([len(t) for t in g1]), min([len(t) for t in g2])
                # for i, val in enumerate(perm1):
                #     if i != perm2[val]:
                #         print "\t", i, perm2[val]

        print "OK min sum: {} total sum: {}".format(ok_min_sum, ok_total_sum)

        total_len = 0
        for t in unique_tracklets:
            total_len += t.length()

        print "STATS:"
        print "\tnum groups: {}\n\ttotal length: {}/expected: {:.2%}/singles: {:.2%}\n\toverlap sum: {}/{:.2%}".format(
            len(groups),
            total_len, total_len/float(expected_t_len), total_len/float(total_single_t_len),
            overlap_sum, overlap_sum/float(total_frame_count))

    def auto_init(self, method='max_sum', use_xgboost=False):
        # self.full_csosit_analysis(use_xgboost=use_xgboost)
        return
        from multiprocessing import cpu_count

        best_frame = None
        best_score = 0

        max_best_frame = None
        max_best_score = 0

        vm = get_auto_video_manager(self.p)
        total_frame_count = vm.total_frame_count()

        frame = 0
        while True:
            group = self.p.chm.chunks_in_frame(frame)
            if len(group) == 0:
                break

            singles_group = filter(lambda x: x.is_single(), group)

            if len(singles_group) == len(self.p.animals):
                m = min([t.length() for t in singles_group])
                mm = sum([t.length() for t in singles_group])
                if m > best_score:
                    best_frame = frame
                    best_score = m

                if mm > max_best_score:
                    max_best_frame = frame
                    max_best_score = mm

            new_frame = min([t.end_frame(self.p.gm) for t in group]) + 1
            # speedup "hack". In extreme cases might be slightly suboptimal
            frame = max(new_frame, frame+30)

            print_progress(frame, total_frame_count, "searching for best initialisation frame using "+method.upper()+" method")

        if method == 'maxsum':
            self.user_decisions = []
            self.separated_frame = max_best_frame
            group = self.p.chm.chunks_in_frame(max_best_frame)
        else:
            group = self.p.chm.chunks_in_frame(best_frame)
            max_best_frame = best_frame

        group = filter(lambda x: x.is_single(), group)

        for id_, t in enumerate(group):
            print "id: {}, len: {}".format(t.id(), t.length())
            self.user_decisions.append({'tracklet_id_set': t.id(), 'type': 'P', 'ids': [id_]})

        try:
            path = '/Users/flipajs/Documents/dev/ferda/data/GT/Cam1_.pkl'
            from utils.gt.gt import GT
            self.GT = GT()
            self.GT.load(path)
            self.GT.set_offset(y=self.p.video_crop_model['y1'],
                                x=self.p.video_crop_model['x1'],
                                frames=self.p.video_start_t)

            permutation_data = []
            for t in self.p.chm.chunks_in_frame(max_best_frame):
                if not t.is_single():
                    continue

                id_ = list(t.P)[0]
                y, x = RegionChunk(t, self.p.gm, self.p.rm).centroid_in_t(max_best_frame)
                permutation_data.append((max_best_frame, id_, y, x))

            self.GT.set_permutation_reversed(permutation_data)

            print "GT permutation set"
        except IOError:
            pass

        return max_best_frame

    def id_with_least_examples(self):
        m = np.inf
        mi = -1
        for i in range(len(self.p.animals)):
            x = np.sum(np.array(self.y) == i)
            if x < m:
                m = x
                mi = i

        return mi

    def question_to_increase_smallest(self, gt):
        id_ = self.id_with_least_examples()

        used = set()
        for it in self.user_decisions:
            used.add(it['tracklet_id_set'])

        best_len = 0
        best_id_ = -1
        for t in self.p.chm.chunk_gen():
            if t not in used:
                _, val = gt.get_class_and_id(t, self.p)
                # val = self.__DEBUG_get_answer_from_GT(t)

                if len(val) == 1 and val[0] == id_:
                    if t.length() > best_len:
                        best_len = t.length()
                        best_id_ = t.id()

        if best_id_ == -1:
            best_id_ = self.get_best_question()

        return self.p.chm[best_id_]

    def question_near_assigned(self, tracklet_gt, min_samples=500, max_frame_d=100):
        y = [0] * len(self.p.animals)
        id2tid = [[] for i in range(len(self.p.animals))]
        for i in range(len(self.user_decisions)):
            d = self.user_decisions[i]
            t_id = d['tracklet_id_set']
            id_ = d['ids'][0]
            # id_ = list(tracklet_gt[t_id])[0]
            y[id_] += self.p.chm[t_id].length()
            id2tid[id_].append(t_id)

        print y

        id_least = -1
        if min(y) < min_samples:
            id_least = np.argmin(y)
        else:
            return None

        possibilities = []
        for tid, id_ in tracklet_gt.iteritems():
            if len(id_) == 1:
                id_ = list(id_)[0]
            else:
                continue

            if tid is None:
                continue

            if id_ == id_least:
                possibilities.append(tid)

        best_t_id_ = -1
        len_ = 0
        for tid in id2tid[id_least]:
            start = self.p.chm[tid].start_frame(self.p.gm)
            end = self.p.chm[tid].end_frame(self.p.gm)

            for tid2 in possibilities:
                if tid2 in id2tid[id_least]:
                    continue

                try:
                    t2 = self.p.chm[tid2]
                    if t2.length() < len_:
                        continue
                except:
                    print "tid2: ", tid2

                t = self.p.chm[tid2]

                if 0 <= t.start_frame(self.p.gm) - end <= max_frame_d:
                    best_t_id_ = t.id()
                    len_ = t2.length()
                elif 0 <=start - t.end_frame(self.p.gm) <= max_frame_d:
                    best_t_id_ = t.id()
                    len_ = t2.length()

        len_ = 0
        if best_t_id_ == -1:
            for tid2 in possibilities:
                if tid2 in id2tid[id_least]:
                    continue

                l_ = self.p.chm[tid].length()
                if l_ > len_:
                    best_t_id_ = tid2
                    len_ = l_
            print "NOT FOUND WITHIN 100 frames, choosing ", best_t_id_

        self.user_decisions.append({'tracklet_id_set': best_t_id_, 'type': 'P', 'ids': [id_least]})
        # print "HIL INIT, adding t_id: {}, len: {}".format(best_t_id_, len_)

        return best_t_id_


def compute_features_process(counter, lock, q_tasks, project_wd, num_frames, first_time=True):
    print "starting..."
    from core.project.project import Project
    from core.id_detection.feature_manager import FeatureManager
    project = Project()
    project.load(project_wd, lightweight=True)

    project.img_manager.max_num_of_instances = 50

    fm = FeatureManager(project_wd, db_name='fm.sqlite3')
    while True:
        if q_tasks.empty():
            time.sleep(0.1)
            continue

        task = q_tasks.get()
        if task is None:
            break

        frame_start, frame_end = task

        for frame in range(frame_start, frame_end):
            for t in project.chm.chunks_in_frame(frame):
                if not t.is_single():
                    continue

                rm = RegionChunk(t, project.gm, project.rm)
                r = rm.region_in_t(frame)

                compute = True
                if not first_time:
                    compute = fm[r.id()][1] == [None]

                if compute:
                    f = get_colornames_hists(r, project, saturated=True, lvls=1)
                    lock.acquire()
                    fm.add(r.id(), f)
                    lock.release()

            counter.increment()
            print_progress(counter.value(), num_frames, "features computation in progress:")

    q_tasks.put(None)

if __name__ == '__main__':
    wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Zebrafish_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'


    p = Project()
    p.load_semistate(wd, state='eps_edge_filter')
    p.img_manager = ImgManager(p)

    learn_proc = LearningProcess(p)