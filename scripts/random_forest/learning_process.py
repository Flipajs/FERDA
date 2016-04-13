from sklearn.ensemble import RandomForestClassifier
from core.project.project import Project
from core.graph.region_chunk import RegionChunk
from skimage.measure import moments_central, moments_hu, moments_normalized, moments
from utils.img_manager import ImgManager
import cv2
from utils.img import get_img_around_pts, replace_everything_but_pts
import cPickle as pickle
import numpy as np
from intervaltree import IntervalTree
from gui.learning.ids_names_widget import IdsNamesWidget
from PyQt4 import QtGui
import sys

class LearningProcess:
    def __init__(self, p):
        self.p = p

        self._eps1 = 0.01
        self._eps2 = 0.1

        self.get_features = self.get_features_var1
        self.animal_id_mapping = {}

        self.old_x_size = 0

        self.collision_chunks = {}

        self.chunk_available_ids = {}

        if False:
            self.p.img_manager = ImgManager(self.p, max_num_of_instances=700)
            self.chunks = self.get_candidate_chunks()

            self.chunks_itree = self.build_itree_()
            self.class_frequences = []

            # basically set every chunk with full set of possible ids
            self.__precompute_availability()
            self.features = self.precompute_features_()

            self.undecided_chunks = {}
            for ch in self.chunks:
                if ch.id_ in self.collision_chunks:
                    continue

                self.undecided_chunks[ch.id_] = True

            self.X = None
            self.y = None
            self.ids = self.get_init_data()

            self.rfc = RandomForestClassifier(class_weight='balanced')
            self.rfc.fit(self.X, self.y)

            with open(p.working_directory+'/temp/rfc.pkl', 'wb') as f:
                d = {'rfc': self.rfc, 'X': self.X, 'y': self.y, 'ids': self.ids,
                     'chunks': self.chunks, 'class_frequences': self.class_frequences,
                     'animal_id_mapping': self.animal_id_mapping, 'chunks_itree': self.chunks_itree,
                     'features': self.features, 'chunk_available_ids': self.chunk_available_ids,
                     'collision_chunks': self.collision_chunks, 'undecided_chunks': self.undecided_chunks,
                     'old_x_size': self.old_x_size}
                pickle.dump(d, f, -1)
        else:
            with open(p.working_directory+'/temp/rfc.pkl', 'rb') as f:
                d = pickle.load(f)
                self.rfc = d['rfc']
                self.X = d['X']
                self.y = d['y']
                self.ids = d['ids']
                self.chunks = d['chunks']
                self.class_frequences = d['class_frequences']
                self.animal_id_mapping = d['animal_id_mapping']
                self.chunks_itree = d['chunks_itree']
                self.features = d['features']
                self.chunk_available_ids = d['chunk_available_ids']
                self.collision_chunks = d['collision_chunks']
                self.undecided_chunks = d['undecided_chunks']
                self.old_x_size = d['old_x_size']

        # for ch in self.chunks:
        #     rch = RegionChunk(ch, self.p.gm, self.p.rm)
        #
        #     if ch.length() > 0:
        #         sum = 0
        #         for r in rch.regions_gen():
        #             sum += r.area()
        #
        #         area_mean = sum/float(ch.length())
        #         c = 'C' if ch.id_ in self.collision_chunks else ' '
        #         area_mean_thr = 1000
        #         p = 'C' if area_mean > area_mean_thr else ' '
        #         print "%s %s %s area: %.2f, id:%d, length:%d" % (p==c, c, p, area_mean, ch.id_, ch.length())
        #
        #         if area_mean > area_mean_thr:
        #             self.collision_chunks[ch.id_] = True
        #             try:
        #                 del self.undecided_chunks[ch.id_]
        #             except:
        #                 pass

        self.next_step()



    def set_ids_(self):
        app = QtGui.QApplication(sys.argv)
        ex = IdsNamesWidget()
        ex.show()

        app.exec_()
        app.deleteLater()

    def precompute_features_(self):
        features = {}
        i = 0
        for ch in self.chunks:
            if ch in self.collision_chunks:
                continue

            # if i > 20:
            #     break
            X = self.get_data(ch)

            i += 1
            features[ch.id_] = X

            print i

        return features

    def build_itree_(self):
        itree = IntervalTree()
        for ch in self.chunks:
            itree.addi(ch.start_frame(self.p.gm) - self._eps1, ch.end_frame(self.p.gm) + self._eps1, ch.id_)

        return itree

    def get_candidate_chunks(self):
        ch_list = self.p.chm.chunk_list()

        print "ALL CHUNKS:", len(ch_list)
        filtered = []
        for ch in ch_list:
            # if ch.start_frame(self.p.gm) > 500:
            #     continue
            # else:
            filtered.append(ch)

            if ch.length() > 0:
                ch_start_vertex = self.p.gm.g.vertex(ch.start_node())

                # ignore chunks of merged regions
                # is_merged = False
                for e in ch_start_vertex.in_edges():
                    if self.p.gm.g.ep['score'][e] == 0 and ch_start_vertex.in_degree() > 1:
                        # is_merged = True
                        self.collision_chunks[ch.id_] = True
                        break

                rch = RegionChunk(ch, self.p.gm, self.p.rm)

                if ch.length() > 0:
                    sum = 0
                    for r in rch.regions_gen():
                        sum += r.area()

                    area_mean = sum/float(ch.length())
                    area_mean_thr = 1000

                    area_mean = sum/float(ch.length())
                    c = 'C' if ch.id_ in self.collision_chunks else ' '
                    area_mean_thr = 1000
                    p = 'C' if area_mean > area_mean_thr else ' '
                    print "%s %s %s area: %.2f, id:%d, length:%d" % (p==c, c, p, area_mean, ch.id_, ch.length())

                    if area_mean > area_mean_thr:
                        self.collision_chunks[ch.id_] = True


                # if not is_merged:
                #     filtered.append(ch)

        # print "FILTERED: ", len(filtered)
        #
        # filtered = sorted(filtered, key=lambda x: x.start_frame(self.p.gm))
        # return filtered

        return filtered

    def __solve_if_clear(self, vertices1, vertices2):
        score = np.zeros((len(vertices1), len(vertices2)))

        for i, v1 in enumerate(vertices1):
            for j, v2 in enumerate(vertices2):
                s, _, _, _ = self.p.solver.assignment_score(self.p.gm.region(v1), self.p.gm.region(v2))
                # s, _, _ = self.p.solver.assignment_score_pos_orient(self.p.gm.region(v1), self.p.gm.region(v2))

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

                id1 = self.animal_id_mapping.get(ch1.id_, -1)
                id2 = self.animal_id_mapping.get(ch2.id_, -1)

                if id1 > -1 and id2 > -1 and id1 != id2:
                    assign_new_ids = []
                    break

                if id1 == id2:
                    continue

                if id1 > -1:
                    assign_new_ids.append((id1, ch2))
                elif id2 > -1:
                    assign_new_ids.append((id2, ch1))

        for id_, ch in assign_new_ids:
            if ch.id_ in self.features:
                self.__learn(ch, id_)

            self.__assign_id(ch, id_)

        return len(assign_new_ids) > 0

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

        # if e_vertext.out_degree() == 1:
        #     for v in e_vertext.out_neighbours():
        #         pass
        #

        return False

    def __learn(self, ch, id_):
        print "LEARNING ", id_
        X = self.features[ch.id_]
        self.X = np.vstack([self.X, np.array(X)])

        y = [id_] * len(X)
        self.y = np.append(self.y, np.array(y))

        self.class_frequences[id_] += len(X)

        if len(self.X) - self.old_x_size > 50:
            self.rfc = RandomForestClassifier(class_weight='balanced')
            self.rfc.fit(self.X, self.y)

            self.old_x_size = len(self.X)

    def __assign_id(self, ch, id_):
        # for i in xrange(len(self.candidate_chunks)):
        #     if self.candidate_chunks[i].id_ == ch.id_:
        #         self.candidate_chunks.pop(i)
        #         break

        # for i, ch_ in enumerate(self.chunks):
        #     if ch.id_ == ch_.id_:
        #         self.chunks.pop(i)
        #
        #         print "pop ", i
        # try:
        #     self.candidate_chunks.remove(ch)
        # except:
        #
        #     pass

        try:
            del self.undecided_chunks[ch.id_]
        except:
            print "PROBLEMATIC CHUNK", ch.id_,  ch.start_frame(self.p.gm), ch.end_frame(self.p.gm), ch, "AID: ", id_

        self.animal_id_mapping[ch.id_] = id_

    def __precompute_availability(self):
        vertices = map(self.p.gm.g.vertex, self.p.gm.get_vertices_in_t(0))

        ids = range(len(vertices))

        for ch in self.chunks:
            self.chunk_available_ids[ch.id_] = list(ids)

    def __propagate_availability(self, ch, remove_id=[]):
        if ch not in self.chunks:
            return []


        S_in = set()
        affected = []
        for u in ch.start_vertex(self.p.gm).in_neighbours():
            ch_, _ = self.p.gm.is_chunk(u)
            if ch_ in self.chunks:
                affected.append(ch_)
                S_in.update(self.chunk_available_ids[ch_.id_])

        S_out = set()
        for u in ch.end_vertex(self.p.gm).out_neighbours():
            ch_, _ = self.p.gm.is_chunk(u)
            if ch_ in self.chunks:
                affected.append(ch_)
                S_out.update(self.chunk_available_ids[ch_.id_])

        S_self = set(self.chunk_available_ids[ch.id_])

        new_S_self = S_self.intersection(S_in.union(S_out))
        for id_ in remove_id:
            new_S_self.discard(id_)

        if S_self == new_S_self:
            return []

        if not new_S_self:
            print "ZERO available IDs set", ch.id_,  ch.start_frame(self.p.gm), ch.end_frame(self.p.gm), ch

        new_S_self = list(new_S_self)
        if len(new_S_self) == 1:
            print "Chunk solved by ID conservation rules", ch.id_,  ch.start_frame(self.p.gm), ch.end_frame(self.p.gm), ch, "AID: ", new_S_self[0]
            self.update_availability(ch, new_S_self[0], learn=True)
        else:
            self.chunk_available_ids[ch.id_] = new_S_self

        return affected

    def update_availability(self, ch, id_, learn=False):
        if learn:
            self.__learn(ch, id_)

        self.__assign_id(ch, id_)
        self.chunk_available_ids[ch.id_] = [id_]

        # processed = set()
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
            queue.extend(self.__propagate_availability(ch))
            # processed.add(v)

    def next_step(self):
        k = 50.0

        while len(self.undecided_chunks):
            best_ch = None
            best_val = 0
            for ch_id_ in self.undecided_chunks:
                ch = self.p.chm[ch_id_]

                if self.test_connected_with_merged(ch):
                    break

                proba, data_len = self.get_chunk_proba(ch)

                uni_probs = np.ones((len(proba), )) / float(len(proba))
                alpha = (min((data_len/k)**2, 0.95))

                # if it is obvious (1.0, 0, 0, 0, 0)...
                if np.max(proba) < 1.0:
                    proba = (1-alpha) * uni_probs + alpha*proba

                if np.max(proba) > best_val:
                    best_val = np.max(proba)
                    best_ch = ch

                    # print "prob: %.2f, ch_len: %d, id: %d, ch_id: %d, %s, ch_start: %d, ch_end: %d" %  (np.max(proba), data_len, np.argmax(proba), ch.id_, proba, ch.start_frame(self.p.gm), ch.end_frame(self.p.gm))
                    # self.classify_chunk(ch, proba)

            # if best_val < 0.5:
            #     break

            ch = best_ch

            if best_ch is None:
                continue

            proba, data_len = self.get_chunk_proba(ch)

            uni_probs = np.ones((len(proba), )) / float(len(proba))
            alpha = (min((data_len/k)**2, 0.95))

            if np.max(proba) < 1.0:
                proba = (1-alpha) * uni_probs + alpha*proba

            print "prob: %.2f, ch_len: %d, id: %d, ch_id: %d, %s, ch_start: %d, ch_end: %d" % (np.max(proba), data_len, np.argmax(proba), ch.id_, proba, ch.start_frame(self.p.gm), ch.end_frame(self.p.gm))
            print "-----------------------------------------------", len(self.undecided_chunks)

            animal_id = np.argmax(proba)

            use_for_learning = True if np.max(proba) > 0.9 else False

            self.update_availability(ch, animal_id, learn=use_for_learning)

        for i in range(6):
            print i, np.sum(self.y == i)

        with open(self.p.working_directory + '/temp/animal_id_mapping.pkl', 'wb') as f_:
            pickle.dump(self.animal_id_mapping, f_)

    def get_frequence_vector_(self):
        return float(np.sum(self.class_frequences)) / self.class_frequences

    def get_chunk_proba(self, ch):
        X = self.features[ch.id_]
        # X = self.get_data(ch)
        if len(X) == 0:
            return None, 0

        probs = self.rfc.predict_proba(np.array(X))
        probs = np.mean(probs, 0)

        probs *= self.get_frequence_vector_()

        probs = self.apply_consistency_rule(ch, probs)

        # normalise
        if np.sum(probs) > 0:
            probs /= float(np.sum(probs))

        return probs, len(X)

    def classify_chunk(self, ch, proba):
        pass

    def recompute_rfc(self):
        pass

    def apply_consistency_rule(self, ch, probs):
        mask = np.zeros(probs.shape)
        for id_ in self.chunk_available_ids[ch.id_]:
            mask[id_] = 1

        probs *= mask

        # start_f = ch.start_frame(self.p.gm)
        # end_f = ch.end_frame(self.p.gm)
        # intervals = self.chunks_itree[start_f-self._eps2:end_f + self._eps2]
        #
        # for i in intervals:
        #     if i.data in self.animal_id_mapping:
        #         animal_id = self.animal_id_mapping[i.data]
        #         probs[animal_id] = 0

        return probs

    def get_features_var1(self, r, p):
        f = []
        # area
        f.append(r.area())

        # # area, modifications
        # f.append(r.area()**0.5)
        # f.append(r.area()**2)
        #
        # contour length
        f.append(len(r.contour()))

        # major axis
        f.append(r.a_)

        # minor axis
        f.append(r.b_)

        # axis ratio
        f.append(r.a_ / r.b_)

        # axis ratio sqrt
        f.append((r.a_ / r.b_)**0.5)

        # axis ratio to power of 2
        f.append((r.a_ / r.b_)**2.0)

        img = p.img_manager.get_whole_img(r.frame_)
        crop, offset = get_img_around_pts(img, r.pts())

        pts_ = r.pts() - offset

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)

        ###### MOMENTS #####
        # #### BINARY
        crop_b_mask = replace_everything_but_pts(np.ones(crop_gray.shape, dtype=np.uint8), pts_)
        f.extend(self.get_hu_moments(crop_b_mask))


        #### ONLY MSER PXs
        # in GRAY
        crop_gray_masked = replace_everything_but_pts(crop_gray, pts_)
        f.extend(self.get_hu_moments(crop_gray_masked))

        # B G R
        for i in range(3):
            crop_ith_channel_masked = replace_everything_but_pts(crop[:, :, i], pts_)
            f.extend(self.get_hu_moments(crop_ith_channel_masked))

        # min, max from moments head/tail
        import math
        from utils.img import rotate_img, centered_crop, get_bounding_box, endpoint_rot
        relative_border = 2.0

        bb, offset = get_bounding_box(r, p, relative_border)
        p_ = np.array([r.a_*math.sin(-r.theta_), r.a_*math.cos(-r.theta_)])
        endpoint1 = np.ceil(r.centroid() + p_) + np.array([1, 1])
        endpoint2 = np.ceil(r.centroid() - p_) - np.array([1, 1])

        bb = rotate_img(bb, r.theta_)
        bb = centered_crop(bb, 8*r.b_, 4*r.a_)

        c_ = endpoint_rot(bb, r.centroid(), -r.theta_, r.centroid())

        endpoint1_ = endpoint_rot(bb, endpoint1, -r.theta_, r.centroid())
        endpoint2_ = endpoint_rot(bb, endpoint2, -r.theta_, r.centroid())
        if endpoint1_[1] > endpoint2_[1]:
            endpoint1_, endpoint2_ = endpoint2_, endpoint1_

        y_ = int(c_[0] - r.b_)
        y2_ = int(c_[0]+r.b_)
        x_ = int(c_[1] - r.a_)
        x2_ = int(c_[1] + r.a_)
        im1_ = bb[y_:y2_, x_:int(c_[1]), :].copy()
        im2_ = bb[y_:y2_, int(c_[1]):x2_, :].copy()

        # ### ALL PXs in crop image given margin
        # crop, offset = get_img_around_pts(img, r.pts(), margin=0.3)
        #
        # # in GRAY
        # crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # f.extend(self.get_hu_moments(crop_gray))
        #
        # B G R
        for i in range(3):
            hu1 = self.get_hu_moments(im1_[:, :, i])
            hu2 = self.get_hu_moments(im2_[:, :, i])

            f.extend(list(np.min(np.vstack([hu1, hu2]), axis=0)))
            f.extend(list(np.max(np.vstack([hu1, hu2]), axis=0)))

        return f


        crop_ = np.asarray(crop, dtype=np.int32)

        # # R G combination
        # crop_rg = crop_[:, :, 1] + crop_[:, :, 2]
        # f.extend(self.get_hu_moments(crop_rg))
        #
        # # B G
        # crop_bg = crop_[:, :, 0] + crop_[:, :, 1]
        # f.extend(self.get_hu_moments(crop_bg))
        #
        # # B R
        # crop_br = crop_[:, :, 0] + crop_[:, :, 2]
        # f.extend(self.get_hu_moments(crop_br))


    def get_hu_moments(self, img):
        m = moments(img)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]

        mu = moments_central(img, cr, cc)
        nu = moments_normalized(mu)
        hu = moments_hu(nu)

        features = [m_ for m_ in hu]

        return features

    def get_data(self, ch):
        X = []
        r_ch = RegionChunk(ch, p.gm, p.rm)
        i = 0
        for r in r_ch.regions_gen():
            if not r.is_virtual:
                f_ = self.get_features(r, p)
                X.append(f_)

                i += 1

        return X

    def get_init_data(self):
        vertices = p.gm.get_vertices_in_t(0)

        chunks = []
        for v in vertices:
            chunks.append(p.chm[p.gm.g.vp['chunk_start_id'][p.gm.g.vertex(v)]])

        X = []
        y = []

        id_ = 0
        for ch in chunks:
            ch_data = self.get_data(ch)
            X.extend(ch_data)

            self.class_frequences.append(len(ch_data))

            y.extend([id_] * len(ch_data))

            id_ += 1

        self.class_frequences = np.array(self.class_frequences)

        self.X = np.array(X)
        self.y = np.array(y)

        # it is in this position, because we need self.X, self.y to be ready for the case when we solve something by conservation rules -> thus we will be learning -> updating self.X...
        # skip last chunk, because it will be assigned autoamtically based on conservation rules
        for id_, ch in enumerate(chunks[:-1]):
            self.update_availability(ch, id_)

        return range(len(chunks))

if __name__ == '__main__':
    p = Project()
    p.load('/Users/flipajs/Documents/wd/GT/Cam2/cam2.fproj')
    p.img_manager = ImgManager(p)

    learn_proc = LearningProcess(p)

