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

        self.id_names_pd = None
        self.old_x_size = 0

        if False:
            self.p.img_manager = ImgManager(self.p, max_num_of_instances=700)
            self.candidate_chunks = self.get_candidate_chunks()

            self.chunks_itree = self.build_itree_()
            self.class_frequences = []

            self.X, self.y, self.ids = self.get_init_data()

            self.rfc = RandomForestClassifier(class_weight='balanced')
            self.rfc.fit(self.X, self.y)

            self.features = self.precompute_features_()

            with open(p.working_directory+'/temp/rfc.pkl', 'wb') as f:
                d = {'rfc': self.rfc, 'X': self.X, 'y': self.y, 'ids': self.ids,
                     'candidate_chunks': self.candidate_chunks, 'class_frequences': self.class_frequences,
                     'animal_id_mapping': self.animal_id_mapping, 'chunks_itree': self.chunks_itree,
                     'features': self.features}
                pickle.dump(d, f, -1)
        else:
            with open(p.working_directory+'/temp/rfc.pkl', 'rb') as f:
                d = pickle.load(f)
                self.rfc = d['rfc']
                self.X = d['X']
                self.y = d['y']
                self.ids = d['ids']
                self.candidate_chunks = d['candidate_chunks']
                self.class_frequences = d['class_frequences']
                self.animal_id_mapping = d['animal_id_mapping']
                self.chunks_itree = d['chunks_itree']
                self.features = d['features']

        # self.rfc = RandomForestClassifier()
        # self.rfc.fit(self.X, self.y)

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
        for ch in self.candidate_chunks:
            # if i > 20:
            #     break
            X = self.get_data(ch)

            i += 1
            features[ch.id_] = X

            print i

        return features

    def build_itree_(self):
        itree = IntervalTree()
        for ch in self.candidate_chunks:
            itree.addi(ch.start_frame(self.p.gm) - self._eps1, ch.end_frame(self.p.gm) + self._eps1, ch.id_)

        return itree

    def get_candidate_chunks(self):
        ch_list = self.p.chm.chunk_list()

        print "ALL CHUNKS:", len(ch_list)
        filtered = []
        for ch in ch_list:
            if ch.length() > 0:
                ch_start_vertex = self.p.gm.g.vertex(ch.start_node())

                # ignore chunks of merged regions
                is_merged = False
                for e in ch_start_vertex.in_edges():
                    if self.p.gm.g.ep['score'][e] == 0 and ch_start_vertex.in_degree() > 1:
                        is_merged = True
                        break

                if not is_merged:
                    filtered.append(ch)

        print "FILTERED: ", len(filtered)

        filtered = sorted(filtered, key=lambda x: x.start_frame(self.p.gm))
        return filtered

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

        for i, ch_ in enumerate(self.candidate_chunks):
            if ch.id_ == ch_.id_:
                self.candidate_chunks.pop(i)

                print "pop ", i
        # try:
        #     self.candidate_chunks.remove(ch)
        # except:
        #
        #     pass

        self.animal_id_mapping[ch.id_] = id_

    def next_step(self):
        for i in range(len(self.candidate_chunks)):
            if not self.candidate_chunks:
                break

            k = 50.0

            best_ch = None
            best_val = 0
            for ch in self.candidate_chunks:
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
            print "-----------------------------------------------", len(self.candidate_chunks)

            animal_id = np.argmax(proba)

            # use it for learning
            if np.max(proba) > 0.9:
                self.__learn(ch, animal_id)

            self.__assign_id(ch, animal_id)

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
        start_f = ch.start_frame(self.p.gm)
        end_f = ch.end_frame(self.p.gm)
        intervals = self.chunks_itree[start_f-self._eps2:end_f + self._eps2]

        for i in intervals:
            if i.data in self.animal_id_mapping:
                animal_id = self.animal_id_mapping[i.data]
                probs[animal_id] = 0

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
            self.candidate_chunks.remove(ch)

            self.animal_id_mapping[ch.id_] = id_
            ch_data = self.get_data(ch)
            X.extend(ch_data)

            self.class_frequences.append(len(ch_data))

            y.extend([id_] * len(ch_data))

            id_ += 1

        self.class_frequences = np.array(self.class_frequences)

        return np.array(X), np.array(y), range(id_-1)

if __name__ == '__main__':
    p = Project()
    p.load('/Users/flipajs/Documents/wd/GT/Cam2/cam2.fproj')
    p.img_manager = ImgManager(p)

    learn_proc = LearningProcess(p)

