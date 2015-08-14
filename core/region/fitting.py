__author__ = 'fnaiser'

from core.region.distance_map import DistanceMap
import numpy as np
from utils.drawing.points import draw_points
from utils.video_manager import get_auto_video_manager
import cv2
import pickle
from core.region.region import Region
import matplotlib.pyplot as plt
from utils.geometry import rotation_matrix, angle_from_matrix, rotate
from utils.roi import ROI

##############
# all point lists are in format [y, x]
###

class TransHelper():
    def __init__(self, animal):
        self.centroid = np.copy(animal.centroid())
        self.global_translation = np.array([0., 0.])
        self.global_rotation = np.identity(2)
        self.angle = 0


class Fitting():
    def __init__(self, region, animals, num_of_iterations=10, use_settled_heuristics=True):
        self.region = region
        self.animals = animals
        self.num_of_iterations = num_of_iterations
        self.use_settled_heuristics = use_settled_heuristics

        self.d_map_region = DistanceMap(region.pts())
        self.d_map_animals = [DistanceMap(a.pts()) for a in animals]

        self.trans_helpers = [TransHelper(a) for a in animals]

        self.animal_history = [[] for a in animals]

    def fit(self):
        # self.plot_situation()
        if self.use_settled_heuristics:
            settled_ids = self.get_settled_ids()

            process_flag = np.ones((len(self.animals)), dtype=np.bool)
            process_flag[settled_ids] = False

            for i in range(self.num_of_iterations/2):
                # print i
                list_of_pairs = self.get_pairs_a2r_region()

                changed = False
                for a_id in range(len(self.animals)):
                    if process_flag[a_id]:
                        changed = True
                        a2r_pairs = self.get_pairs_a2r_a_outside_r(self.d_map_animals[a_id], self.d_map_region)
                        a2r_pairs.extend(list_of_pairs[a_id])

                        a2r_pairs = np.array(a2r_pairs)

                        t, R = self.compute_transformation(np.copy(a2r_pairs))
                        # print t, R
                        self.animal_history[a_id].append({'t': t, 'R': R})
                        self.apply_transform(a_id, t, R)

                        if self.test_t_convergence(self.animal_history[a_id], i):
                            process_flag[a_id] = False

                        # self.plot_situation()

                if not changed:
                    break

        process_flag = np.ones((len(self.animals)), dtype=np.bool)
        for i in range(self.num_of_iterations):
            # print i
            list_of_pairs = self.get_pairs_a2r_region()

            changed = False
            for a_id in range(len(self.animals)):
                if process_flag[a_id]:
                    changed = True
                    a2r_pairs = self.get_pairs_a2r_a_outside_r(self.d_map_animals[a_id], self.d_map_region)
                    a2r_pairs.extend(list_of_pairs[a_id])

                    a2r_pairs = np.array(a2r_pairs)

                    t, R = self.compute_transformation(np.copy(a2r_pairs))
                    # print t, R
                    self.animal_history[a_id].append({'t': t, 'R': R})
                    self.apply_transform(a_id, t, R)

                    if self.test_t_convergence(self.animal_history[a_id], i):
                        process_flag[a_id] = False

                    # self.plot_situation()

            if not changed:
                break

        self.prepare_results()
        # self.plot_situation()

    def test_t_convergence(self, animal_history, frame):
        if frame > 0:
            if np.array_equal(animal_history[frame-1]['t'], animal_history[frame]['t']):
                if np.array_equal(animal_history[frame-1]['R'], animal_history[frame]['R']):
                    return True

        return False

    def prepare_results(self):
        for a_id in range(len(self.animals)):
            # The transformation is done using back projection to supress holes inside objects due rounding to int...
            roi_t2 = self.d_map_animals[a_id].roi
            m_ = 3
            roi_t2 = ROI(roi_t2.y_-m_,roi_t2.x_-m_, roi_t2.height_+2*m_, roi_t2.width_+2*m_)

            pts_t2 = []
            for y in range(int(roi_t2.y_), int(roi_t2.y_max_+1)):
                for x in range(int(roi_t2.x_), int(roi_t2.x_max_+1)):
                    pts_t2.append([y, x])

            pts_t2 = np.array(pts_t2)

            pts_t1 = pts_t2 - self.trans_helpers[a_id].centroid
            pts_t1 = np.dot(pts_t1, rotation_matrix(-self.trans_helpers[a_id].angle).T)
            pts_t1 += self.animals[a_id].centroid()

            d = DistanceMap(self.animals[a_id].pts().copy())

            pts_ = []
            for i in range(len(pts_t2[:, 0])):
                if d.is_inside_object(np.asarray(np.round(pts_t1[i, :]), dtype=np.uint32)):
                    pts_.append(pts_t2[i, :])

            pts_ = np.array(pts_)

            self.animals[a_id].pts_ = np.asarray(np.round(pts_), dtype=np.uint32)
            self.animals[a_id].centroid_ = self.trans_helpers[a_id].centroid
            self.animals[a_id].is_virtual = True

    def plot_situation(self):
        plt.close()
        plt.ioff()
        plt.clf()

        fig = plt.figure(figsize=(4, 6.5))
        reg = plt.scatter(self.d_map_region.cont_pts()[:,1], self.d_map_region.cont_pts()[:,0], color='#777777', s=35, edgecolor='k')

        colors = ['magenta', 'cyan', 'yellow', 'blue', 'red', 'green']
        legends = []
        for a_id in range(len(self.animals)):
            a = self.d_map_animals[a_id]
            #
            # leg = plt.scatter(a.cont_pts()[:,1], a.cont_pts()[:,0], color=colors[a_id], s=35, edgecolor='black')

            a = self.animals[a_id]
            pts = a.pts()
            leg = plt.scatter(pts[:,1], pts[:,0], color=colors[a_id], s=35, edgecolor='black')

            legends.append(leg)

        plt.ion()
        plt.show()
        plt.waitforbuttonpress(0)

    def apply_transform(self, a_id, translation, rot):
        pts_ = self.d_map_animals[a_id].cont_pts()

        pts_ = np.dot(pts_, rot.T)
        pts_ += translation

        self.trans_helpers[a_id].centroid = np.asarray(np.dot(self.trans_helpers[a_id].centroid, rot.T) + translation, dtype=np.int32)
        # self.trans_helpers[a_id].global_translation += translation
        # self.trans_helpers[a_id].global_rotation = np.dot(self.trans_helpers[a_id].global_rotation, rot.T)
        self.trans_helpers[a_id].angle += angle_from_matrix(rot)

        self.d_map_animals[a_id] = DistanceMap(pts_, only_cont=True)

        # pts_ = self.animals[a_id].pts()
        # pts_ = np.dot(pts_, rot.T)
        # pts_ += translation
        # self.animals[a_id].pts_ = np.asarray(pts_, dtype=np.int32)

    def compute_transformation(self, pairs):
        apts = pairs[:,0]
        rpts = pairs[:,1]

        use_weights = True

        if use_weights:
            weights = np.linalg.norm(apts-rpts, axis=1) * 2
            weights_ = np.array([weights, weights]).T

            w_sum = np.sum(weights)
            p = np.sum(apts * weights_, axis=0) / w_sum
            q = np.sum(rpts * weights_, axis=0) / w_sum

            # centering
            apts -= p
            rpts -= q


            W = np.diag(weights)

            s = np.dot(apts.transpose(), W)
            s = np.dot(s, rpts)

        else:
            weights = np.linalg.norm(apts-rpts, axis=1)

            p = np.mean(apts, axis=0)
            q = np.mean(rpts, axis=0)

            apts -= p
            rpts -= q

            W = np.diag(weights)

            s = np.dot(apts.transpose(), W)
            s = np.dot(s, rpts)
            # s = np.dot(apts.T, rpts)

        U, _, V = np.linalg.svd(s)

        middle = np.array([[1, 0], [0, np.linalg.det(np.dot(V.T, U.T))]])
        R = np.dot(V.T, middle)
        R = np.dot(R, U.T)

        t = q - np.dot(R, p)

        return t, R


    def get_pairs_a2r_region(self):
        """
        returns pairs (animal pt -> region pt). Each region pt choose nearest animal pt (from all animals).

        :return:
        """

        pairs = []
        for a_id in range(len(self.animals)):
            pairs.append([])

        for pt in self.d_map_region.cont_pts():
            apt, dist, best_a = self.nearest_animal_pt(pt)
            if best_a != -1:
                pairs[best_a].append(np.array([apt, pt]))

        return pairs

    # TODO remove constant
    def nearest_animal_pt(self, pt, max_dist=20):
        """
        pt is point from region
        returns nearest point, distance, and animal_id
        if animal_id = -1, no point within max_dist was found

        :param pt:
        :param max_dist:
        :return:
        """

        best_dist = max_dist
        best_animal = -1
        best = None

        for a_id in range(len(self.animals)):
            apt, d = self.d_map_animals[a_id].get_nearest_point(pt)

            if d < best_dist:
                best = apt
                best_dist = d
                best_animal = a_id

        return best, best_dist, best_animal



    def get_pairs_a2r_a_outside_r(self, a_dmap, r_dmap):
        """
        returns pairs (animal pt -> region pt) for each point from animal which is not inside region

        :param a_dmap:
        :param r_dmap:
        :param pairs:
        :return:
        """

        pairs = []

        for apt in a_dmap.cont_pts():
            if not r_dmap.is_inside_object(apt):
                pt, d = r_dmap.get_nearest_point(apt)
                pairs.append(np.array([apt, pt]))

        return pairs

    def get_settled_ids(self):
        # TODO: remove this constant
        stable_thresh = 0.75

        stability = np.array([self.get_animal_stability(a, self.d_map_region) for a in self.d_map_animals])

        ids = stability > stable_thresh
        return ids

    def get_animal_stability(self, animal_dmap, region_dmap):
        # TODO: remove CONSTANT!!!
        dist_thresh = 4

        score = 0

        for apt in animal_dmap.cont_pts():
            pt, d = region_dmap.get_nearest_point(apt)
            if d < dist_thresh:
                score += 1
            elif region_dmap.is_inside_object(apt):
                score += 0
            else:
                score -= 1

        score /= float(len(animal_dmap.cont_pts()))

        return score


if __name__ == '__main__':
    with open('/Volumes/Seagate Expansion Drive/regions-merged/74.pkl', 'rb') as f:
        data = pickle.load(f)

    # project = ...
    # vid = get_auto_video_manager(project)
    im = vid.seek_frame(74)

    for r in data['ants'][0].state.region['rle']:
        r['col1'] += 10
        r['col2'] += 10
        r['line'] += -3

    # split_by_contours.solve(data['region'], None, [0, 1], data['ants'], p, im.shape, debug=True)



    reg = Region(data['region'])
    draw_points(im, reg.pts())


    a1 = Region(data['ants'][0].state.region)
    # a1.pts_ += np.array([-3., 10.])

    draw_points(im, a1.pts(), color=(0,0,255,0.4))

    a2 = Region(data['ants'][1].state.region)
    draw_points(im, a2.pts(), color=(0,255,0,0.4))

    f = Fitting(reg, [a1, a2], num_of_iterations=200)
    f.fit()

    test_pairs = []
    test_pairs.append(np.array([np.array([1.,0.]), np.array([0., 0.])]))
    test_pairs.append(np.array([np.array([1.,1.]), np.array([1., 0.])]))
    test_pairs.append(np.array([np.array([0.,1.]), np.array([1., 1.])]))
    test_pairs.append(np.array([np.array([0.,0.]), np.array([0., 1.])]))

    test_pairs = np.array(test_pairs)

    t, r = f.compute_transformation(np.copy(test_pairs))
    print t
    print r

    new_pts = []
    for pt in test_pairs[:, 0]:
        # pt -= np.array([2/4., 2/4.])
        pt = np.dot(r, pt.reshape(2,))
        # pt += np.array([2/4., 2/4.])
        pt = pt + t
        new_pts.append(pt)

    print new_pts

    cv2.imshow('im', im)
    cv2.moveWindow('im', 0, 0)
    cv2.waitKey(0)