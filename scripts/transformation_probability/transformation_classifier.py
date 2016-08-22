import cPickle as pickle
import logging
import os
import random
import sys
from PyQt4 import QtGui
from os.path import exists

import numpy as np
from numpy.linalg import norm
from scipy.spatial.qhull import ConvexHull
from sklearn.ensemble import RandomForestClassifier

import view_widget
from core.project.project import Project
from core.region.region import get_orientation
from transformation_trainer import TransformationTrainer, hash_region_tuple

R1 = 1
R2 = 2
R3 = 3

FNAME = 'region_probability_descriptors.p'


class TransformationClassifier():
    def __init__(self, project, regions, results):
        self.project = project
        self.regions = regions
        self.results = results
        self.feature_vectors = {}
        self.classification = {}
        self.probability = {}

        self.fname = os.path.join(self.project.working_directory, FNAME)
        logging.info("Loading previous feature veectors from %s" % self.fname)
        if exists(self.fname):
            self.feature_vectors = pickle.load(open(self.fname, 'rb'))
        else:
            self.feature_vectors = {}
        logging.info("Loaded {0} feature vectors from database".format(len(self.results)))

    def compute_accuracy(self, training_regions, testing_regions, save_results=False, verbose=True):
        X = [self.feature_vectors[hash_region_tuple(r)] for r in training_regions]
        y = [self.results[hash_region_tuple(r)] for r in training_regions]
        rfc = RandomForestClassifier(class_weight='balanced_subsample')
        rfc.fit(X, y)
        X1 = [self.feature_vectors[hash_region_tuple(r)] for r in testing_regions]
        y1 = [self.results[hash_region_tuple(r)] for r in testing_regions]
        accuracy = rfc.score(X1, y1)
        if verbose:
            logging.info("Random forest with {0:.3f} accuracy".format(accuracy))
            t = len(filter(lambda x: x, y))
            f = len(filter(lambda x: not x, y))
            logging.info("Training: True: {0}, False: {1}".format(t, f))
            t = len(filter(lambda x: x, y1))
            f = len(filter(lambda x: not x, y1))
            logging.info("Testing: True: {0}, False: {1}".format(t, f))
        if save_results:
            for r in testing_regions:
                h = hash_region_tuple(r)
                desc = self.feature_vectors[h]
                self.classification[h] = rfc.predict([desc])[0]
                self.probability[h] = rfc.predict_proba([desc])[0]
        return accuracy

    def test(self, regions):
        accuracies = []
        for seed in range(10):
            random.seed(seed)
            np.random.seed(seed)
            random.shuffle(regions)
            training_regions = regions[len(regions) / 2:]
            testing_regions = regions[:len(regions) / 2]
            accuracy = self.compute_accuracy(training_regions, testing_regions, save_results=False)
            accuracies.append(accuracy)

        accuracies = np.array(accuracies)
        logging.info("Mean: {0:.3f}".format(np.mean(accuracies)))
        logging.info("Median: {0:.3f}".format(np.median(accuracies)))
        logging.info("Std: {0:.3f}".format(np.std(accuracies)))

        random.shuffle(regions)
        training_regions = regions[len(regions) / 2:]
        testing_regions = regions[:len(regions) / 2]
        t = filter(lambda x: results[hash_region_tuple(x)], testing_regions)
        accuracy = self.compute_accuracy(training_regions, t, save_results=True)
        logging.info("True classified with {0:.3f} accuracy".format(accuracy))
        f = filter(lambda x: not results[hash_region_tuple(x)], testing_regions)
        accuracy = self.compute_accuracy(training_regions, f, save_results=True)
        logging.info("False classified with {0:.3f} accuracy".format(accuracy))
        self.view_results()

    def view_results(self):
        regions = filter(lambda x: hash_region_tuple(x) in self.classification, self.regions)
        r_t = []
        r_f = []
        for r in regions:
            if bool(self.classification[hash_region_tuple(r)]) == self.results[hash_region_tuple(r)]:
                r_t.append(r)
            else:
                r_f.append(r)
        n_false = len(r_f)

        yes_f_v = map(lambda x: self.feature_vectors[hash_region_tuple(x)], r_t)
        avg_vector_yes = np.mean(yes_f_v, axis=0)
        std_yes = np.std(yes_f_v, axis=0)
        median_yes = np.median(yes_f_v, axis=0)
        no_f_v = map(lambda x: self.feature_vectors[hash_region_tuple(x)], r_f)
        avg_vector_no = np.mean(no_f_v, axis=0)
        std_no = np.std(no_f_v, axis=0)
        median_no = np.median(no_f_v, axis=0)

        widget = view_widget.ViewWidget(self.project, r_f + r_t, self.classification, self.probability, self,
                                        avg_vector_yes, std_yes, median_yes, avg_vector_no, std_no, median_no, n_false)
        widget.show()
        app.exec_()

    def feature_vector(self, regions):
        r1 = regions[0]
        r2 = regions[1]
        ret = []

        # centroid distance
        centr_dist = norm(r2.centroid() - r1.centroid())
        ret.append(centr_dist)

        # margin difference
        margin_diff = abs(r2.margin() - r1.margin())
        ret.append(margin_diff)

        # intensity
        max_intensity = abs(r2.max_intensity_ - r1.max_intensity_)
        min_intensity = abs(r2.min_intensity_ - r1.min_intensity_)
        ret.append(max_intensity)
        ret.append(min_intensity)

        # area difference
        area_diff = abs(r2.area() - r1.area())
        ret.append(area_diff)

        # axis difference
        major_axis_diff = abs(r2.major_axis_ - r1.major_axis_)
        minor_axis_diff = abs(r2.minor_axis_ - r1.minor_axis_)
        ret.append(major_axis_diff)
        ret.append(minor_axis_diff)

        regions_t1 = map(lambda x: self.project.gm.region(x), self.project.gm.get_vertices_in_t(r1.frame()))
        regions_t2 = map(lambda x: self.project.gm.region(x), self.project.gm.get_vertices_in_t(r2.frame()))
        regions_t1 = filter(lambda x: x != r1, regions_t1)
        regions_t2 = filter(lambda x: x != r2, regions_t2)
        m = self.project.stats.major_axis_median

        # small centroid neighbourhood
        ret.append(abs(self.count_regions_in_neighbourhood_contour(r1, regions_t1, R1 * m) -
                       self.count_regions_in_neighbourhood_contour(r2, regions_t2, R1 * m)))
        # bigger centroid neighbourhood
        ret.append(abs(self.count_regions_in_neighbourhood_contour(r1, regions_t1, R2 * m) -
                       self.count_regions_in_neighbourhood_contour(r2, regions_t2, R2 * m)))
        # the centroid biggest neighbourhood
        ret.append(abs(self.count_regions_in_neighbourhood_contour(r1, regions_t1, R3 * m) -
                       self.count_regions_in_neighbourhood_contour(r2, regions_t2, R3 * m)))

        # small neighbourhood
        ret.append(abs(self.count_regions_in_neighbourhood(r1, regions_t1, R1 * m) -
                       self.count_regions_in_neighbourhood(r2, regions_t2, R1 * m)))
        # bigger neighbourhood
        ret.append(abs(self.count_regions_in_neighbourhood(r1, regions_t1, R2 * m) -
                       self.count_regions_in_neighbourhood(r2, regions_t2, R2 * m)))
        # the biggest neighbourhood
        ret.append(abs(self.count_regions_in_neighbourhood(r1, regions_t1, R3 * m) -
                       self.count_regions_in_neighbourhood(r2, regions_t2, R3 * m)))

        # closest and farthest distance between two regions
        a = self.find_extreme_points(r1.contour_without_holes(), r2.contour_without_holes())
        ret.append(a[0])
        ret.append(a[1])

        # closest and farthest distance among all
        a = self.extremes_among_neighbours(r1, regions_t1)
        b = self.extremes_among_neighbours(r2, regions_t2)
        ret.append(abs(a[0] - b[0]))
        ret.append(abs(a[1] - b[1]))

        ret.append(abs(r1.theta_ % np.math.pi - r2.theta_ % np.math.pi))

        return ret

    def descriptor_representation(self, regions):
        desc = self.feature_vector(regions)
        features = ['centroid dist', 'margin diff', 'max_intensity', 'min_intensity', 'area diff',
                    'major_axis diff', 'minor_axis_diff', 'c_small_neigh', 'c_middle_neigh', 'c_big_neigh',
                    'small_neigh', 'middle_neigh', 'big_neigh', 'dist_big', 'dist_small', 'dist_big_all',
                    'dist_small_all', 'theta diff']
        return zip(features, desc)

    def count_regions_in_neighbourhood_contour(self, r, regions, radius):
        c = r.centroid()
        return len(filter(lambda x: norm(c - x.centroid()) < radius, regions))

    def count_regions_in_neighbourhood(self, r, regions, radius):
        c = r.centroid()
        return len(filter(lambda x: self.find_closest_point(c, x.contour_without_holes()) < radius, regions))

    def extremes_among_neighbours(self, r, regions):
        c = r.contour_without_holes()
        farthest = 0
        closest = np.inf
        for reg in regions:
            dist = self.find_extreme_points(c, reg.contour_without_holes())
            farthest = max(farthest, dist[0])
            closest = min(closest, dist[1])
        return farthest, closest

    def find_closest_point(self, c, points):
        convex_hull = ConvexHull(points)
        convex_hull = points[convex_hull.vertices, :]
        dist = norm(c - convex_hull[0])
        for p in convex_hull[1:]:
            d = norm(c - p)
            dist = min(dist, d)
        return dist

    def find_extreme_points(self, points1, points2):
        a = ConvexHull(points1)
        a = points1[a.vertices, :]
        b = ConvexHull(points2)
        b = points2[b.vertices, :]
        farthest = 0
        closest = np.inf
        for p in a:
            for q in b:
                dist = norm(p - q)
                farthest = max(dist, farthest)
                closest = min(dist, closest)
        return farthest, closest

    def update_feature_vectors(self, all=False):
        if not all:
            regions = filter(lambda x: hash_region_tuple(x) not in self.feature_vectors, self.regions)
        else:
            regions = self.regions
        logging.info("Creating features for {0} region tuples".format(len(regions)))
        self.feature_vectors.update({hash_region_tuple(r): self.feature_vector(r) for r in regions})
        logging.info("Saving {0} feature vectors to database. It now contains {1} entries.".format(
            len(regions), len(self.feature_vectors)))
        pickle.dump(self.feature_vectors, open(self.fname, 'wb'))


if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/CompleteGraph/CompleteGraph.fproj")
    logging.basicConfig(level=logging.INFO)

    app = QtGui.QApplication(sys.argv)

    trainer = TransformationTrainer(project)
    regions, results = trainer.get_ground_truth()
    classifier = TransformationClassifier(project, regions, results)
    classifier.update_feature_vectors(all=False)
    # classifier.update_feature_vectors(all=True)
    classifier.test(regions)

    app.quit()
