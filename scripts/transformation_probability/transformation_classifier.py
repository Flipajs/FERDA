import logging
import random
import sys
from PyQt4 import QtGui
import os
import numpy as np
from os.path import exists
from sklearn.ensemble import RandomForestClassifier
from transformation_trainer import TransformationTrainer, hash_region_tuple
import view_widget
from core.project.project import Project

R1 = 1
R2 = 2
R3 = 3

FNAME = 'region_probability_descriptors.p'


class TransformationClassifier():
    def __init__(self, project, results):
        self.project = project
        self.results = results
        self.feature_vectors = {}
        self.classification = {}
        self.probability = {}

        self.fname = os.path.join(self.project.working_directory, FNAME)
        logging.info("Loading previous descriptors from %s" % self.fname)
        if exists(self.fname):
            self.feature_vectors = np.pickle.load(open(self.fname, 'rb'))
        else:
            self.feature_vectors = {}
        logging.info("Loaded {0} descriptors from database".format(len(self.results)))

    def compute_accuracy(self, training_regions, testing_regions, save_results=False, verbose=True):
        X = [self.feature_vectors[r] for r in training_regions]
        y = [self.results[hash_region_tuple(r)] for r in training_regions]
        rfc = RandomForestClassifier(class_weight='balanced_subsample')
        rfc.fit(X, y)
        X1 = [self.feature_vectors[r] for r in testing_regions]
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
                desc = self.feature_vector(r)
                self.classification[r] = rfc.predict([desc])[0]
                self.probability[r] = rfc.predict_proba([desc])[0]
        return accuracy

    def test(self, regions):
        accuracies = []
        for seed in range(0):
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

        # random.shuffle(regions)
        # training_regions = regions[len(regions) / 2:]
        # testing_regions = regions[:len(regions) / 2]
        # t = filter(lambda x: results[hash_region_tuple(x)], testing_regions)
        # accuracy = self.compute_accuracy(training_regions, t, save_results=True)
        # logging.info("True classified with {0:.3f} accuracy".format(accuracy))
        # f = filter(lambda x: not results[hash_region_tuple(x)], testing_regions)
        # accuracy = self.compute_accuracy(training_regions, f, save_results=True)
        # logging.info("False classified with {0:.3f} accuracy".format(accuracy))

    def view_results(self):
        regions = [k for k, v in self.classification.items() if
                   (bool(v) != self.results[hash_region_tuple(k)])]
        avg_vector = np.mean(np.array(self.feature_vectors.values(), axis=0))
        # widget = view_widget.ViewWidget(self.project, regions, self.classification, self.probability, self, avg_vector)
        # widget.show()
        app.exec_()

    def feature_vector(self, regions):
        r1 = regions[0]
        r2 = regions[1]
        ret = []

        # centroid distance
        centr_dist = np.linalg.norm(r2.centroid() - r1.centroid())
        ret.append(centr_dist
                   )
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
        axis_diff = abs(r2.major_axis_ - r1.major_axis_)
        ret.append(axis_diff)

        regions_t1 = map(lambda x: self.project.gm.region(x).centroid(), self.project.gm.get_vertices_in_t(r1.frame()))
        regions_t2 = map(lambda x: self.project.gm.region(x).centroid(), self.project.gm.get_vertices_in_t(r2.frame()))
        m = self.project.stats.major_axis_median
        # small neighbourhood
        ret.append(abs(self._count_regions_in_neighbourhood(r1, regions_t1, R1 * m) -
                       self._count_regions_in_neighbourhood(r2, regions_t2, R1 * m)))
        # bigger neighbourhood
        ret.append(abs(self._count_regions_in_neighbourhood(r1, regions_t1, R2 * m) -
                       self._count_regions_in_neighbourhood(r2, regions_t2, R2 * m)))
        # the biggest neighbourhood
        ret.append(abs(self._count_regions_in_neighbourhood(r1, regions_t1, R3 * m) -
                       self._count_regions_in_neighbourhood(r2, regions_t2, R3 * m)))

        return ret

    def descriptor_representation(self, regions):
        desc = self.feature_vector(regions)
        features = ['centroid distance', 'margin difference', 'max_intensity', 'min_intensity', 'area difference',
                    'axis difference', 'small_neigh', 'bigger_neigh', 'the biggest_neigh']
        return zip(features, desc)

    def _count_regions_in_neighbourhood(self, r, centroids, radius):
        c = r.centroid()
        return len(filter(lambda x: np.linalg.norm(c - x) < radius, centroids))

    def update_feature_vectors(self, regions, all=False):
        regions = filter(lambda x: x not in self.feature_vectors, regions)
        self.feature_vectors.update({r: self.feature_vector(r) for r in regions})
        logging.info("Saving {0} descriptors to database. It now contains {1} entries.".format(len(regions),
                                                                                               len(self.feature_vectors)))

if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/CompleteGraph/CompleteGraph.fproj")
    logging.basicConfig(level=logging.INFO)

    app = QtGui.QApplication(sys.argv)

    trainer = TransformationTrainer(project)
    regions, results = trainer.get_ground_truth()
    classifier = TransformationClassifier(project, results)
    classifier.update_feature_vectors(regions, all=False)
    # classifier.update_feature_vectors(regions, all=True)
    classifier.test(regions)
    classifier.view_results()

    app.quit()
