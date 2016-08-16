import logging
import random
import sys
from PyQt4 import QtGui

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from transformation_trainer import TransformationTrainer, hash_region_tuple
import view_widget
from core.project.project import Project


def descriptor_representation(regions):
    desc = descriptor(regions)
    features = ['centroid distance', 'margin difference', 'max_intensity', 'min_intensity', 'area difference', 'axis difference']
    return zip(features, desc)


def descriptor(regions):
    r1 = regions[0]
    r2 = regions[1]
    ret = []

    # centroid distance
    centr_dist = np.linalg.norm(r2.centroid() - r1.centroid())
    ret.append(centr_dist
               )
    # margin difference
    margin_diff = r2.margin() - r1.margin()
    ret.append(margin_diff)

    # intensity
    max_intensity = r2.max_intensity_ - r1.max_intensity_
    min_intensity = r2.min_intensity_ - r1.min_intensity_
    ret.append(max_intensity)
    ret.append(min_intensity)

    # area difference
    area_diff = r2.area() - r1.area()
    ret.append(area_diff)

    # axis difference
    axis_diff = r2.major_axis_ - r1.major_axis_
    ret.append(axis_diff)

    return ret


class TransformationClassifier():
    def __init__(self, project, results):
        self.project = project
        self.results = results
        self.classification = {}
        self.probability = {}

    def compute_accuracy(self, training_regions, testing_regions, save_results=False):
        X = [descriptor(r) for r in training_regions]
        y = [self.results[hash_region_tuple(r)] for r in training_regions]
        rfc = RandomForestClassifier()
        rfc.fit(X, y)
        X1 = [descriptor(r) for r in testing_regions]
        y1 = [self.results[hash_region_tuple(r)] for r in testing_regions]
        accuracy = rfc.score(X1, y1)
        if save_results:
            for r in testing_regions:
                desc = descriptor(r)
                self.classification[r] = rfc.predict([desc])[0]
                self.probability[r] = rfc.predict_proba([desc])[0]
        return accuracy

    def test(self):
        accuracies = []
        for seed in range(10):
            random.seed(seed)
            np.random.seed(seed)
            random.shuffle(regions)
            training_regions = regions[len(regions) / 2:]
            testing_regions = regions[:len(regions) / 2]

            accuracy = self.compute_accuracy(training_regions, testing_regions, save_results=False)
            logging.info("Random forest with {0:.3f} accuracy".format(accuracy))
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

    def view_results(self):
        regions = [k for k, v in self.classification.items() if
                (bool(v) != self.results[hash_region_tuple(k)])]
        widget = view_widget.ViewWidget(self.project, regions, self.classification, self.probability)
        widget.show()
        app.exec_()


if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/CompleteGraph/CompleteGraph.fproj")
    logging.basicConfig(level=logging.INFO)

    app = QtGui.QApplication(sys.argv)

    trainer = TransformationTrainer(project)
    regions, results = trainer.get_ground_truth()
    classifier = TransformationClassifier(project, results)
    classifier.test()
    classifier.view_results()

    app.quit()
