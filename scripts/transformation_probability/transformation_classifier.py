import logging
import random
from PyQt4 import QtGui

import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier

from core.project.project import Project
from scripts.transformation_probability.graph_supplier import GraphSupplier
from scripts.transformation_probability.transformation_trainer import TransformationTrainer
import transformation_trainer
from scripts.transformation_probability.view_widget import ViewWidget


class TransformationClassifier():
    def __init__(self, project, regions, results):
        self.project = project
        # for sure
        random.seed(4)
        random.shuffle(regions)
        self.training_regions = regions[len(regions) / 2:]
        self.testing_regions = regions[:len(regions) / 2]
        self.results = results
        self.classification = {}
        self.probability = {}

    def process(self):
        X = [self.descriptor(r) for r in self.training_regions]
        y = [self.results[transformation_trainer.hash_region_tuple(r)] for r in self.training_regions]
        rfc = RandomForestClassifier()
        rfc.fit(X, y)
        X1 = [self.descriptor(r) for r in self.testing_regions]
        y1 = [self.results[transformation_trainer.hash_region_tuple(r)] for r in self.testing_regions]
        logging.info("Random forest with {0} accuracy".format(rfc.score(X1, y1)))
        for r in self.testing_regions:
            desc = self.descriptor(r)
            self.classification[r] = rfc.predict([desc])[0]
            self.probability[r] = rfc.predict_proba([desc])[0]

    def descriptor(self, regions):
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

    def view_results(self):
        regions = [k for k, v in self.classification.items() if
                (bool(v) != self.results[transformation_trainer.hash_region_tuple(k)])]
        widget = ViewWidget(self.project, regions, self.classification, self.probability)
        widget.show()
        app.exec_()


if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/CompleteGraph/CompleteGraph.fproj")
    logging.basicConfig(level=logging.INFO)

    app = QtGui.QApplication(sys.argv)
    trainer = TransformationTrainer(project)
    regions, results = trainer.get_ground_truth()
    classifier = TransformationClassifier(project, regions, results)
    classifier.process()
    classifier.view_results()
    app.quit()
