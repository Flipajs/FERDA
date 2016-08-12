import logging
from PyQt4 import QtCore
from PyQt4 import QtGui
import cPickle as pickle
import numpy as np
import sys
import os
from os.path import exists

from core.project.project import Project
import ground_truth_widget
from scripts.transformation_probability.graph_supplier import GraphSupplier

PRIME = 2 ** 8 + 1
FNAME = 'region_probability_results.p'


class TransformationClassifier:
    def __init__(self, project):
        self.project = project

        self.fname = os.path.join(self.project.working_directory, FNAME)
        logging.info("Loading previous results from %s" % self.fname)
        if exists(self.fname):
            self.results = pickle.load(open(self.fname, 'rb'))
        else:
            self.results = {}
        logging.info("Loaded {0} results from database".format(len(self.results)))

    def improve_ground_truth(self, data):
        regions = filter(lambda x: hash_region_tuple(x) not in self.results, data)
        regions = sorted(regions, key=lambda x: abs(x[0].area_ - x[1].area_))
        widget = ground_truth_widget.GroundTruthWidget(project, self)
        widget.set_data(regions)
        widget.show()
        # app.connect(app, QtCore.SIGNAL("aboutToQuit()"), widget.close)
        app.exec_()

    def correct_answer(self, id1, id2, answer=True):
        self.results[hash_region_tuple((self.project.rm[id1], self.project.rm[id2]))] = answer

    def accept_results(self, results):
        self.results.update(results)
        logging.info(
            "Saving {0} results to database. It now contains {1} entries.".format(len(results), len(self.results)))
        pickle.dump(self.results, open(self.fname, 'wb'))

    def descriptor(self, r1, r2):
        # centroid distance
        centr_dist = np.linalg.norm(r2.centroid() - r1.centroid())

        # margin difference
        margin_diff = r2.margin() - r1.margin()

        # intensity
        max_intensity = r2.max_intensity_ - r1.max_intensity_
        min_intensity = r2.min_intensity_ - r1.min_intensity_

        # area difference
        area_diff = r2.area() - r1.area()

        # axis difference
        axis_diff = r2.major_axis_ - r1.major_axis_

        return centr_dist, margin_diff, max_intensity, min_intensity, area_diff, axis_diff


def hash_region_tuple(region_tuple):
    return (PRIME + region_tuple[0].id()) * PRIME + region_tuple[1].id()


if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/CompleteGraph/CompleteGraph.fproj")
    logging.basicConfig(level=logging.INFO)

    app = QtGui.QApplication(sys.argv)
    classifier = TransformationClassifier(project)
    supplier = GraphSupplier(project.gm)
    # classifier.correct_answer(2237, 2242)
    classifier.improve_ground_truth(supplier.get_nodes_tuples())
    app.quit()
