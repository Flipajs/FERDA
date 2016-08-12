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
FNAME = "region_probability_results.p"


class TransformationClassifier:

    def __init__(self, project):
        self.project = project
        self.graph_supplier = GraphSupplier(project.gm)

        self.results = self.load_previous_results()

    def load_previous_results(self):
        fname = os.path.join(self.project.working_directory, FNAME)
        if exists(fname):
            return pickle.load(open(fname, 'rb'))
        else:
            return {}

    def improve_ground_truth(self, data):
        regions = filter(lambda x: hash_region_tuple(x) not in self.results, data)
        print "a"
        app = QtGui.QApplication(sys.argv)
        widget = ground_truth_widget.GroundTruthWidget(project, self)
        widget.set_data(regions)
        widget.show()
        # app.aboutToQuit.connect(widget.close)
        app.exec_()
        app.quit()

    def accept_results(self, results):
        print results
        self.results.update(results)
        pickle.dump(self.results, open(FNAME, 'wb'))

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

    classifier = TransformationClassifier(project)

    data = [(project.rm[1], project.rm[2]), (project.rm[2], project.rm[3])]
    classifier.improve_ground_truth(data)
    data = [(project.rm[2], project.rm[3]), (project.rm[3], project.rm[4])]
    classifier.improve_ground_truth(data)
