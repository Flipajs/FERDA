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


class TransformationTrainer:
    def __init__(self, project):
        self.project = project
        self.supplier = GraphSupplier(project.gm)
        self.fname = os.path.join(self.project.working_directory, FNAME)
        logging.info("Loading previous results from %s" % self.fname)
        if exists(self.fname):
            self.results = pickle.load(open(self.fname, 'rb'))
        else:
            self.results = {}
        logging.info("Loaded {0} results from database".format(len(self.results)))

    def improve_ground_truth(self):
        data = self.supplier.get_nodes_tuples()
        # regions = filter(lambda x: x[0].id() == 1799 and x[1].id() == 1805, data)
        regions = filter(lambda x: hash_region_tuple(x) not in self.results, data)
        regions = sorted(regions, key=lambda x: abs(x[0].area_ - x[1].area_))
        widget = ground_truth_widget.GroundTruthWidget(project, self)
        widget.set_data(regions)
        widget.show()
        app.exec_()

    def correct_answer(self, id1, id2, answer=True):
        self.results[hash_region_tuple((self.project.rm[id1], self.project.rm[id2]))] = answer
        self.save_results()

    def delete_answer(self, id1, id2):
        del self.results[hash_region_tuple((self.project.rm[id1], self.project.rm[id2]))]
        self.save_results()

    def accept_results(self, results):
        self.results.update(results)
        logging.info(
            "Saving {0} results to database. It now contains {1} entries.".format(len(results), len(self.results)))
        self.save_results()

    def save_results(self):
        pickle.dump(self.results, open(self.fname, 'wb'))

    def get_ground_truth(self):
        data = self.supplier.get_nodes_tuples()
        regions = filter(lambda x: hash_region_tuple(x) in self.results, data)
        return regions, self.results

def hash_region_tuple(region_tuple):
    return (PRIME + region_tuple[0].id()) * PRIME + region_tuple[1].id()


if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/CompleteGraph/CompleteGraph.fproj")
    logging.basicConfig(level=logging.INFO)

    app = QtGui.QApplication(sys.argv)
    classifier = TransformationTrainer(project)
    # print classifier.results[]
    # classifier.correct_answer(2348, 2357, answer=True)
    # classifier.delete_answer(4813, 4814)
    classifier.improve_ground_truth()
    app.quit()
