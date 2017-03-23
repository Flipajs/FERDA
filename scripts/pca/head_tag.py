import cPickle as pickle
import logging
import os
import sys
from PyQt4 import QtGui
from os.path import exists

from core.project.project import Project
from scripts.pca.widgets import head_widget

PRIME = 2 ** 8 + 1


class HeadGT:
    def __init__(self, project, fname):
        self.project = project
        self.fname = fname
        logging.info("Loading previous results from %s" % self.fname)
        if exists(self.fname):
            self.results = pickle.load(open(self.fname, 'rb'))
        else:
            self.results = {}
        logging.info("Loaded {0} results from database".format(len(self.results)))

    def improve_ground_truth(self, regions):
        regions = filter(lambda x: x.id() not in self.results, regions)
        # regions = filter(lambda x: x.id() in self.results and not self.results[x.id()], regions)
        if len(regions) > 0:
            widget = head_widget.HeadWidget(self.project, self)
            widget.set_data(regions)
            widget.show()
            # app.exec_()
        else:
            logging.info("You already labeled all provided ants")

    def correct_answer(self, r_id, answer=True):
        self.results[r_id] = answer
        self.save_results()

    def delete_answer(self, r_id):
        del self.results[r_id]
        self.save_results()

    def accept_results(self, results):
        self.results.update(results)
        print "Saving {0} results to database. It now contains {1} entries.".format(len(results), len(self.results))
        self.save_results()

    def save_results(self):
        pickle.dump(self.results, open(self.fname, 'wb'))

    def get_ground_truth(self):
        return self.results


if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/CompleteGraph/CompleteGraph.fproj")
    logging.basicConfig(level=logging.INFO)

    app = QtGui.QApplication(sys.argv)
    trainer = HeadGT(project)
    # print trainer.results[10, 10]
    # trainer.correct_answer(1790, 1796, answer=True)
    # trainer.delete_answer(597, 602)
    trainer.improve_ground_truth()
    app.quit()
