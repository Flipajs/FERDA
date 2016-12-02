import cPickle as pickle
import logging
import os
import sys
from PyQt4 import QtGui
from os.path import exists

from core.project.project import Project
from scripts.pca.cluster_range.gt_widget import GTWidget
from scripts.pca.widgets import head_widget

FNAME = 'clusters_Cam_1_gt.p'


class GTManager:
    def __init__(self, project):
        self.project = project
        self.fname = os.path.join(self.project.working_directory, FNAME)
        logging.info("Loading previous results from %s" % self.fname)
        if exists(self.fname):
            self.results = pickle.load(open(self.fname, 'rb'))
        else:
            self.results = {} # id : (ants)
        logging.info("Loaded {0} results from database".format(len(self.results)))

    def improve_ground_truth(self, chunks_with_clusters):
        # regions = filter(lambda x: x.id() not in self.results, regions)
        # regions = filter(lambda x: x.id() in self.results and not self.results[x.id()], regions)

        # if len(regions) > 0:
        widget = GTWidget(self.project, self.results, self, chunks_with_clusters)
        # widget.set_data(regions)
        widget.show()
            # app.exec_()
        # else:
        #     logging.info("You already labeled all provided ants")

    def delete_answer(self, r_id):
        del self.results[r_id]
        self.save_results()

    def accept_results(self, results):
        self.results.update(results)
        logging.info(
            "Saving {0} results to database.".format(len(results)))
        self.save_results()

    def view_results(self):
        import matplotlib.pyplot as plt
        for id, ants in self.results.iteritems():
            logging.info("Showing id {0}".format(id))
            for a in ants:
                plt.plot(a[:, 0], a[:, 1])
            plt.axis('equal')
            plt.gca().invert_yaxis()
            plt.show()

    def save_results(self):
        pickle.dump(self.results, open(self.fname, 'wb'))

    def get_ground_truth(self):
        return self.results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project = Project()
    project.load("/home/simon/FERDA/projects/clusters_gt/Cam1_/cam1.fproj")
    chunks = project.gm.chunk_list()

    # data for clusters_gt/Cam1_
    f = open('/home/simon/FERDA/ferda/scripts/pca/data/clusters_Cam_1_cluster_tracklets')
    chunks_with_clusters = []
    for line in f:
        chunks_with_clusters += line.split()
    chunks_with_clusters = map(lambda x: int(x), chunks_with_clusters)
    f.close()
    chunks_without_clusters = list(set(range(579)) - set(chunks_with_clusters))

    chunks_with_clusters = map(lambda x: chunks[x], chunks_with_clusters)
    chunks_without_clusters = map(lambda x: chunks[x], chunks_without_clusters)

    app = QtGui.QApplication(sys.argv)

    manager = GTManager(project)
    # manager.delete_answer(9)
    # manager.view_results()

    manager.improve_ground_truth(chunks_with_clusters)

    app.exec_()
