import logging
import pickle
from PyQt4 import QtGui

import sys

from core.project.project import Project
from scripts.pca.cluster_range.gt_manager import GTManager
from scripts.pca.widgets.tracklet_viewer import TrackletViewer


def transform_index_to_ids(project, index_fname, id_fname):
    f = open(index_fname)
    chunks_with_clusters = []
    for line in f:
        chunks_with_clusters += line.split()
    chunks_with_clusters = map(lambda x: int(x), chunks_with_clusters)
    f.close()

    chunks = project.gm.chunk_list()
    ids = map(lambda x: chunks[x], chunks_with_clusters)
    ids = sorted(ids)

    f = open(id_fname, 'w')
    for i in ids:
        f.write("{0}\n".format(i))
    f.close()


def tag_chunks(project):
    app = QtGui.QApplication(sys.argv)
    i = 0
    for ch in project.gm.chunk_list():
        print i
        i += 1
        chv = TrackletViewer(project.img_manager, ch, project.chm, project.gm, project.gm.rm)
        chv.show()
        app.exec_()


def get_chunk_with_clusters(fname):
    f = open(fname)
    chunks_with_clusters = []
    for line in f:
        chunks_with_clusters += line.split()
    chunks_with_clusters = map(lambda x: int(x), chunks_with_clusters)
    return chunks_with_clusters


def cluster_gt(project, index_fname, results_fname):
    app = QtGui.QApplication(sys.argv)
    chunks_indexes = get_chunk_with_clusters(index_fname)
    chunks = project.gm.chunk_list()
    chunks_ids = [chunks[x] for x in chunks_indexes]

    manager = GTManager(project, results_fname)
    manager.improve_ground_truth(chunks_ids)
    app.exec_()


def get_cluster_gt(pickle_fname):
    return pickle.load(open(pickle_fname, 'rb'))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project = Project()
    project.load("/home/simon/FERDA/projects/clusters_gt/Cam1_/cam1.fproj")

    # label chunks with chunk viewer
    # tag_chunks(project)

    # get gt in dictionary region_id : [list of region convex hulls]
    # gt = get_cluster_gt('/home/simon/FERDA/ferda/scripts/pca/data/clusters_Cam_1_clusters.p')

    # create gt for clusters in file
    results_fname = '/home/simon/FERDA/ferda/scripts/pca/data/clusters_Cam_1_clusters.p'
    # cluster_gt(project, results_fname,
               # '/home/simon/FERDA/ferda/scripts/pca/data/clusters_Cam_1_cluster_tracklets')

    # load
    manager = GTManager(project, results_fname)
    manager.view_results()
    # delete given REGION id's answer
    manager.delete_answer(9)

