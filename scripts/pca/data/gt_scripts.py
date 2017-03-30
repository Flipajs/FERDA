import logging
import cPickle as pickle
import os
from PyQt4 import QtGui

import sys

from core.graph.region_chunk import RegionChunk
from core.project.project import Project
from scripts.pca.cluster_range.gt_manager import GTManager
from scripts.pca.head_tag import HeadGT
from scripts.pca.widgets.tracklet_viewer import TrackletViewer

GT_LOC = '/home/simon/FERDA/ferda/scripts/pca/data'

def transform_index_to_ids(project):
    index_fname = os.path.join(GT_LOC, '{0}_cluster_tracklets_idxs'.format(project.name))
    id_fname = os.path.join(GT_LOC, '{0}_cluster_tracklets_ids'.format(project.name))
    f = open(index_fname)
    chunks_with_clusters = []
    for line in f:
        chunks_with_clusters += line.split()
    chunks_with_clusters = map(lambda x: int(x), chunks_with_clusters)
    f.close()

    chunks = project.chm.chunk_list()
    ids = map(lambda x: chunks[x].id(), chunks_with_clusters)
    # ids = sorted(ids)

    f = open(id_fname, 'w')
    for i in ids:
        f.write("{0}\n".format(i))
    f.close()


def chunks_gt(project, begin=0):
    app = QtGui.QApplication(sys.argv)
    i = begin

    c = set(get_cluster_tracklets(project))
    for ch in project.chm.chunk_list()[i:]:
        if ch.id() in c:
            print i
            print ch
            i += 1
            chv = TrackletViewer(project.img_manager, ch, project.chm, project.gm, project.gm.rm)
            chv.show()
            app.exec_()


def get_cluster_tracklets(project):
    fname = os.path.join(GT_LOC, "{0}_cluster_tracklets_ids".format(project.name))
    f = open(fname)
    chunks_with_clusters = []
    for line in f:
        chunks_with_clusters += line.split()
    chunks_with_clusters = map(lambda x: int(x), chunks_with_clusters)
    return chunks_with_clusters


def get_non_cluster_tracklets(project):
    # takes the name of file with ids of cluster tracklets and returns its complement (i.e. non-cluster-tracklets)
    tracklets_ids = map(lambda x: x.id(), project.chm.chunk_list())
    cluster_tracklets = get_cluster_tracklets(project)
    for i in cluster_tracklets:
        if i not in set(tracklets_ids):
            print "A probable mistake in GT! {0} id not present in project!".format(i)
    return sorted(list(set(tracklets_ids) - set(cluster_tracklets)))


def get_regions_from_tracklets(tracklets):
    regions = []
    for chunk in tracklets:
        ch = project.chm[chunk]
        r_ch = RegionChunk(ch, project.gm, project.rm)
        # first and last three
        if len(r_ch) < 3:
            regions += r_ch
        else:
            regions.append(r_ch[0])
            regions.append(r_ch[-1])
            regions.append(r_ch[len(r_ch) / 2])
    return regions


def get_head_gt(project):
    fname = os.path.join(GT_LOC, "{0}_heads_GT.p".format(project.name))
    viewer = HeadGT(project, fname)
    return viewer.get_ground_truth()


def head_gt(project):
    app = QtGui.QApplication(sys.argv)
    trainer = HeadGT(project, os.path.join(GT_LOC, "{0}_heads_GT.p".format(project.name)))
    regions = get_regions_from_tracklets(get_non_cluster_tracklets(project))

    trainer.improve_ground_truth(regions)
    app.exec_()
    return trainer


def cluster_gt(project, index_fname, results_fname):
    app = QtGui.QApplication(sys.argv)
    chunks_indexes = get_cluster_tracklets(index_fname)
    chunks = project.chm.chunk_list()
    chunks_ids = [chunks[x] for x in chunks_indexes]

    manager = GTManager(project, results_fname)
    manager.improve_ground_truth(chunks_ids)
    app.exec_()


def get_cluster_gt(pickle_fname):
    return pickle.load(open(pickle_fname, 'rb'))

if __name__ == "__main__":
    PROJECT = 'zebrafish'
    project = Project()
    project.load("/home/simon/FERDA/projects/clusters_gt/{0}/{1}.fproj".format(PROJECT, PROJECT))

    ####################################
    # view and label chunks with chunk viewer

    chunks_gt(project, begin=341)

    # check clusters / non-clusters
    # app = QtGui.QApplication(sys.argv)
    # c = set(get_cluster_tracklets(project))
    # c = set(get_non_cluster_tracklets(project))
    # for ch in project.chm.chunk_list():
    #     if ch.id() in c:
    #         print ch
    #         chv = TrackletViewer(project.img_manager, ch, project.chm, project.gm, project.gm.rm)
    #         chv.show()
    #         app.exec_()

    # transform idx file to id file
    # transform_index_to_ids(project)

    ####################################
    # label heads

    viewer = head_gt(project)
    # print viewer.results
    # print viewer.results[71308]
    # viewer.correct_answer(71308, answer=False)
    # viewer.delete_answer(71297)

    ####################################
    # create clusters GT
    # get gt in dictionary region_id : [list of region convex hulls]
    # gt = get_cluster_gt('/home/simon/FERDA/ferda/scripts/pca/data/clusters_Cam_1_clusters.p')

    # create gt for clusters in file
    # results_fname = '/home/simon/FERDA/ferda/scripts/pca/data/clusters_Cam_1_clusters.p'
    # cluster_gt(project, results_fname,
               # '/home/simon/FERDA/ferda/scripts/pca/data/Cam_1_cluster_tracklets_idxs')

    # load
    # manager = GTManager(project, results_fname)
    # manager.view_results()
    # delete given REGION id's answer
    # manager.delete_answer(9)

