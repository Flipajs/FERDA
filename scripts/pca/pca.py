import logging
import math
from PyQt4 import QtGui
from matplotlib import pyplot as plt

import numpy as np
import sys
from sklearn.decomposition import PCA

import head_tag
from core.project.project import Project
from scripts.pca.ant_extract import get_matrix
from scripts.pca.cluster_range.gt_widget import GTWidget
from scripts.pca.range_computer import OptimalRange
from scripts.pca.results_generate import view_ant_composition
from scripts.pca.widgets.tracklet_viewer import TrackletViewer
from utils.geometry import rotate
import os

def extract_heads(X, head_range):
    # if head_range % 2 is not 0:
    #     logging.warn("Using odd range, results may vary!")
    return X[:, range(head_range * 2 + 2) + range(X.shape[1] - head_range * 2, X.shape[1])]


def shift_heads_to_origin(X, head_range):
    heads = extract_heads(X, head_range)
    R = np.zeros_like(X)
    means = np.zeros((X.shape[0], 2))
    for i in range(X.shape[0]):
        points = zip(heads[i, ::2], heads[i, 1::2])
        means[i] = np.mean(points, axis=0)
        R[i,] = (zip(X[i, ::2], X[i, 1::2]) - means[i]).flatten()
    return R, means


def extract_bottoms(X, bottom_range):
    # if bottom_range % 2 is not 0:
    #     logging.warn("Using odd range, results may vary!")
    part = X.shape[1] - (bottom_range * 4 + 2)
    part /= 2
    return X[:, range(part - 1, X.shape[1] - part + 1)]


def shift_bottoms_to_origin(X, bottom_range):
    bottoms = extract_bottoms(X, bottom_range)
    R = np.zeros_like(X)
    means = np.zeros((X.shape[0], 2))
    for i in range(X.shape[0]):
        points = zip(bottoms[i, ::2], bottoms[i, 1::2])
        means[i] = np.mean(points, axis=0)
        R[i,] = (zip(X[i, ::2], X[i, 1::2]) - means[i]).flatten()
    return R, means


def get_pca_compatible_data(X):
    X_comp = np.zeros((X.shape[0], X.shape[1] * 2))
    for i in range(X.shape[0]):
        X_comp[i] = X[i].flatten()
    return X_comp


def fit_cluster(number_of_data, cluster, freq, r_head, pca_shifted_cut_head, pca_shifted_whole_head,
                r_bottom, pca_shifted_cut_bottom, pca_shifted_whole_bottom):
    n = len(cluster) / freq / 2
    plt.hold(True)
    cluster = np.expand_dims(cluster, axis=0)
    results = np.zeros((n * 2, number_of_data * 2))
    scores = []
    for i in range(n):
        blob = extract_heads(np.roll(cluster, - i * freq * 2, axis=1), r_head)
        blob, mean = shift_heads_to_origin(blob, r_head)
        ant, score = fit_point(blob, mean, pca_shifted_cut_head, pca_shifted_whole_head)
        results[2 * i, :] = ant
        scores.append(score)

        blob = extract_bottoms(np.roll(cluster, - i * freq * 2, axis=1), r_bottom)
        blob, mean = shift_bottoms_to_origin(blob, r_bottom)
        ant, score = fit_point(blob, mean, pca_shifted_cut_bottom, pca_shifted_whole_bottom)
        results[2 * i + 1, :] = ant
        scores.append(score)

    plt.axis('equal')
    plt.hold(True)
    plt.scatter(cluster[0, ::2], cluster[0, 1::2], c='grey')
    plt.plot(cluster[0, ::2], cluster[0, 1::2], c='grey')
    for i in sorted(range(n * 2), key=lambda n: scores[n], reverse=True)[:3]:
        ant = results[i]
        # plt.scatter(cluster[0, freq * i * 2], cluster[0, freq * i * 2 + 1])
        plt.plot(ant[::2], ant[1::2], c='b')
    plt.show()


def fit_point(blob, mean, pca_shifted_cut, pca_shifted_whole):
    # extract bottom, shift to origin and rotate
    blob = blob.squeeze()
    mean = mean.squeeze()
    ang = math.atan2(blob[0], blob[1])
    blob = np.array(rotate(zip(blob[::2], blob[1::2]), ang))
    # carry out pca and project back
    blob = blob.flatten().reshape(1, -1)
    ant = np.dot(pca_shifted_cut.transform(blob), pca_shifted_whole.components_)
    ant += pca_shifted_whole.mean_
    ant = np.array(rotate(ant.reshape((ant.shape[1] / 2, 2)), -ang))
    ant = (ant + mean).flatten()
    return ant, pca_shifted_cut.score(blob)


if __name__ == '__main__':
    PROJECT = 'zebrafish'
    EIGEN_DIM = 10
    FEATURES = 40

    logging.basicConfig(level=logging.INFO)

    project = Project()
    project.load("/home/simon/FERDA/projects/clusters_gt/{0}/{1}.fproj".format(PROJECT, PROJECT))

    ######################################
    # LABELING CHUNK WITH/WITHOUT ANT CLUSTERS

    from data import gt_scripts
    # indexes!
    cluster_tracklets = gt_scripts.get_cluster_tracklets(project)
    non_cluster_tracklets = gt_scripts.get_non_cluster_tracklets(project)

    # VIEW TRACKLETS
    # app = QtGui.QApplication(sys.argv)
    # for i in cluster_tracklets:
    #     chunk = project.chm[i]
    #     viewer = TrackletViewer(project.img_manager, chunk, project.chm, project.gm, project.rm)
    #     viewer.show()
    #     app.exec_()

    ######################################
    # GET HEADS (for rotating of bodies)

    heads = gt_scripts.get_head_gt(project)

    ######################################
    # EXTRACT DATA

    X_ants, avg_dist, sizes = get_matrix(project, non_cluster_tracklets, FEATURES, heads)
    X = get_pca_compatible_data(X_ants)
    head_range = 5
    bottom_range = 5
    H = extract_heads(X, head_range)
    B = extract_bottoms(X, bottom_range)
    # VIEW RESULTS OF EXTRACTING
    for j in range(10):
        plt.plot(np.append(X[j, ::2], X[j, 0]), np.append(X[j, 1::2], X[j, 1]), c='b')
        plt.plot(np.append(H[j, ::2], H[j, 0]), np.append(H[j, 1::2], H[j, 1]), c='g')
        plt.plot(np.append(B[j, ::2], B[j, 0]), np.append(B[j, 1::2], B[j, 1]), c='r')
        plt.show()

    # PCA ON WHOLE ANT
    pca_whole = PCA(EIGEN_DIM)
    X_C = pca_whole.fit_transform(X)
    eigen_ants_whole = pca_whole.components_
    eigen_values_whole = pca_whole.explained_variance_
    X_R = pca_whole.inverse_transform(pca_whole.transform(X))

    # PCA ON HEADS
    pca_head = PCA(EIGEN_DIM)
    H_C = pca_head.fit_transform(H)
    eigen_ants_head = pca_head.components_
    eigen_values_head = pca_head.explained_variance_
    H_R = np.dot(H_C, eigen_ants_whole) + pca_whole.mean_

    # PCA ON BOTTOMS
    # pca_bottom = PCA(number_of_eigen_v)
    # B_C = pca_bottom.fit_transform(B)
    # eigen_ants_bottom = pca_bottom.components_
    # eigen_values_bottom = pca_bottom.explained_variance_
    # B_R = np.dot(B_C, eigen_ants_whole) + pca_whole.mean_

    # VIEW PCA RECONSTRUCTING RESULTS
    # for j in range(10):
    #     plt.plot(np.append(H[j, ::2], H[j, 0]), np.append(H[j, 1::2], H[j, 1]), c='r')
    #     plt.plot(np.append(H_R[j, ::2], H_R[j, 0]), np.append(H_R[j, 1::2], H_R[j, 1]), c='b')
    #     plt.show()
    #     plt.plot(np.append(B[j, ::2], B[j, 0]), np.append(B[j, 1::2], B[j, 1]), c='r')
    #     plt.plot(np.append(B_R[j, ::2], B_R[j, 0]), np.append(B_R[j, 1::2], B_R[j, 1]), c='b')
    #     plt.show()


    # GENERATING RESULTS FIGURE
    # generate_eigen_ants_figure(project, eigen_ants_whole, number_of_eigen_v)
    # rows = 3
    # columns = 11
    # generate_ants_reconstructed_figure(project, X, X_R, X_C, rows, columns)

    # VIEW I-TH ANT AS COMPOSITION
    i = 1
    view_ant_composition(pca_whole, eigen_ants_whole, eigen_values_whole, X_C[i])

    # CLUSTER DECOMPOSITION
    # freq = 1
    # C = get_cluster_region_matrix(project, chunks_with_clusters, avg_dist)
    # H_S, means = shift_heads_to_origin(X, head_range)
    # pca_head_shifted_whole = PCA(number_of_eigen_v)
    # pca_head_shifted_whole.fit(H_S)
    # pca_head_shifted_cut = PCA(number_of_eigen_v)
    # pca_head_shifted_cut.fit(extract_heads(H_S, head_range))
    # B_S, means = shift_bottoms_to_origin(X, bottom_range)
    # pca_bottom_shifted_whole = PCA(number_of_eigen_v)
    # pca_bottom_shifted_whole.fit(B_S)
    # pca_bottom_shifted_cut = PCA(number_of_eigen_v)
    # pca_bottom_shifted_cut.fit(extract_bottoms(B_S, bottom_range))
    # for v in B_S:
    #     plt.scatter(v[::2], v[1::2])
    #     plt.scatter([0],[0],c='r')
    #     plt.show()
    # for cluster in C:
    #     fit_cluster(number_of_data, cluster, freq, head_range, pca_head_shifted_cut, pca_head_shifted_whole, bottom_range,
    #                 pca_bottom_shifted_cut, pca_bottom_shifted_whole)

    # optimal k for each point in contour
    # range_comp = OptimalRange(X, sizes, number_of_data, number_of_eigen_v)
    # for i in range(number_of_data):
    #     print range_comp.get_optimal_k(i)

    # optimal k for clusters
    # gt = GTWidget(project, chunks_with_clusters)