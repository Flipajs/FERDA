import logging
import math
from PyQt4 import QtGui
from matplotlib import pyplot as plt
import random

import numpy as np
import sys
from sklearn.decomposition import PCA

from core.project.project import Project
from scripts.pca.ant_extract import get_matrix
from scripts.pca.results_generate import generate_eigen_ants_figure, generate_ants_reconstructed_figure
from scripts.pca.widgets.eigen_widget import EigenWidget
from utils.geometry import rotate


# def fit_cluster(number_of_data, cluster, freq, r_head, pca_shifted_cut_head, pca_shifted_whole_head,
#                 r_bottom, pca_shifted_cut_bottom, pca_shifted_whole_bottom):
#     n = len(cluster) / freq / 2
#     plt.hold(True)
#     cluster = np.expand_dims(cluster, axis=0)
#     results = np.zeros((n * 2, number_of_data * 2))
#     scores = []
#     for i in range(n):
#         blob = extract_heads(np.roll(cluster, - i * freq * 2, axis=1), r_head)
#         blob, mean = shift_heads_to_origin(blob, r_head)
#         ant, score = fit_point(blob, mean, pca_shifted_cut_head, pca_shifted_whole_head)
#         results[2 * i, :] = ant
#         scores.append(score)
#
#         blob = extract_bottoms(np.roll(cluster, - i * freq * 2, axis=1), r_bottom)
#         blob, mean = shift_bottoms_to_origin(blob, r_bottom)
#         ant, score = fit_point(blob, mean, pca_shifted_cut_bottom, pca_shifted_whole_bottom)
#         results[2 * i + 1, :] = ant
#         scores.append(score)
#
#     plt.axis('equal')
#     plt.hold(True)
#     plt.scatter(cluster[0, ::2], cluster[0, 1::2], c='grey')
#     plt.plot(cluster[0, ::2], cluster[0, 1::2], c='grey')
#     for i in sorted(range(n * 2), key=lambda n: scores[n], reverse=True)[:3]:
#         ant = results[i]
#         # plt.scatter(cluster[0, freq * i * 2], cluster[0, freq * i * 2 + 1])
#         plt.plot(ant[::2], ant[1::2], c='b')
#     plt.show()
#
#
# def fit_point(blob, mean, pca_shifted_cut, pca_shifted_whole):
#     # extract bottom, shift to origin and rotate
#     blob = blob.squeeze()
#     mean = mean.squeeze()
#     ang = math.atan2(blob[0], blob[1])[]
#     blob = np.array(rotate(zip(blob[::2], blob[1::2]), ang))
#     # carry out pca and project back
#     blob = blob.flatten().reshape(1, -1)
#     ant = np.dot(pca_shifted_cut.transform(blob), pca_shifted_whole.components_)
#     ant += pca_shifted_whole.mean_
#     ant = np.array(rotate(ant.reshape((ant.shape[1] / 2, 2)), -ang))
#     ant = (ant + mean).flatten()
#     return ant, pca_shifted_cut.score(blob)


class AnimalFitting:
    HEAD_RANGE = 3  # x on each side + head -> 2x + 1 points altogether
    BOTTOM_RANGE = 3  # x on each side + bottom -> 2x + 1 points altogether
    EIGEN_DIM = 20
    FEATURES = 40

    def __init__(self, X):
        self.n = X.shape[0]
        self.train_n = int(X.shape[0] * 0.9)
        self.test_n = self.n - self.train_n

        # SPLIT TRAIN / TEST DATA 0.9 / 0.1
        # indexing of X_train beginning at head ccw
        self.X_train, self.X_test = np.split(X, [self.train_n])

        # TRAIN PCA TWICE FOR BOTH HEAD AND BOTTOM SITUATIONS
        self.pca_head = PCA(AnimalFitting.EIGEN_DIM)
        self.pca_head.fit(AnimalFitting.get_pca_compatible_data(self.roll_to_head(self.X_train)))
        self.eigen_vectors_head = self.pca_head.components_.T
        self.mean_head = np.expand_dims(self.pca_head.mean_, axis=0).T
        self.eigen_values_head = self.pca_head.explained_variance_

        self.pca_bottom = PCA(AnimalFitting.EIGEN_DIM)
        self.pca_bottom.fit(AnimalFitting.get_pca_compatible_data(self.roll_to_bottom(self.X_train)))
        self.eigen_vectors_bottom = self.pca_bottom.components_.T
        self.mean_bottom = np.expand_dims(self.pca_bottom.mean_, axis=0).T
        self.eigen_values_bottom = self.pca_bottom.explained_variance_

    def get_head_fits(self, X):
        """
            Accepts [n * c * 2] ndarrays where n is number of examples, c is number of points in contour
        """
        # transpose for column vectors
        head_example = self.get_pca_compatible_data(self.extract_heads(X)).T

        head_example = head_example - self.mean_head[:(AnimalFitting.HEAD_RANGE * 2 + 1) * 2, :]
        head_coordinates = np.dot(self.eigen_vectors_head[:(AnimalFitting.HEAD_RANGE * 2 + 1) * 2, :].T, head_example)
        head_fit = np.dot(self.eigen_vectors_head, head_coordinates) + self.mean_head

        # transpose back and to original dataframe
        head_fit = AnimalFitting.unroll_to_head(AnimalFitting.get_data_from_pca_data(head_fit.T))

        return head_fit, head_coordinates.T

    def get_bottom_fits(self, X):
        """
            Accepts [n * c * 2] ndarrays where n is number of examples, c is number of points in contour.
            Returns [n * c * 2] ndarray of reconstructions and [n * EIGEN_DIM] of coordinates in orthogonal space of egienvectors
        """
        # transpose for column vectors
        bottom_example = self.get_pca_compatible_data(self.extract_bottoms(X)).T

        bottom_example = bottom_example - self.mean_bottom[:(AnimalFitting.BOTTOM_RANGE * 2 + 1) * 2, :]
        bottom_coordinates = np.dot(self.eigen_vectors_bottom[:(AnimalFitting.BOTTOM_RANGE * 2 + 1) * 2, :].T, bottom_example)
        bottom_fit = np.dot(self.eigen_vectors_bottom, bottom_coordinates) + self.mean_bottom

        # transpose back and to original dataframe
        bottom_fit = AnimalFitting.unroll_to_bottom(AnimalFitting.get_data_from_pca_data(bottom_fit.T))

        return bottom_fit, bottom_coordinates.T

    def show_fits(self, X):
        """
            Accepts [n * c * 2] ndarrays where n is number of examples, c is number of points in contour
            Returns [n * c * 2] ndarray of reconstructions and [n * EIGEN_DIM] of coordinates in orthogonal space of egienvectors
        """
        head_examples = self.extract_heads(X)
        bottom_examples = self.extract_bottoms(X)
        head_fits, _ = self.get_head_fits(X)
        bottom_fits, _ = self.get_bottom_fits(X)

        self.plot_fits(X, head_examples, head_fits, bottom_examples, bottom_fits)

    def show_random_fit_result(self, n=3):
        for j in [random.randint(0, self.X_test.shape[0] - 1) for _ in range(n)]:
            self.show_fits(np.copy(self.X_test[j:j + 1, :]))

    def plot_fits(self, examples, head_examples, head_fits, bottom_examples, bottom_fits):
        for example, head_example, head_fit, bottom_example, bottom_fit in \
                zip(examples, head_examples, head_fits, bottom_examples, bottom_fits):
            plt.plot(np.append(example[:, 0], example[0, 0]), np.append(example[:, 1], example[0, 1]), c='g',
                     label='Test example')
            # plt.plot(np.append(head_example[:, 0], head_example[0]), np.append(head_example[1::2], head_example[1]), c='g',
            #          alpha=0.75)
            plt.plot(np.append(head_fit[:, 0], head_fit[0, 0]), np.append(head_fit[:, 1], head_fit[0, 1]), c='r',
                     label='Fit')
            # plt.scatter(np.append(example[:, 0], example[0, 0]), np.append(example[:, 1], example[0, 1]),
            #             c='g')
            plt.scatter(np.append(head_example[:, 0], head_example[0, 0]),
                        np.append(head_example[:, 1], head_example[0, 1]),
                        label='Head part', c='b', alpha=0.75)
            # plt.scatter(np.append(head_fit[:, 0], head_fit[0, 0]), np.append(head_fit[:, 1], head_fit[0, 1]),
            #             c='r')

            example[:, 0] += 50
            bottom_example[:, 0] += 50
            bottom_fit[:, 0] += 50

            plt.plot(np.append(example[:, 0], example[0, 0]), np.append(example[:, 1], example[0, 1]), c='g')
            # plt.plot(np.append(bottom_example[:, 0], bottom_example[0, 0]), np.append(bottom_example[:, 1], bottom_example[0, 1]),
            #          c='m')
            plt.plot(np.append(bottom_fit[:, 0], bottom_fit[0, 0]), np.append(bottom_fit[:, 1], bottom_fit[0, 1]),
                     c='r',
                     alpha=0.75)
            # plt.scatter(np.append(example[:, 0], example[0, 0]), np.append(example[:, 1], example[0, 1]),
            #             c='g')
            plt.scatter(np.append(bottom_example[:, 0], bottom_example[0, 0]),
                        np.append(bottom_example[:, 1], bottom_example[0, 1]),
                        label='Bottom part', c='m', alpha=0.75)
            # plt.scatter(np.append(bottom_fit[:, 0], bottom_fit[0, 0]), np.append(bottom_fit[:, 1], bottom_fit[0, 1]),
            #             c='r')
            plt.title("Head and Bottom fits")
            plt.legend(loc='best')
            plt.axis('equal')
            plt.show()

    def view_ant_composition(self, X, type='head'):
        if type == 'head':
            pca = self.pca_head
            eigen_ants = self.eigen_vectors_head
            eigen_values = self.eigen_values_head
            transformation = self.get_head_fits
        elif type == 'bottom':
            pca = self.pca_bottom
            eigen_ants = self.eigen_vectors_bottom
            eigen_values = self.eigen_values_bottom
            transformation = self.get_bottom_fits
        else:
            raise AttributeError("Type should be either 'whole', 'head', or 'bottom'")

        app = QtGui.QApplication(sys.argv)
        X_t = transformation(X)
        w = EigenWidget(pca, eigen_ants, eigen_values, X_t)
        w.showMaximized()
        w.close_figures()
        app.exec_()

    def generate_eigen_ants_figure(self, type='head'):
        if type not in ['head', 'bottom']:
            raise AttributeError("Type should be either 'head', or 'bottom'")

        eigen_ants = self.eigen_vectors_head if type == 'head' else \
            self.eigen_vectors_bottom
        generate_eigen_ants_figure(eigen_ants, AnimalFitting.EIGEN_DIM)

    def generate_ants_reconstructed_figure(self, rows, columns, fnames):
        # X_R, X_C = self.get_head_fits(self.X_test)
        X_R, X_C = self.get_bottom_fits(self.X_test)

        generate_ants_reconstructed_figure(self.X_test, X_R, X_C, rows, columns, fnames)

    @staticmethod
    def plot_contour_with_annotations(X):
        if X.ndim == 3:  # multiple instance case
            for i in range(X.shape[0]):
                fig, ax = plt.subplots()
                j = 0
                while j < X.shape[1]:
                    x, y = X[i, j]
                    ax.scatter(x, y)
                    ax.annotate(j, (x, y))
                    plt.axis('equal')
                    j += 1
                plt.show()
        else:
            fig, ax = plt.subplots()
            i = 0
            while i < X.shape[0]:
                x, y = X[i]
                ax.scatter(x, y)
                ax.annotate(i, (x, y))
                plt.axis('equal')
                i += 1
            plt.show()

    @staticmethod
    def roll_to_head(X):
        """
            Rolls examples so that they begin at the beginning of head part.
            Can be used to shift training vectors of whole contour or extracting heads form original data.
            Expects data in form [n, c, (x/y)] where n is nth example, c is cth point in contour and (x/y) are both coordinates
        """
        return np.roll(X, AnimalFitting.HEAD_RANGE, axis=1 if X.ndim == 3 else 0)

    @staticmethod
    def unroll_to_head(X):
        """
            Inverse operation to roll_to_head
        """
        return np.roll(X, -AnimalFitting.HEAD_RANGE, axis=1 if X.ndim == 3 else 0)

    @staticmethod
    def roll_to_bottom(X):
        """
            Rolls examples so that they begin at the beginning of bottom part.
            Can be used to shift training vectors of whole contour or extracting bottoms form original data.
            Expects data in form [n, c, (x/y)] where n is nth example, c is cth point in contour and (x/y) are both coordinates
        """
        beginning = FEATURES / 2 - AnimalFitting.BOTTOM_RANGE
        return np.roll(X, -1 * beginning, axis=1 if X.ndim == 3 else 0)

    @staticmethod
    def unroll_to_bottom(X):
        """
            Inverse operation to roll_to_bottom
        """
        return np.roll(X, FEATURES / 2 - AnimalFitting.BOTTOM_RANGE, axis=1 if X.ndim == 3 else 0)

    @staticmethod
    def extract_heads(X):
        return (AnimalFitting.roll_to_head(X))[:, :(AnimalFitting.HEAD_RANGE * 2 + 1)][::-1]

    @staticmethod
    def extract_bottoms(X):
        return (AnimalFitting.roll_to_bottom(X))[:, :(AnimalFitting.BOTTOM_RANGE * 2 + 1)][::-1]

    # @staticmethod
    # def shift_heads_to_origin(X):
    #     heads = AnimalFitting.extract_heads(X)
    #     R = np.zeros_like(X)
    #     means = np.zeros((X.shape[0], 2))
    #     for i in range(X.shape[0]):
    #         points = zip(heads[i, ::2], heads[i, 1::2])
    #         means[i] = np.mean(points, axis=0)
    #         R[i,] = (zip(X[i, ::2], X[i, 1::2]) - means[i]).flatten()
    #     return R, means

    # @staticmethod
    # def shift_bottoms_to_origin(X):
    #     bottoms = AnimalFitting.extract_bottoms(X)
    #     R = np.zeros_like(X)
    #     means = np.zeros((X.shape[0], 2))
    #     for i in range(X.shape[0]):
    #         points = zip(bottoms[i, ::2], bottoms[i, 1::2])
    #         means[i] = np.mean(points, axis=0)
    #         R[i,] = (zip(X[i, ::2], X[i, 1::2]) - means[i]).flatten()
    #     return R, means

    @staticmethod
    def get_pca_compatible_data(X):
        X_res = np.zeros((X.shape[0], reduce(lambda x, y: x * y, X.shape[1:])))
        for i in range(X.shape[0]):
            X_res[i] = X[i].flatten()
        return X_res

    @staticmethod
    def get_data_from_pca_data(X):
        X_res = np.zeros((X.shape[0], X.shape[1] / 2, 2))
        for i in range(X.shape[0]):
            X_res[i] = X[i].reshape((-1, 2))
        return X_res


if __name__ == '__main__':
    # REPRODUCABILITY
    random.seed(0)
    np.random.seed(0)

    PROJECT = 'zebrafish'
    logging.basicConfig(level=logging.INFO)

    project = Project()
    project.load("/home/simon/FERDA/projects/clusters_gt/{0}/{1}.fproj".format(PROJECT, PROJECT))

    ######################################
    # LABELING CHUNK WITH/WITHOUT ANT CLUSTERS

    from data import gt_scripts

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

    FEATURES = 40
    X_ants, avg_dist, sizes = get_matrix(project, non_cluster_tracklets, FEATURES, heads)
    pca = AnimalFitting(X_ants)
    pca.FEATURES = FEATURES

    # VIEW RESULTS OF EXTRACTING
    # pca.show_extracting_random_result(5)

    # VIEW PCA RECONSTRUCTING RESULTS
    # pca.show_random_fit_result(50)

    # GENERATING RESULTS FIGURE
    # pca.generate_eigen_ants_figure()
    rows = 3
    columns = 11
    pca.generate_ants_reconstructed_figure(rows, columns, "3_3_20_bottom")

    # VIEW I-TH ANT AS COMPOSITION
    i = 2
    # pca.view_ant_composition(X_ants[i], type='head')
    # pca.view_ant_composition(X_ants[i], type='bottom')

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
