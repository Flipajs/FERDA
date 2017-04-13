import logging
import math
from PyQt4 import QtGui
from matplotlib import pyplot as plt

import numpy as np
import sys
from sklearn.decomposition import PCA

from core.project.project import Project
from scripts.pca.ant_extract import get_matrix
from scripts.pca.results_generate import generate_eigen_ants_figure, generate_ants_reconstructed_figure
from scripts.pca.widgets.eigen_widget import EigenWidget
from utils.geometry import rotate


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


class AnimalFitting:
    HEAD_RANGE = 10
    BOTTOM_RANGE = 10
    EIGEN_DIM = 10
    FEATURES = 40

    def __init__(self, X):
        self.X = AnimalFitting.get_pca_compatible_data(X)
        self.H = AnimalFitting.extract_heads(self.X)
        self.B = AnimalFitting.extract_bottoms(self.X)

        # fig, ax = plt.subplots()
        # i = 0
        # for x,y in zip(self.X[0, 0::2], self.X[0, 1::2]):
        #     ax.scatter(x, y)
        #     ax.annotate(i, (x, y))
        #     plt.axis('equal')
        #     i += 1
        # plt.show()
        #
        # fig, ax = plt.subplots()
        # i = 0
        # for x,y in zip(self.H[0, 0::2], self.H[0, 1::2]):
        #     ax.scatter(x, y)
        #     ax.annotate(i, (x, y))
        #     plt.axis('equal')
        #     i += 1
        # plt.show()
        # fig, ax = plt.subplots()
        # i = 0
        # for x,y in zip(self.B[0, 0::2], self.B[0, 1::2]):
        #     ax.scatter(x, y)
        #     ax.annotate(i, (x, y))
        #     plt.axis('equal')
        #     i += 1
        # plt.show()

        # SPLIT TRAIN / TEST DATA 0.9 / 0.1
        self.X_train, self.X_test = np.split(self.X, [int(self.X.shape[0] * 0.9)])
        self.H_train, self.H_test = np.split(self.H, [int(self.H.shape[0] * 0.9)])
        self.B_train, self.B_test = np.split(self.B, [int(self.B.shape[0] * 0.9)])

        # PCA ON WHOLE ANIMAL
        self.pca_whole = PCA(AnimalFitting.EIGEN_DIM)
        # self.X_C = self.pca_whole.fit(self.X_train)
        self.pca_whole.fit(self.X_train)
        self.eigen_ants_whole = self.pca_whole.components_
        self.eigen_values_whole = self.pca_whole.explained_variance_
        # X_R = self.pca_whole.inverse_transform(self.pca_whole.transform(self.X_train))

        # PCA ON HEADS
        self.pca_head = PCA(AnimalFitting.EIGEN_DIM)
        # H_C = self.pca_head.fit_transform(self.H_train)
        self.pca_head.fit(self.H_train)
        self.eigen_ants_head = self.pca_head.components_
        self.eigen_values_head = self.pca_head.explained_variance_
        # H_R = np.dot(self.H_C, self.eigen_ants_whole) + self.pca_whole.mean_

        # PCA ON BOTTOMS
        self.pca_bottom = PCA(AnimalFitting.EIGEN_DIM)
        # B_C = self.pca_bottom.fit_transform(self.B_train)
        self.pca_bottom.fit(self.B_train)
        self.eigen_ants_bottom = self.pca_bottom.components_
        self.eigen_values_bottom = self.pca_bottom.explained_variance_
        # B_R = np.dot(self.B_C, self.eigen_ants_whole) + self.pca_whole.mean_

    def show_extracting_random_result(self, n=3):
        import random
        for j in [random.randint(0, self.X.shape[0] - 1) for x in range(n)]:
            plt.plot(np.append(self.X[j, ::2], self.X[j, 0]), np.append(self.X[j, 1::2], self.X[j, 1]), c='b')
            plt.plot(np.append(self.H[j, ::2], self.H[j, 0]), np.append(self.H[j, 1::2], self.H[j, 1]), c='g')
            plt.plot(np.append(self.B[j, ::2], self.B[j, 0]), np.append(self.B[j, 1::2], self.B[j, 1]), c='r')
            plt.axis('equal')
            plt.show()

    def get_fit(self, X):
        head_example = np.squeeze(self.extract_heads(np.expand_dims(X, axis=0)))
        bottom_example = np.squeeze(self.extract_bottoms(np.expand_dims(X, axis=0)))
        head_fit = self.pca_whole.inverse_transform(self.pca_head.transform(np.reshape(head_example, (1, -1))))[0]
        bottom_fit = self.pca_whole.inverse_transform(self.pca_bottom.transform(np.reshape(bottom_example, (1, -1))))[0]

        # fig, ax = plt.subplots()
        # i = 0
        # for x,y in zip(X[0::2], X[1::2]):
        #     ax.scatter(x, y)
        #     ax.annotate(i, (x, y))
        #     plt.axis('equal')
        #     i += 1
        # # plt.show()
        #
        # fig, ax = plt.subplots()
        # i = 0
        # for x,y in zip(head_example[0::2], head_example[1::2]):
        #     ax.scatter(x, y)
        #     ax.annotate(i, (x, y))
        #     plt.axis('equal')
        #     i += 1
        # plt.show()
        eigen_values = self.pca_head.transform(np.reshape(head_example, (1, -1)))[0]
        # fig, ax = plt.subplots()
        # i = 0
        # for x,y in zip(head_fit[0::2], head_fit[1::2]):
        #     ax.scatter(x, y)
        #     ax.annotate(i, (x, y))
        #     plt.axis('equal')
        #     i += 1
        print sorted(eigen_values, reverse=True)
        head_fit = np.dot(sorted(eigen_values, reverse=True), self.eigen_ants_whole) + self.pca_whole.mean_
        print head_fit
        fig, ax = plt.subplots()
        i = 0
        for x, y in zip(head_fit[0::2], head_fit[1::2]):
            ax.scatter(x, y)
            ax.annotate(i, (x, y))
            plt.axis('equal')
            i += 1
        plt.show()

        return head_fit, bottom_fit

    def show_fit(self, X):
        head_example = np.squeeze(self.extract_heads(np.expand_dims(X, axis=0)))
        bottom_example = np.squeeze(self.extract_bottoms(np.expand_dims(X, axis=0)))
        head_fit, bottom_fit = self.get_fit(X)
        self.plot_fits(X, head_example, head_fit, bottom_example, bottom_fit)

    def show_random_fit_result(self, n=3):
        import random
        for j in [random.randint(0, self.X_test.shape[0] - 1) for x in range(n)]:
            self.show_fit(self.X_test[j, :])

    def plot_fits(self, example, head_example, head_fit, bottom_example, bottom_fit):
        plt.plot(np.append(example[::2], example[0]), np.append(example[1::2], example[1]), c='g', label='Test example')
        # plt.plot(np.append(head_example[::2], head_example[0]), np.append(head_example[1::2], head_example[1]), c='g',
        #          alpha=0.75)
        plt.plot(np.append(head_fit[::2], head_fit[0]), np.append(head_fit[1::2], head_fit[1]), c='r', label='Fit')
        # plt.scatter(np.append(example[::2], example[0]), np.append(example[1::2], example[1]),
        #             c='g')
        plt.scatter(np.append(head_example[::2], head_example[0]), np.append(head_example[1::2], head_example[1]),
                    label='Head part', c='b', alpha=0.75)
        # plt.scatter(np.append(head_fit[::2], head_fit[0]), np.append(head_fit[1::2], head_fit[1]),
        #             c='r')

        example[::2] += 50
        bottom_example[::2] += 50
        bottom_fit[::2] += 50

        plt.plot(np.append(example[::2], example[0]), np.append(example[1::2], example[1]), c='g')
        # plt.plot(np.append(bottom_example[::2], bottom_example[0]), np.append(bottom_example[1::2], bottom_example[1]),
        #          c='m')
        plt.plot(np.append(bottom_fit[::2], bottom_fit[0]), np.append(bottom_fit[1::2], bottom_fit[1]), c='r',
                 alpha=0.75)
        # plt.scatter(np.append(example[::2], example[0]), np.append(example[1::2], example[1]),
        #             c='g')
        plt.scatter(np.append(bottom_example[::2], bottom_example[0]),
                    np.append(bottom_example[1::2], bottom_example[1]),
                    label='Bottom part', c='m', alpha=0.75)
        # plt.scatter(np.append(bottom_fit[::2], bottom_fit[0]), np.append(bottom_fit[1::2], bottom_fit[1]),
        #             c='r')
        plt.title("Head and Bottom fits")
        plt.legend(loc='best')
        plt.axis('equal')
        plt.show()

    def view_ant_composition(self, X, type='whole'):
        if type == 'whole':
            pca = self.pca_whole
            eigen_ants = self.eigen_ants_whole
            eigen_values = self.eigen_values_whole
            transformation = lambda x : x  # identity function
        elif type == 'head':
            pca = self.pca_head
            eigen_ants = self.eigen_ants_head
            eigen_values = self.eigen_values_head
            transformation = self.extract_heads
        elif type == 'bottom':
            pca = self.pca_bottom
            eigen_ants = self.eigen_ants_bottom
            eigen_values = self.eigen_values_bottom
            transformation = self.extract_bottoms
        else:
            raise AttributeError("Type should be either 'whole', 'head', or 'bottom'")

        app = QtGui.QApplication(sys.argv)
        X = AnimalFitting.get_pca_compatible_data(np.expand_dims(X, axis=0))
        X = transformation(X)
        X_t = pca.transform(X)[0]
        w = EigenWidget(pca, eigen_ants, eigen_values, X_t)
        w.showMaximized()
        w.close_figures()
        app.exec_()

    def generate_eigen_ants_figure(self, type='whole'):
        if type not in ['whole', 'head', 'bottom']:
            raise AttributeError("Type should be either 'whole', 'head', or 'bottom'")

        eigen_ants =    self.eigen_ants_whole if type == 'whole' else \
                        self.eigen_ants_head if type == 'head' else \
                        self.eigen_ants_bottom
        generate_eigen_ants_figure(eigen_ants, AnimalFitting.EIGEN_DIM)

    def generate_ants_reconstructed_figure(self, rows, columns):
        H_C = self.pca_head.transform(self.H_test)
        H_R = self.pca_whole.inverse_transform(H_C)
        # H_R = self.pca_head.inverse_transform(H_C)
        # generate_ants_reconstructed_figure(self.H_test, H_R, H_C, rows, columns, "heads_reconstructed_test_")
        generate_ants_reconstructed_figure(self.X_test, H_R, H_C, rows, columns, "heads_reconstructed_")

        # X_C = self.pca_whole.transform(self.X_test)
        # X_R = self.pca_whole.inverse_transform(X_C)
        # generate_ants_reconstructed_figure(self.X_test, X_R, X_C, rows, columns)

    @staticmethod
    def extract_heads(X):
        if AnimalFitting.HEAD_RANGE % 2 is not 0:
            logging.warn("Using odd range, results may vary!")
        X_r = np.zeros(X.shape)
        X_r[:,
        range(AnimalFitting.HEAD_RANGE * 2 + 2) + range(X.shape[1] - AnimalFitting.HEAD_RANGE * 2, X.shape[1])] = \
        X[:,
        range(AnimalFitting.HEAD_RANGE * 2 + 2) + range(X.shape[1] - AnimalFitting.HEAD_RANGE * 2, X.shape[1])]
        Ax = X_r[:, AnimalFitting.HEAD_RANGE * 2]
        Ay = X_r[:, AnimalFitting.HEAD_RANGE * 2 + 1]
        Bx = X_r[:, X.shape[1] - AnimalFitting.HEAD_RANGE * 2]
        By = X_r[:, X.shape[1] - AnimalFitting.HEAD_RANGE * 2 + 1]
        vec = (Bx - Ax, By - Ay)
        # size = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
        vec = (vec[0] / (FEATURES - AnimalFitting.HEAD_RANGE * 2 + 1),
               vec[1] / (FEATURES - AnimalFitting.HEAD_RANGE * 2 + 1))
        k = 1

        for i in range(AnimalFitting.HEAD_RANGE * 2 + 2, X.shape[1] - AnimalFitting.HEAD_RANGE * 2 + 1, 2):
            X_r[:, i] = Ax + vec[0] * k
            X_r[:, i+1] = Ay + vec[1] * k
            k += 1
        return X_r

        # return X[:,
        #        range(AnimalFitting.HEAD_RANGE * 2 + 2) + range(X.shape[1] - AnimalFitting.HEAD_RANGE * 2, X.shape[1])]
        # return (np.roll(X, AnimalFitting.HEAD_RANGE * 2 + 2, axis=1))[:, :(AnimalFitting.HEAD_RANGE * 2) * 2][::-1]


    @staticmethod
    def extract_bottoms(X):
        if AnimalFitting.BOTTOM_RANGE % 2 is not 0:
            logging.warn("Using odd range, results may vary!")
        part = X.shape[1] - (AnimalFitting.BOTTOM_RANGE * 4 + 2)
        part /= 2
        return X[:, range(part - 1, X.shape[1] - part + 1)]

    @staticmethod
    def shift_heads_to_origin(X):
        heads = AnimalFitting.extract_heads(X)
        R = np.zeros_like(X)
        means = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            points = zip(heads[i, ::2], heads[i, 1::2])
            means[i] = np.mean(points, axis=0)
            R[i,] = (zip(X[i, ::2], X[i, 1::2]) - means[i]).flatten()
        return R, means

    @staticmethod
    def shift_bottoms_to_origin(X):
        bottoms = AnimalFitting.extract_bottoms(X)
        R = np.zeros_like(X)
        means = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            points = zip(bottoms[i, ::2], bottoms[i, 1::2])
            means[i] = np.mean(points, axis=0)
            R[i,] = (zip(X[i, ::2], X[i, 1::2]) - means[i]).flatten()
        return R, means

    @staticmethod
    def get_pca_compatible_data(X):
        X_comp = np.zeros((X.shape[0], X.shape[1] * 2))
        for i in range(X.shape[0]):
            X_comp[i] = X[i].flatten()
        return X_comp

if __name__ == '__main__':
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
    pca.FEATURES = 40

    # VIEW RESULTS OF EXTRACTING
    # pca.show_extracting_random_result(5)

    # VIEW PCA RECONSTRUCTING RESULTS
    # pca.show_random_fit_result(3)

    # GENERATING RESULTS FIGURE
    # pca.generate_eigen_ants_figure()
    rows = 3
    columns = 11
    pca.generate_ants_reconstructed_figure(rows, columns)

    # VIEW I-TH ANT AS COMPOSITION
    i = 2
    # pca.view_ant_composition(X_ants[i])
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
