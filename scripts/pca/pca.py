import logging
import math
import os
from PyQt4 import QtCore
from PyQt4 import QtGui
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.pyplot import gcf
from matplotlib import patches as mpatches
import numpy as np
import sys
from numpy.linalg import eig, norm
from sklearn.decomposition import PCA

from core.graph.region_chunk import RegionChunk
from core.project.project import Project
from gui.gui_utils import cvimg2qtpixmap
from scripts.pca.eigen_widget import EigenWidget
import head_tag
from utils.geometry import rotate


class ChunkViewer(QtGui.QWidget):
    WIDTH = HEIGHT = 300

    def __init__(self, im, ch, chm, gm, rm):
        super(ChunkViewer, self).__init__()
        self.im = im
        self.regions = list(self.get_regions(ch, chm, gm, rm))
        self.setLayout(QtGui.QVBoxLayout())
        self.buttons = QtGui.QHBoxLayout()
        self.next_b = QtGui.QPushButton('next')
        self.prev_b = QtGui.QPushButton('prev')
        self.img = QtGui.QLabel()
        self.current = -1
        self.prepare_layout()
        self.next_action()
        self.prev_b.setDisabled(True)
        if len(self.regions) == 1:
            self.next_b.setDisabled(True)

    def prepare_layout(self):
        self.layout().addWidget(self.img)
        self.layout().addLayout(self.buttons)
        self.buttons.addWidget(self.prev_b)
        self.buttons.addWidget(self.next_b)
        self.connect(self.prev_b, QtCore.SIGNAL('clicked()'), self.prev_action)
        self.connect(self.next_b, QtCore.SIGNAL('clicked()'), self.next_action)

    def view_region(self):
        region = self.regions[self.current]
        img = self.im.get_crop(region.frame(), region, width=self.WIDTH, height=self.HEIGHT, margin=200)
        pixmap = cvimg2qtpixmap(img)
        self.img.setPixmap(pixmap)
        # plt.scatter(contour[:, 0], contour[:, 1])
        # plt.scatter(contour[0, 0], contour[0, 1], c='r')
        # plt.scatter(region.centroid()[0], region.centroid()[1])
        # plt.show()

    def get_regions(self, ch, chm, gm, rm):
        chunk = chm[ch]
        print chunk
        r_ch = RegionChunk(chunk, gm, rm)
        return r_ch

    def next_action(self):
        print self.current
        if self.current != len(self.regions) - 1:
            self.current += 1
            self.view_region()
            self.prev_b.setDisabled(False)
            if self.current == len(self.regions) - 1:
                self.next_b.setDisabled(True)

    def prev_action(self):
        print self.current
        if self.current != 0:
            self.current -= 1
            self.view_region()
            self.next_b.setDisabled(False)
            if self.current == 0:
                self.prev_b.setDisabled(True)


average = 0


def get_chunks_regions(ch, chm, gm):
    chunk = chm[ch]
    r_ch = RegionChunk(chunk, gm, gm.rm)
    # TODO reverse
    # for region in r_ch:
    #     yield region

    length = chunk.length()
    for region in range(0, length, 4):
        yield r_ch[region]


def get_matrix(chunks, number_of_data, results):
    matrix = []
    sum = 0
    i = 1
    for ch in chunks:
        print "Chunk #{0}".format(i)
        i += 1
        vectors, s = get_feature_vectors(ch, number_of_data, project.chm, project.gm, results)
        for vector in vectors:
            matrix.append(vector)
        sum += s
    matrix = np.array(matrix)
    sum /= len(matrix)
    return matrix, sum


def get_feature_vectors(chunk, number_of_data, chm, gm, results):
    vectors = []
    sum = 0
    for region in get_chunks_regions(chunk, chm, gm):
        if region.id() in results:
            # if results.get(region.id(), False):
            v, s = get_feature_vector(region, number_of_data, results[region.id()])
            vectors.append(v)
            sum += s
    return vectors, sum


def get_feature_vector(region, number_of_data, right_orientation):
    centroid = region.centroid()
    contour = region.contour_without_holes() - centroid
    ang = -region.theta_
    if not right_orientation:
        ang -= math.pi
    contour = np.array(rotate(contour, ang))

    if len(contour) < number_of_data * 2:
        logging.warn("Number of data should be much smaller than contours length")

    ant_head, index = find_head_index(contour, region)
    contour[index:index] = ant_head
    contour = np.roll(contour, - index, axis=0)
    con_length = len(contour)
    distances = [0]
    perimeter = vector_norm(contour[0] - contour[-1])
    for i in range(1, con_length):
        perimeter += vector_norm(contour[i] - contour[i - 1])
        distances.append(perimeter)
    result = np.zeros((number_of_data, 2))
    step = perimeter / float(number_of_data)
    i = 0
    p = 0
    while i < number_of_data:
        if distances[p] >= i * step:
            result[i,] = (contour[p - 1] + (contour[p] - contour[p - 1]) * ((distances[p] - i * step) / step))
            i += 1
        p = (p + 1) % con_length

    # plt.axis('equal')
    # plt.scatter(contour[:,0], contour[:,1], c='r')
    # plt.scatter(result[:,0], result[:,1], c='g')
    # plt.hold(False)
    # plt.show()
    return result.flatten(), step


def find_head_index(contour, region):
    a = list(enumerate(contour))
    a = filter(lambda v: v[1][1] - 0 > region.a_ / 2, a)
    index = 0
    ant_head = None
    i = 0
    while a[i][1][0] > 0:
        i += 1
    for v in range(len(a)):
        x1 = a[i][1][0]
        x2 = a[i - 1][1][0]
        if x1 <= 0 < x2:
            y1 = a[i][1][1]
            y2 = a[i - 1][1][1]
            ant_head = [0, y1 + (y2 - y1) * ((0 - x1) / (x2 - x1))]
            index = a[i - 1][0]
            break
        i = (i + 1) % len(a)
    # plt.axis('equal')
    # plt.plot(contour[:,0], contour[:,1], c='r')
    # plt.scatter(ant_head[0], ant_head[1], c ='g')
    # plt.show()
    return np.array(ant_head), index


def compute_contour_perimeter(contour):
    length = len(contour)
    perimeter = vector_norm(contour[0] - contour[length - 1])
    for i in range(1, length):
        perimeter += vector_norm(contour[i] - contour[i - 1])
    return perimeter


def vector_norm(u):
    return math.sqrt(sum(i ** 2 for i in u))


def extract_heads(X, head_range):
    if head_range % 2 is not 0:
        logging.warn("Using odd range, results may vary!")
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
    if bottom_range % 2 is not 0:
        logging.warn("Using odd range, results may vary!")
    part = X.shape[1] - (bottom_range * 4 + 2)
    part /= 2
    return X[:, range(part, X.shape[1] - part)]


def shift_bottoms_to_origin(X, bottom_range):
    bottoms = extract_bottoms(X, bottom_range)
    R = np.zeros_like(X)
    means = np.zeros((X.shape[0], 2))
    for i in range(X.shape[0]):
        points = zip(bottoms[i, ::2], bottoms[i, 1::2])
        means[i] = np.mean(points, axis=0)
        R[i,] = (zip(X[i, ::2], X[i, 1::2]) - means[i]).flatten()
    return R, means


def get_cluster_region_matrix(chunks, avg_dist):
    X = []
    i = 1
    for ch in chunks:
        print "Chunk #{0}".format(i)
        i += 1
        for vector in get_cluster_regions(ch, project.chm, project.gm, avg_dist):
            X.append(vector)
    X = np.array(X)
    return X


def get_cluster_regions(chunk, chm, gm, avg_dist):
    vectors = []
    for region in get_chunks_regions(chunk, chm, gm):
        v = get_cluster_feature_vector(region, avg_dist)
        vectors.append(v)
    return vectors


def get_cluster_feature_vector(cluster, avg_dist):
    contour = cluster.contour_without_holes() - cluster.centroid()
    distances = [0]
    con_length = len(contour)
    ang = -cluster.theta_
    contour = np.array(rotate(contour, ang))

    perimeter = vector_norm(contour[0] - contour[-1])
    for i in range(1, con_length):
        perimeter += vector_norm(contour[i] - contour[i - 1])
        distances.append(perimeter)

    result = np.zeros((int(math.ceil(perimeter / avg_dist)), 2))
    i = 0
    p = 0
    while p < con_length:
        if distances[p] >= i * avg_dist:
            result[i,] = contour[p - 1] + (contour[p] - contour[p - 1]) * ((distances[p] - i * avg_dist) / avg_dist)
            i += 1
        p += 1

    # plt.axis('equal')
    # plt.scatter(contour[:,0], contour[:,1], c='r')
    # plt.scatter(result[:,0], result[:,1], c='g')
    # plt.show()
    # return result.flatten()
    return result.flatten()


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


def generate_eigen_ants_figure(ants, number_of_eigen_v):
    f = plt.figure(figsize=(number_of_eigen_v / 6 + 1, 6))
    gs1 = gridspec.GridSpec(number_of_eigen_v / 6 + 1, 6)
    gs1.update(wspace=0.3, hspace=0.1)
    for i in range(number_of_eigen_v):
        ax = plt.subplot(gs1[i])
        ax.plot(np.append(ants[i, ::2], ants[i, 0]), np.append(ants[i, 1::2], ants[i, 1]))
        ax.set_title("Eigenant #{0}".format(i + 1))
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes], visible=False)
    fig = gcf()
    fig.suptitle('Dim reduction: {0}'.format(number_of_eigen_v), fontsize=23)
    plt.axis('equal')
    f.set_size_inches(30, 20)
    fold = os.path.join(project.working_directory, 'pca_results')
    if not os.path.exists(fold):
        os.mkdir(fold)
    f.savefig(os.path.join(fold, 'eigen_ants'), dpi=f.dpi)
    plt.ioff()


def generate_ants_image(X, X_R, X_C, r, c, i, fold):
    f = plt.figure(figsize=(r, c))
    gs1 = gridspec.GridSpec(r, c)
    gs1.update(wspace=0.025, hspace=0.05)
    for j in range(len(X)):
        ax1 = plt.subplot(gs1[j])
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.plot(np.append(X[j, ::2], X[j, 0]), np.append(X[j, 1::2], X[j, 1]), c='r')
        ax1.scatter(np.append(X[j, ::2], X[j, 0]), np.append(X[j, 1::2], X[j, 1]), c='r')
        ax1.plot(np.append(X_R[j, ::2], X_R[j, 0]), np.append(X_R[j, 1::2], X_R[j, 1]), c='b')
        ax1.scatter(np.append(X_R[j, ::2], X_R[j, 0]), np.append(X_R[j, 1::2], X_R[j, 1]), c='b')
        ax1.plot(np.arange(len(X_C[j, :])) + 1, X_C[j, :], c='g')

    # red_patch = mpatches.Patch(color='red', label='original')
    # blue_patch = mpatches.Patch(color='blue', label='reconstructed')
    # f.legend(handles=[red_patch], labels=[])
    f.set_size_inches(30, 20)
    f.savefig(os.path.join(fold, str(i)), dpi=f.dpi)
    plt.ioff()


def generate_ants_reconstructed_figure(X, X_R, X_C, rows, columns):
    number_in_pic = rows * columns
    fold = os.path.join(project.working_directory, 'pca_results')
    if not os.path.exists(fold):
        os.mkdir(fold)
    i = 0
    while X.shape[0] != 0:
        generate_ants_image(X[:number_in_pic, :], X_R[:number_in_pic, :], X_C[:number_in_pic, :], rows, columns, i,
                            fold)
        X = np.delete(X, range(number_in_pic), axis=0)
        X_R = np.delete(X_R, range(number_in_pic), axis=0)
        X_C = np.delete(X_C, range(number_in_pic), axis=0)
        i += 1
    generate_ants_image(X, X_R, X_C, rows, columns, i, fold)


def view_ant(pca, eigen_ants, eigen_values, ant):
    w = EigenWidget(pca, eigen_ants, eigen_values, ant)
    w.showMaximized()
    w.close_figures()


if __name__ == '__main__':
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")
    chunks = project.gm.chunk_list()

    # LABELING CHUNK WITH/WITHOUT ANT CLUSTERS
    chunks_without_clusters = [0, 1, 2, 3, 4, 5]
    chunks_with_clusters = [6, 10, 12, 13, 17, 18, 26, 28, 29, 32, 37, 39, 40, 41, 43, 47, 51, 54, 57, 58, 60, 61, 65,
                            67, 69, 73, 75, 78, 81, 84, 87, 90, 93, 94, 96, 99, 102, 105]
    chunks_without_clusters = map(lambda x: chunks[x], chunks_without_clusters)
    chunks_with_clusters = map(lambda x: chunks[x], chunks_with_clusters)
    # app = QtGui.QApplication(sys.argv)
    # i = 0
    # for ch in chunks:
    #     print i
    #     i += 1
    #     chv = ChunkViewer(project.img_manager, ch, project.chm, project.gm, project.gm.rm)
    #     chv.show()
    #     app.exec_()

    number_of_eigen_v = 10
    number_of_data = 40

    logging.basicConfig(level=logging.INFO)

    # TRAINING PART (HEAD LABELING AND ROTATING ANTS)
    trainer = head_tag.HeadGT(project)
    # app = QtGui.QApplication(sys.argv)
    # training_regions = []
    # for chunk in chunks:
    #     ch = project.chm[chunk]
    #     r_ch = RegionChunk(ch, project.gm, project.rm)
    #     training_regions += r_ch
    # trainer.improve_ground_truth(training_regions)
    # app.exec_()
    # trainer.correct_answer(1790, 1796, answer=True)
    # trainer.delete_answer(597, 602)
    # app.quit()
    results = trainer.get_ground_truth()

    # EXTRACTING DATA
    X, avg_dist = get_matrix(chunks_without_clusters, number_of_data, results)
    head_range = 4
    bottom_range = 4
    H = extract_heads(X, head_range)
    B = extract_bottoms(X, bottom_range)

    # PCA ON WHOLE ANT
    pca_whole = PCA(number_of_eigen_v)
    X_C = pca_whole.fit_transform(X)
    eigen_ants_whole = pca_whole.components_
    eigen_values_whole = pca_whole.explained_variance_
    X_R = pca_whole.inverse_transform(pca_whole.transform(X))

    # PCA ON HEADS
    pca_head = PCA(number_of_eigen_v)
    H_C = pca_head.fit_transform(H)
    eigen_ants_head = pca_head.components_
    eigen_values_head = pca_head.explained_variance_
    H_R = np.dot(H_C, eigen_ants_whole) + pca_whole.mean_

    # PCA ON BOTTOMS
    pca_bottom = PCA(number_of_eigen_v)
    B_C = pca_bottom.fit_transform(B)
    eigen_ants_bottom = pca_bottom.components_
    eigen_values_bottom = pca_bottom.explained_variance_
    B_R = np.dot(B_C, eigen_ants_whole) + pca_whole.mean_

    # VIEW PCA RECONSTRUCTING RESULTS
    # for j in range(1):
    #     plt.plot(np.append(H[j, ::2], H[j, 0]), np.append(H[j, 1::2], H[j, 1]), c='r')
    #     plt.plot(np.append(H_R[j, ::2], H_R[j, 0]), np.append(H_R[j, 1::2], H_R[j, 1]), c='b')
    #     plt.show()
    #     plt.plot(np.append(B[j, ::2], B[j, 0]), np.append(B[j, 1::2], B[j, 1]), c='r')
    #     plt.plot(np.append(B_R[j, ::2], B_R[j, 0]), np.append(B_R[j, 1::2], B_R[j, 1]), c='b')
    #     plt.show()

    # WIDGET
    # app = QtGui.QApplication(sys.argv)
    # for i in range(1):
    #     view_ant(pca_whole, eigen_ants_whole, eigen_values_whole, X_C[i])
    #     app.exec_()
    # app.quit()

    # GENERATING RESULTS FIGURE
    # generate_eigen_ants_figure(eigen_ants, number_of_eigen_v)
    # rows = 3
    # columns = 11
    # generate_ants_reconstructed_figure(X, X_R, X_C, rows, columns)

    # CLUSTER DECOMPOSITION
    freq = 1
    C = get_cluster_region_matrix(chunks_with_clusters, avg_dist)
    H_S, means = shift_heads_to_origin(X, head_range)
    pca_head_shifted_whole = PCA(number_of_eigen_v)
    pca_head_shifted_whole.fit(H_S)
    pca_head_shifted_cut = PCA(number_of_eigen_v)
    pca_head_shifted_cut.fit(extract_heads(H_S, head_range))
    B_S, means = shift_bottoms_to_origin(X, bottom_range)
    pca_bottom_shifted_whole = PCA(number_of_eigen_v)
    pca_bottom_shifted_whole.fit(B_S)
    pca_bottom_shifted_cut = PCA(number_of_eigen_v)
    pca_bottom_shifted_cut.fit(extract_bottoms(B_S, bottom_range))
    # for v in B_S:
    #     plt.scatter(v[::2], v[1::2])
    #     plt.scatter([0],[0],c='r')
    #     plt.show()
    for cluster in C:
        fit_cluster(number_of_data, cluster, freq, head_range, pca_head_shifted_cut, pca_head_shifted_whole, bottom_range,
                    pca_bottom_shifted_cut, pca_bottom_shifted_whole)
