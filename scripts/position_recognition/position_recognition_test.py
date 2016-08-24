import math
from PyQt4 import QtCore
from PyQt4 import QtGui

import numpy as np
import numpy
import sys
from numpy import transpose
from numpy.linalg import eig, norm
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from utils.geometry import rotate
from core.graph.region_chunk import RegionChunk
from core.project.project import Project
from gui.gui_utils import cvimg2qtpixmap

INPUT_DIM = 40
NUMBER_OF_EIGEN_V = 20

average = 0

def pca_basis(X):
    # implementation trick
    T = X.T.dot(X)
    eigenValues, eigenVectors = np.linalg.eig(T)

    # sorting
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    # normalization
    Y = X.dot(eigenVectors)
    for i in range(Y.shape[1]):
        Y[:, i] = Y[:, i] / norm(Y[:, i])

    eigenValues = eigenValues/np.sum(eigenValues)

    return Y, eigenValues


def reconstruct(X, Y, X_mean):
    m, num_samples = X.shape

    # Apply the liner combination and add the mean image
    Z = np.zeros((Y.shape[0], X.shape[1]))
    for i in range(num_samples):
        temp = (Y[:, m - 1] * X[m - 1, i] + X_mean).real
        Z[:, i] = temp.reshape((Z.shape[0], ))

    return Z


def compact_representation(X, Y, m):
    W = Y[:, :m].T.dot(X)
    return W


def get_pca(chunks, number_of_data, chm, gm):
    matrix = []
    i = 1
    for ch in chunks:
        print "Chunk #{0}".format(i)
        i+= 1
        for vector in get_matrix(ch, number_of_data, chm, gm):
            # plt.plot(vector[::2], vector[1::2])
            # plt.show()
            matrix.append(vector)

    print "Constructing eigen ants"

    matrix = np.matrix(matrix)
    X = matrix.T

    X_mean = np.mean(X, axis=1)

    # center the data
    X = X-X_mean

    eigenAnts, eigenValues = pca_basis(X)

    X_c = compact_representation(X, eigenAnts, NUMBER_OF_EIGEN_V)
    Z = reconstruct(X_c, eigenAnts[:, :NUMBER_OF_EIGEN_V], X_mean)

    # print matrix.T - Z

    return eigenAnts


def get_eigenfaces(m):
    covariance_matrix = m.T.dot(m)
    eigenvalues, eigenvectors = eig(covariance_matrix)
    index = eigenvalues.argsort()[::-1]
    eigenfaces = eigenvectors[:, index]
    eigenfaces = eigenfaces[:, :NUMBER_OF_EIGEN_V]
    eigenfaces = m.dot(eigenfaces)
    return eigenfaces


def get_chunks_regions(ch, chm, gm):
    chunk = chm[ch]
    print chunk
    r_ch = RegionChunk(chunk, gm, gm.rm)
    for region in r_ch:
        yield region


def get_matrix(chunk, number_of_data, chm, gm):
    distance_matrix = []
    for region in get_chunks_regions(chunk, chm, gm):
        distance_matrix.append(get_region_vector(region, number_of_data))
    return distance_matrix

def get_region_vector(region, number_of_data):
    centroid = region.centroid()
    contour = region.contour_without_holes() - centroid
    contour = np.array(rotate(contour, -region.theta_))
    centroid = [0,0]

    if len(contour) < number_of_data:
        print("Number of data should be smaller than contours length")

    ant_head, index = find_head_index(centroid, contour, region)
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
    while i < INPUT_DIM:
        if distances[p] >= i * step:
            result[i,] = (contour[p] + (contour[p-1] - contour[p]) * ((distances[p] - i * step) / step))
            i += 1
        p = (p + 1) % con_length

    # plt.axis('equal')
    # plt.plot(contour[:,0], contour[:,1], c='r')
    # plt.scatter(result[:,0], result[:,1], c='g')
    # plt.show()
    return result.flatten()


def find_head_index(centroid, contour, region):
    a = list(enumerate(contour))
    a = filter(lambda v: v[1][1] - centroid[1] > region.a_ / 2, a)
    index = 0
    ant_head = None
    i = 0
    while a[i][1][0] > centroid[0]:
        i += 1
    for v in range(len(a)):
        x1 = a[i][1][0]
        x2 = a[i - 1][1][0]
        if x1 <= centroid[0] < x2:
            y1 = a[i][1][1]
            y2 = a[i - 1][1][1]
            ant_head = [centroid[0], y1 + (y2 - y1) * ((centroid[0] - x1) / (x2 - x1))]
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

def view_chunk(ch):
   pass

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


if __name__ == '__main__':
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")
    # project.load("/Users/flipajs/Documents/wd/GT/Cam1/cam1.fproj")
    chunks = project.gm.chunk_list()

    app = QtGui.QApplication(sys.argv)
    i = 0
    # for ch in chunks:
    #     print i
    #     i += 1
    #     chv = ChunkViewer(project.img_manager, ch, project.chm, project.gm, project.gm.rm)
    #     chv.show()
    #     app.exec_()

    matrix = []
    i = 1
    for ch in chunks:
        print "Chunk #{0}".format(i)
        i += 1
        for vector in get_matrix(ch, INPUT_DIM, project.chm, project.gm):
            matrix.append(vector)

    print "Constructing eigen ants"

    matrix = np.matrix(matrix)
    X = matrix.T
    pca = PCA(NUMBER_OF_EIGEN_V)
    eigen_ants = pca.fit_transform(X)
    X1 = pca.transform(X)
    inverse = pca.inverse_transform(X1)
    # eigen_ants = get_pca(chunks[:5], INPUT_DIM, project.chm, project.gm)
    plt.axis('equal')
    for i in range(len(inverse[0])):
        # print eigen_ant
        # print '\n\n\n'
        # print '\n\n\n'
        plt.plot(inverse[::2, i], inverse[1::2, i])
        plt.plot(X[::2, i], X[1::2, i], c='r')
        plt.show()
    for i in range(NUMBER_OF_EIGEN_V):
        # print eigen_ant
        # print '\n\n\n'
        # print '\n\n\n'
        plt.plot(eigen_ants[::2, i], eigen_ants[1::2, i])
        plt.show()