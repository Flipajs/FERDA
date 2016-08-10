import math

import numpy as np
import numpy
from numpy import transpose
from numpy.linalg import eig, norm
from matplotlib import pyplot as plt

from sklearn.preprocessing import normalize

from core.project.project import Project

NUMBER_OF_DATA = 40
NUMBER_OF_PC = 10

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
    for ch in chunks:
        for vector in get_matrix(ch, number_of_data, chm, gm):
            matrix.append(vector)

    matrix = np.matrix(matrix)
    X = matrix.T
    X_mean = np.mean(X, axis=1)

    # center the data
    X = X-X_mean

    eigenAnts, eigenValues = pca_basis(X)

    m = 10
    X_c = compact_representation(X, eigenAnts, m)
    Z = reconstruct(X_c, eigenAnts[:, :m], X_mean)

    print matrix.T - Z

    return eigenAnts


def get_eigenfaces(m):
    covariance_matrix = m.T.dot(m)
    eigenvalues, eigenvectors = eig(covariance_matrix)
    index = eigenvalues.argsort()[::-1]
    eigenfaces = eigenvectors[:, index]
    eigenfaces = eigenfaces[:, :NUMBER_OF_PC]
    eigenfaces = m.dot(eigenfaces)
    return eigenfaces


def get_chunks_regions(ch, chm, gm):
    chunk = chm[ch]
    chunk_start = chunk.start_frame(gm)
    chunk_end = chunk.end_frame(gm)
    while chunk_start <= chunk_end:
        yield project.gm.region(chunk[chunk_start])
        chunk_start += 1


def get_matrix(chunk, number_of_data, chm, gm):
    distance_matrix = []
    for region in get_chunks_regions(chunk, chm, gm):
        distance_matrix.append(get_region_vector(region, number_of_data))
    return distance_matrix

def get_region_vector(region, number_of_data):
    contour = region.contour_without_holes()
    con_length = len(contour)

    if len(contour) < number_of_data:
        print("Number of data should be smaller than contours length")

    distances = [0]
    perimeter = vector_norm(contour[0] - contour[-1])
    for i in range(1, con_length):
        perimeter += vector_norm(contour[i] - contour[i - 1])
        distances.append(perimeter)

    result = []
    step = con_length / float(number_of_data)
    i = 0
    while i <= con_length - 1:
        result.append(distances[int(i)])
        i += step

    return result


def compute_contour_perimeter(contour):
    length = len(contour)
    perimeter = vector_norm(contour[0] - contour[length - 1])
    for i in range(1, length):
        perimeter += vector_norm(contour[i] - contour[i - 1])
    return perimeter


def vector_norm(u):
    return math.sqrt(sum(i ** 2 for i in u))

if __name__ == '__main__':
    # from scripts import fix_project
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")
    # project.load("/Users/flipajs/Documents/wd/GT/Cam1/cam1.fproj")
    chunks = project.gm.chunk_list()
    pca = get_pca(chunks[:1], NUMBER_OF_DATA, project.chm, project.gm)