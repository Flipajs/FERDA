import logging

import numpy as np

from scripts.pca.data.gt_scripts import get_regions_from_tracklets
from utils.geometry import rotate


# Methods below extract important data from ants

def get_matrix(project, chunks, number_of_data, results):
    regions = get_regions_from_tracklets(project, chunks)
    regions = filter(lambda x: x.id() in results, regions)
    sum_dist = 0
    matrix = np.zeros((len(regions), number_of_data, 2))
    sizes = []

    for i in range(len(regions)):
        r = regions[i]
        example, s = get_feature_vector(r, number_of_data, results[r.id()])
        matrix[i] = example
        sum_dist += s
        sizes.append(r.area())

    # for ch in chunks:
    #     vectors, s, size = get_feature_vectors(ch, number_of_data, project.chm, project.gm, results)
    #     for vector in vectors:
    #         matrix.append(vector)
    #     sum_dist += s
    #     sizes += size
    # matrix = np.array(matrix)
    sum_dist /= len(regions)
    return matrix, sum_dist, sizes


# def get_feature_vectors(chunk, number_of_data, chm, gm, results):
#     vectors = []
#     sizes = []
#     sum = 0
#     for region in get_regions_tr(chunk, chm, gm):
#         if region.id() in results:
#             if results.get(region.id(), False):
            # v, s = get_feature_vector(region, number_of_data, results[region.id()])
            # vectors.append(v)
            # sum += s
            # sizes.append(region.area())
    # return vectors, sum, sizes


def get_feature_vector(region, number_of_data, right_orientation):
    centroid = region.centroid()
    contour = region.contour_without_holes() - centroid
    ang = -region.theta_
    if not right_orientation:
        ang -= np.math.pi
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

    # import matplotlib.pyplot as plt
    # plt.axis('equal')
    # plt.scatter(contour[:,0], contour[:,1], c='r')
    # plt.scatter(result[:,0], result[:,1], c='g')
    # plt.hold(False)
    # plt.show()
    return result, step

def find_head_index(contour, region):
    # BEAM ALGORITHM

    # a = list(enumerate(contour))
    # a = filter(lambda v: v[1][1] - 0 > region.a_ / 2, a)
    # index = 0
    # ant_head = None
    #
    # i = 0
    # while a[i][1][0] > 0:
    #     i += 1
    # for v in range(len(a)):
    #     x1 = a[i][1][0]
    #     x2 = a[i - 1][1][0]
    #     if x1 <= 0 < x2:
    #         y1 = a[i][1][1]
    #         y2 = a[i - 1][1][1]
    #         ant_head = [0, y1 + (y2 - y1) * ((0 - x1) / (x2 - x1))]
    #         index = a[i - 1][0]
    #         break
    #     i = (i + 1) % len(a)

    # SIMPLE MAX X
    index = 0
    for i in range(contour.shape[0]):
        if contour[i][1] > contour[index][1]:
            index = i
    ant_head = contour[index]

    # import matplotlib.pyplot as plt
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
    return np.math.sqrt(sum(i ** 2 for i in u))