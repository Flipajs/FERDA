import math
import numpy as np
from core import project
from scripts.pca.ant_extract import get_chunks_regions
from utils.geometry import rotate


# Methods below extract important data from clusters


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


def vector_norm(u):
    return math.sqrt(sum(i ** 2 for i in u))
