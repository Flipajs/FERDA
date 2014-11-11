__author__ = 'filip@naiser.cz'
import pickle
import time
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


def get_bounds(region):
    points = []
    for r in region['rle']:
        points.append([r['line'], r['col1']])
        points.append([r['line'], r['col2']])

    return points


def is_in_list(l, el):
    for a in l:
        if a == el:
            return True

    return False


def get_points(region):
    points = []
    for r in region['rle']:
        for c in range(r['col1'], r['col2'] + 1):
            points.append([r['line'], c])

    return points


def find_interesting_pairs(hull_points):
    interesting = []
    for i in range(len(hull_points)):
        p = hull_points[i]
        for j in range(i, len(hull_points)):
            a = hull_points[j]
            if 25 > np.linalg.norm(p-a) > 15:
                interesting.append([p, a])

    return interesting



file10 = open('../out/collisions/regions_437pkl', 'rb')
regions10 = pickle.load(file10)
file10.close()

mser = regions10[0]
start = time.time()
bound = get_bounds(mser)
#bound = get_points(mser)
bound = np.array(bound)
hull = ConvexHull(bound)
time_length = time.time() - start

print time_length

plt.ion()
plt.figure()
plt.plot(bound[:, 0], bound[:, 1], 'bo')

hull_points_idx = []

for simplex in hull.simplices:
    plt.plot(bound[simplex, 0], bound[simplex, 1], 'k-')
    plt.plot(bound[simplex[0], 0], bound[simplex[0], 1], 'ro')
    plt.plot(bound[simplex[1], 0], bound[simplex[1], 1], 'ro')

    if not is_in_list(hull_points_idx, simplex[0]):
        hull_points_idx.append(simplex[0])
    if not is_in_list(hull_points_idx, simplex[1]):
        hull_points_idx.append(simplex[1])

    plt.axis('equal')

    a, b, c, d = plt.axis()
    plt.axis((a-1, b+1, c-1, d+1))
    plt.show()
    plt.pause(0.0001)

    #raw_input("Press [enter] to continue.")
    print simplex


plt.pause(60)
#print hull_points_idx
hull_points = bound[hull_points_idx]

interesting_points = find_interesting_pairs(bound)


#plt.axis('equal')
#a, b, c, d = plt.axis()
#plt.axis((a-1, b+1, c-1, d+1))
#
#for ps in interesting_points:
#    plt.plot([ps[0][0], ps[1][0]], [ps[0][1], ps[1][1]], 'r-')
#    plt.show()
#    plt.pause(0.0001)
#    raw_input("Press [enter] to continue.")


print hull_points
