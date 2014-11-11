__author__ = 'filip@naiser.cz'

import pickle
from numpy.linalg import norm
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import split_by_contours


def load_data():
    dir = os.path.expanduser('~/dump/eight')
    afile = open(dir+"/split_by_cont/695_0.pkl", "rb")
    #afile = open(dir+"/split_by_cont/1143_1.pkl", "rb")
    #afile = open(dir+"/split_by_cont/209_3.pkl", "rb")
    #afile = open(dir+"/split_by_cont/703_0.pkl", "rb")
    pack = pickle.load(afile)
    afile.close()
    ants = pack[0]
    region = pack[1]

    afile = open(dir+"/regions/695.pkl", "rb")
    #afile = open(dir+"/regions/1143.pkl", "rb")
    #afile = open(dir+"/regions/209.pkl", "rb")
    #afile = open(dir+"/regions/703.pkl", "rb")

    regions = pickle.load(afile)
    afile.close()

    region = regions[1]
    #region = regions[12]
    #region = regions[11]
    #region = regions[2]
    points = split_by_contours.get_points(region)

    region = split_by_contours.prepare_region(None, points)

    #
    #afile = open("../out/split_by_cont_eight/28_2.pkl", "rb")
    #pack = pickle.load(afile)
    #afile.close()
    #ants = pack[0]
    #region = pack[1]

    return ants, region


def get_trio_angle(a, b, c):
    l = c - a
    m = b - a

    val = np.dot(l, m) / (norm(l) * norm(m))
    #for case when val is +- 1.0000000001...
    val = max(-1, min(1, val))

    return math.acos(val)


def get_contour_pt(cont, index):
    return cont[index % len(cont)]


def get_trio(cont, index, step):
    a = np.array(get_contour_pt(cont, index))
    b = np.array(get_contour_pt(cont, index - step))
    c = np.array(get_contour_pt(cont, index + step))

    return a, b, c


def show_points(cont, ids, x, y):
    fig = plt.figure()
    xs = []
    ys = []
    plt.ion()
    for i in range(len(cont)):
        xs.append(x[ids[i]])
        ys.append(y[ids[i]])

        plt.close()
        plt.scatter(x, y, color='yellow', s=19, edgecolor='black')
        plt.scatter(xs, ys, color='red', s=35, edgecolor='black')
        plt.axis('equal')
        plt.show()
        plt.waitforbuttonpress(0)


def get_ant_a_id(cont, pt, step):
    pt = np.array([pt.x, pt.y])

    min = 100000
    min_i = -1
    for i in range(len(cont)):
        d = norm(cont[i] - pt)
        if d < min:
            min = d
            min_i = i

    vals = np.zeros(2*step)

    for i in range(2*step):
        a, b, c = get_trio(cont, min_i + i - step, step)
        vals[i] = get_trio_angle(a, b, c)

    id = (np.argmin(vals) + min_i - step) % len(cont)

    #plt.scatter(cont[:, 0], cont[:, 1], color='yellow', s=19, edgecolor='black')
    #plt.scatter(cont[id, 0], cont[id, 1], color='red', s=35, edgecolor='black')
    #plt.axis('equal')
    #plt.show()

    return id


def find_translation_and_rotation(from_pts, to_pts):
    p = np.sum(from_pts, 0) / len(from_pts)
    q = np.sum(to_pts, 0) / len(from_pts)

    #centering
    p1 = from_pts - p
    p2 = to_pts - q

    s = np.dot(p1.transpose(), p2)
    U, _, V = np.linalg.svd(s)

    middle = np.array([[1, 0], [0, np.linalg.det(np.dot(V.T, U.T))]])
    R = np.dot(V.T, middle)
    R = np.dot(R, U.T)

    t = q - np.dot(R, p)

    return t, R


def trans_pts(pts, t, rot):
    pts = np.dot(rot, pts.T)
    pts = pts.T + t

    return pts


def get_sub_contour(cont, index, step):
    pts = np.zeros((step*2, 2))
    j = 0
    for i in range(index-step, index+step):
        pts[j] = cont[i%len(cont)]
        j += 1

    return pts


def get_points_distance(rpts, apts):
    a = rpts-apts
    square = a*a
    s = np.sum(square, 1)
    return np.sum(np.sqrt(s))


def find_best_matching_part(rpts, ants, part_state, step):
    best_ant_id = -1
    best_val = 100000000.
    head = False

    for a in ants:
        is_head = True
        for head_back in [a['head_start'], a['back_start']]:
            #if true means used
            part = 'back'
            if is_head:
                part = 'head'

            if part_state[a['id']][part]:
                continue

            acont = np.array(a['cont'])
            a_index = get_ant_a_id(acont, head_back, step)

            apts = get_sub_contour(acont, a_index, step)
            trans, rot_mat = find_translation_and_rotation(apts, rpts)
            apts = trans_pts(apts, trans, rot_mat)

            score = get_points_distance(rpts, apts)

            if score < best_val:
                best_ant_id = a['id']
                head = is_head
                best_val = score

            is_head = not is_head

    #print best_ant_id, best_val, head


#musi byt zaruceno, ze cont je spravne serazena
def main():
    ants, region = load_data()
    cont = np.array(region['cont'])

    step = 5

    vals = np.zeros(len(cont))
    x = np.zeros(len(cont))
    y = np.zeros(len(cont))

    for i in range(len(cont)):
        a, b, c = get_trio(cont, i, step)

        vals[i] = get_trio_angle(a, b, c)

        center_of_gravity = (b+c) / 2.
        if not split_by_contours.is_inside_region(center_of_gravity, region):
            vals[i] += math.pi

        x[i] = cont[i][0]
        y[i] = cont[i][1]

    #plt.scatter(cont[:, 0], cont[:, 1], c=vals, s=45, cmap=plt.cm.jet)
    #plt.axis('equal')
    #plt.show()
    #
    #return

    ids = np.argsort(vals)
    ra_id = ids[2]
    rpts = get_sub_contour(cont, ra_id, step)


    start = time.time()

    part_state = {}
    for a in ants:
        part_state[a['id']] = {'head': False, 'back': False}

    find_best_matching_part(rpts, ants, part_state, step)
    end = time.time()
    print end - start


    a = ants[1]


    acont = np.array(a['cont'])
    aa_id = get_ant_a_id(acont, a['head_start'], step)

    pts2 = get_sub_contour(acont, aa_id, step)

    #plt.scatter(acont[:, 0], acont[:, 1], color='red', s=45, edgecolor='black')
    #plt.scatter(cont[:, 0], cont[:, 1], color='yellow', s=15, edgecolor='black')

    plt.scatter(cont[:, 0], cont[:, 1], color='black', s=45)
    plt.scatter(rpts[:, 0], rpts[:, 1], color='red', s=45, edgecolor='black')
    plt.scatter(pts2[:, 0], pts2[:, 1], color='yellow', s=15, edgecolor='black')

    trans, rot_mat = find_translation_and_rotation(pts2, rpts)
    acont = trans_pts(acont, trans, rot_mat)


    plt.scatter(acont[:, 0], acont[:, 1], color='green', s=35, edgecolor='black')
    plt.axis('equal')
    plt.show()


    print trans, rot_mat


if __name__ == '__main__':
    main()