from features import get_curvature_kp
from core.project.project import Project
from utils.video_manager import get_auto_video_manager
from matplotlib import pyplot as plt
import cv2
import numpy as np
from core.region.mser import ferda_filtered_msers
import scipy.ndimage as ndimage
from utils.geometry import rotate
from scipy.spatial.distance import cdist


def data_cam2():
    #Cam2
    collisions = {
        1: {'s': [6, 7], 'm': 1, 'e': [18, 23]},
        2: {'s': [64, 65], 'm': 50, 'e': [48, 62]},
        3: {'s': [111, 112, 120], 'm': 132, 'e': [123, 124, 116]},
        394: {'s': [60, 61], 'm': 49, 'e': [77, 78]},
        1985: {'s': [306, 307], 'm': 302, 'e': [297, 305]},
        3130: {'s': [425, 411], 'm': 420, 'e': [421, 422]},
        3350: {'s': [464, 460], 'm': 457, 'e': [452, 467]}
    }

    return collisions


def __get_rts(a1, a2, b1, b2):
    a_ = a2 - a1
    a_n = np.linalg.norm(a_)
    b_ = b2 - b1
    b_n = np.linalg.norm(b_)

    if a_n == 0 or b_n == 0:
        return None, None, None, None

    t = (b1+b2) / 2 - (a1+a2) / 2
    s = b_n / a_n

    x_ = np.dot(a_.T, b_) / (a_n * b_n)
    # numerical errors fix
    x_ = min(1, max(-1, x_))

    from math import acos
    theta = acos(x_)

    # compute the orientation
    # http://math.stackexchange.com/questions/317874/calculate-the-angle-between-two-vectors

    U = np.array([[a_[0], b_[0]], [a_[1], b_[1]]])
    if np.linalg.det(U) < 0:
        theta = -theta % 2*np.pi

    return t, theta, s, (a1+a2) / 2

def __transform_pts(pts, r, t, rot_center):
    pts = np.array(rotate(pts, r, rot_center))

    pts[:, 0] = pts[:, 0] + t[0]
    pts[:, 1] = pts[:, 1] + t[1]

    return pts

def get_geom_s(start_, coef, num):
    r = []
    for i in range(num):
        start_ = start_ * coef
        r.append(start_)

    return r

def __get_support(pts1, p_type_starts1, pts2, p_type_starts2, type_weights, r, t, rot_center, intensities=None, thresh=1):
    pts1 =__transform_pts(pts1, r, t, rot_center)

    # threshs = get_geom_s(2, 2**0.5, 5)
    supp = 0
    # -1 because, there is last number describing the end of interval in p_type_starts...
    for c in range(len(p_type_starts1)-1):
        ids1_ = slice(p_type_starts1[c], p_type_starts1[c+1])
        ids2_ = slice(p_type_starts2[c], p_type_starts2[c+1])

        d = cdist(pts1[ids1_, :], pts2[ids2_, :])
        mins_ = np.min(d, axis=1)
        amins_ = np.argmin(d, axis=1)

        mins_ = mins_**2
        mins_[mins_ > thresh] = thresh

        supp += np.sum(mins_) * type_weights[c]
        if intensities is not None:
            int2 = intensities[1][ids2_]
            int_diff = intensities[0][ids1_] - int2[amins_]
            int_diff = abs(np.asarray(int_diff[mins_ < thresh], dtype=np.float) / 10)

            supp += sum(int_diff) * type_weights[c]

        c = 50
        if False:
            supp += np.linalg.norm(t)**1.2 + c*r


    return supp

def estimate_rt(kps1, kps2):
    from numpy import random
    random.seed(19)

    p_type1 = []
    type_starts1 = []
    pts1 = []
    angles1 = []
    intensities1 = []
    si = 0

    for a in sorted([int(x) for x in kps1]):
        type_starts1.append(len(pts1))
        for b in kps1[a]:
            pts1.append(b['point'])
            angles1.append(b['angle'])
            intensities1.append(b['intensity'])
            p_type1.append(si)

        si += 1

    type_starts1.append(len(pts1))
    intensities1 = np.array(intensities1, dtype=np.int32)

    pts1 = np.array(pts1)

    type_starts2 = []
    pts2 = []
    angles2 = []
    intensities2 = []
    for a in sorted([int(x) for x in kps2]):
        type_starts2.append(len(pts2))
        for b in kps2[a]:
            pts2.append(b['point'])
            angles2.append(b['angle'])
            intensities2.append(b['intensity'])

    intensities2 = np.array(intensities2, dtype=np.int32)
    type_starts2.append(len(pts2))

    pts2 = np.array(pts2)


    max_steps = 10000
    num_trials = 500

    best_t = []
    best_r = []
    best_rot_center = []
    best_supp = [np.inf]

    trials = 0
    for i in range(max_steps):
        ai = random.randint(len(pts1), size=2)
        s_ = p_type1[ai[0]]
        bi1 = random.randint(type_starts2[s_], type_starts2[s_+1])
        s_ = p_type1[ai[1]]
        bi2 = random.randint(type_starts2[s_], type_starts2[s_+1])

        pa1 = pts1[ai[0], :]
        pa2 = pts1[ai[1], :]
        pb1 = pts2[bi1, :]
        pb2 = pts2[bi2, :]

        # test if they are reasonable pairs
        if abs(np.linalg.norm(pa1-pa2) - np.linalg.norm(pb1-pb2)) > 5:
            continue

        t, r, s, rot_center = __get_rts(pa1, pa2, pb1, pb2)

        if t is None:
            continue

        trials += 1

        # type_weights = [0.2, 0.35, 0.7, 1.3, 2]
        type_weights = [1, 2]
        # type_weights = [1, 1, 1, 1, 1]
        supp = __get_support(pts1, type_starts1, pts2, type_starts2, type_weights, r, t, rot_center, intensities=(intensities1, intensities2), thresh=5)

        j = 0
        while j < len(best_supp) and supp < best_supp[j]:
            j += 1

        if j > 0:
            best_r.insert(j, r)
            best_t.insert(j, t)
            best_rot_center.insert(j, rot_center)
            best_supp.insert(j, supp)

        if trials >= num_trials:
            break

    print "SKIPPED: ", i - num_trials

    return best_t, best_r, best_rot_center, best_supp


if __name__ == '__main__':
    p = Project()
    data = data_cam2()
    name = 'Cam2/cam2.fproj'
    wd = '/Users/flipajs/Documents/wd/gt/'
    p.load(wd+name)
    vm = get_auto_video_manager(p)

    d = data[1985]

    rs1 = p.gm.region(p.chm[d['s'][0]].end_vertex_id())
    rs2 = p.gm.region(p.chm[d['s'][1]].end_vertex_id())
    # rs3 = p.gm.region(p.chm[d['s'][2]].end_vertex_id())

    im = vm.get_frame(rs1.frame())
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    # plt.waitforbuttonpress()
    #
    plt.ion()

    # kps1 = get_curvature_kp(rs1.contour_without_holes(), True)
    # kps2 = get_curvature_kp(rs2.contour_without_holes(), True)
    # kps3 = get_curvature_kp(rs3.contour_without_holes(), True)


    from core.graph.region_chunk import RegionChunk
    rch = RegionChunk(p.chm[d['m']], p.gm, p.rm)

    # r = p.gm.region(p.chm[d['m']].start_vertex_id())
    r = rch[1]
    # kpsm = get_curvature_kp(r.contour_without_holes(), True)

    step = 5

    rs__ = rs1
    test1 = {0: [], 1:[]}
    pts1 = []
    pts__ = rs__.pts()
    pts__ = pts__[np.random.randint(len(pts__), size=len(pts__)/step), :]

    for i in range(len(pts__)):
        # if i % step == 0:
            p = pts__[i, :]
            test1[0].append({'point': p, 'angle': 0, 'intensity': gray[p[0], p[1]]})
            pts1.append(p)

    pts__ = rs__.contour_without_holes()
    for i in range(len(pts__)):
        if i % step == 0:
            p = pts__[i, :]
            test1[1].append({'point': p, 'angle': 0, 'intensity': gray[p[0], p[1]]})

    im = vm.get_frame(r.frame())
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    pts1 = np.array(pts1)

    testm = {0: [], 1: []}
    ptsm = []
    pts__ = r.pts()
    pts__ = pts__[np.random.randint(len(pts__), size=len(pts__)/step), :]
    for i in range(len(pts__)):
        # if i % step == 0:
            p = pts__[i, :]
            testm[0].append({'point': p, 'angle': 0, 'intensity': gray[p[0], p[1]]})
            ptsm.append(p)

    pts__ = r.contour_without_holes()
    for i in range(len(pts__)):
        if i % step == 0:
            p = pts__[i, :]
            testm[1].append({'point': p, 'angle': 0, 'intensity': gray[p[0], p[1]]})

    ptsm = np.array(ptsm)

    best_t, best_r, best_rot_center, best_supp = estimate_rt(test1, testm)

    plt.figure()
    plt.scatter(ptsm[:, 1], ptsm[:, 0], c='k', s=30, alpha=.70)
    plt.hold(True)
    plt.scatter(pts1[:, 1], pts1[:, 0], c='r', s=30, alpha=.20)

    cs = ['g', 'b', 'c', 'm', 'k', 'w', 'y']
    # pts__ = []
    # for a in kps1:
    #     for b in kps1[a]:
    #         pts__.append(b['point'])

    for i in reversed(xrange(len(best_t))):
        print i
        print best_supp[i+1]
        pts_ = __transform_pts(pts1, best_r[i], best_t[i], best_rot_center[i])
        plt.hold(True)
        plt.scatter(pts_[:, 1], pts_[:, 0], c=cs[i%len(cs)], s=100, alpha=0.4)
        plt.hold(False)

        plt.axis('equal')

        plt.show()
        plt.waitforbuttonpress()