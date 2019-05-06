from features import get_curvature_kp
from core.project.project import Project
from utils.video_manager import get_auto_video_manager, VideoManager
from matplotlib import pyplot as plt
import cv2
import numpy as np
from core.region.mser import get_filtered_regions
import scipy.ndimage as ndimage
from utils.geometry import rotate
from scipy.spatial.distance import cdist
import cPickle as pickle


def data_cam2():
    #Cam2
    collisions = [
        {'s': [6, 7], 'm': 1, 'e': [18, 23]},
        {'s': [64, 65], 'm': 50, 'e': [48, 62]},
        {'s': [111, 112, 120], 'm': 132, 'e': [123, 124, 116]},
                  ]

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

def __get_support(pts1, p_type_starts1, pts2, p_type_starts2, type_weights, r, t, rot_center, thresh=1):
    pts1 =__transform_pts(pts1, r, t, rot_center)

    # threshs = get_geom_s(2, 2**0.5, 5)
    supp = 0
    # -1 because, there is last number describing the end of interval in p_type_starts...
    for c in range(len(p_type_starts1)-1):
        d = cdist(pts1[p_type_starts1[c]:p_type_starts1[c+1], :], pts2[p_type_starts2[c]:p_type_starts2[c+1], :])
        mins_ = np.min(d, axis=1)

        if True:
            mins_ = mins_**2 - thresh**2
            mins_[mins_ > 0] = 0
            supp += type_weights[c] * -sum(mins_)
        else:
            supp += type_weights[c] * np.sum(mins_ < thresh)
# supp += np.sum(mins_ < threshs[c])

    return supp

def estimate_rt(kps1, kps2, thresh=3):
    from numpy import random
    random.seed(19)

    p_type1 = []
    type_starts1 = []
    pts1 = []
    si = 0

    for a in sorted([int(x) for x in kps1]):
        type_starts1.append(len(pts1))
        for b in kps1[a]:
            pts1.append(b['point'])
            p_type1.append(si)

        si += 1

    type_starts1.append(len(pts1))

    pts1 = np.array(pts1)

    type_starts2 = []
    pts2 = []
    for a in sorted([int(x) for x in kps2]):
        type_starts2.append(len(pts2))
        for b in kps2[a]:
            pts2.append(b['point'])

    type_starts2.append(len(pts2))

    pts2 = np.array(pts2)


    steps = 5000

    best_t = []
    best_r = []
    best_rot_center = []
    best_supp = [0]

    for i in range(steps):
        ai = random.randint(len(pts1), size=2)
        s_ = p_type1[ai[0]]
        bi1 = random.randint(type_starts2[s_], type_starts2[s_+1])
        s_ = p_type1[ai[1]]
        bi2 = random.randint(type_starts2[s_], type_starts2[s_+1])

        t, r, s, rot_center = __get_rts(pts1[ai[0], :], pts1[ai[1], :], pts2[bi1, :], pts2[bi2, :])

        if t is None:
            continue

        # type_weights = [0.2, 0.35, 0.7, 1.3, 2]
        type_weights = [1, 1, 1, 1, 1]
        supp = __get_support(pts1, type_starts1, pts2, type_starts2, type_weights, r, t, rot_center, thresh=thresh)

        i = 0
        while i < len(best_supp) and supp > best_supp[i]:
            i += 1

        if i > 0:
            best_r.insert(i, r)
            best_t.insert(i, t)
            best_rot_center.insert(i, rot_center)
            best_supp.insert(i, supp)


    return best_t, best_r, best_rot_center, best_supp

def __pt_dict(pts):
    d = {0: []}
    for i in range(len(pts)):
        p = pts[i, :]

        # t = 0 if i < 2 else 1
        t = 0

        d[t].append({'point': p})

    return d

def __merge_pts_from_dict(d):
    pts = []
    for aid in d:
        pts.extend(d[aid])

    return pts

if __name__ == '__main__':
    # p = Project()
    # # data = data_cam2()
    # name = 'Cam1/cam1.fproj'
    # wd = '/Users/flipajs/Documents/wd/gt/'
    # p.load(wd+name)
    vm = VideoManager('/Users/flipajs/Documents/wd/Cam1_clip.avi')
    # vm = get_auto_video_manager(p)


    with open('/Users/flipajs/Documents/wd/antennas_gt/Cam1/50.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        data = up.load()

    frames = sorted([f for f in data])

    pts1 = np.array(data[frames[0]][1])
    ptsm = np.array(__merge_pts_from_dict(data[frames[1]]))

    best_t, best_r, best_rot_center, best_supp = estimate_rt(__pt_dict(pts1), __pt_dict(ptsm), thresh=10)

    plt.ion()
    cs = ['g', 'b', 'c', 'm', 'w', 'y']
    plt.figure()
    plt.scatter(ptsm[:, 0], ptsm[:, 1], c='k', s=30, alpha=.90)
    plt.hold(True)
    # plt.scatter(pts1[:, 0], pts1[:, 1], c='r', s=30, alpha=.70)
    for i in reversed(xrange(len(best_t))):
        plt.cla()
        plt.scatter(ptsm[:, 0], ptsm[:, 1], c='k', s=30, alpha=.90)

        print i
        print best_supp[i+1]
        pts_ = __transform_pts(pts1, best_r[i], best_t[i], best_rot_center[i])

        if pts_[0, 1] > 365 or pts_[0, 0] > 180:
            continue

        plt.hold(True)
        plt.scatter(pts_[:, 0], pts_[:, 1], c=cs[i%len(cs)], s=30, alpha=0.6)
        plt.hold(False)

        plt.show()
        plt.waitforbuttonpress()

    # d = data[2]
    #
    # rs1 = p.gm.region(p.chm[d['s'][0]].end_vertex_id())
    # rs2 = p.gm.region(p.chm[d['s'][1]].end_vertex_id())
    # rs3 = p.gm.region(p.chm[d['s'][2]].end_vertex_id())
    #
    # plt.ion()
    # #
    # # im = vm.get_frame(rs1.frame())
    # # plt.figure()
    # # plt.imshow(im)
    # # plt.show()
    # # plt.waitforbuttonpress()
    # #
    # # im = vm.get_frame(rs1.frame() + 1)
    # # plt.figure()
    # # plt.imshow(im)
    # # plt.show()
    # # plt.waitforbuttonpress()
    #
    #
    # kps1 = get_curvature_kp(rs1.contour_without_holes(), True)
    # kps2 = get_curvature_kp(rs2.contour_without_holes(), True)
    # kps3 = get_curvature_kp(rs3.contour_without_holes(), True)
    #
    #
    # r = p.gm.region(p.chm[d['m']].start_vertex_id())
    # kpsm = get_curvature_kp(r.contour_without_holes(), True)
    #
    #
    # step = 5
    #
    # test1 = {0: []}
    # pts1 = []
    # pts__ = rs1.pts()
    # pts__ = pts__[np.random.randint(len(pts__), size=len(pts__)/step), :]
    # for i in range(len(pts__)):
    #     # if i % step == 0:
    #         p = pts__[i, :]
    #         test1[0].append({'point': p, 'angle': 0})
    #         pts1.append(p)
    # pts1 = np.array(pts1)
    #
    # testm = {0: []}
    # ptsm = []
    # pts__ = r.pts()
    # pts__ = pts__[np.random.randint(len(pts__), size=len(pts__)/step), :]
    # for i in range(len(pts__)):
    #     # if i % step == 0:
    #         p = pts__[i, :]
    #         testm[0].append({'point': p, 'angle': 0})
    #         ptsm.append(p)
    #
    # ptsm = np.array(ptsm)
    #
    # best_t, best_r, best_rot_center, best_supp = estimate_rt(test1, testm)
    #
    # plt.figure()
    # plt.scatter(ptsm[:, 0], ptsm[:, 1], c='k', s=30, alpha=.70)
    # plt.hold(True)
    # plt.scatter(pts1[:, 0], pts1[:, 1], c='r', s=30, alpha=.70)
    #
    # cs = ['g', 'b', 'c', 'm', 'k', 'w', 'y']
    # # pts__ = []
    # # for a in kps1:
    # #     for b in kps1[a]:
    # #         pts__.append(b['point'])
    #
    # for i in reversed(xrange(len(best_t))):
    #     print i
    #     print best_supp[i+1]
    #     pts_ = __transform_pts(pts1, best_r[i], best_t[i], best_rot_center[i])
    #     plt.hold(True)
    #     plt.scatter(pts_[:, 0], pts_[:, 1], c=cs[i%len(cs)], s=30, alpha=0.4)
    #     plt.hold(False)
    #
    #     plt.show()
    #     plt.waitforbuttonpress()