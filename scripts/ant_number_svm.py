__author__ = 'fnaiser'

import pickle
import numpy as np
from utils.video_manager import get_auto_video_manager
from utils.drawing.points import draw_points, draw_points_crop, draw_points_crop_binary
import cv2
from math import sin, cos
from PyQt4 import QtGui, QtCore
import sys
from gui.img_grid.img_grid_widget import ImgGridWidget
from gui.gui_utils import get_image_label
from core.region.mser import get_msers_
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
from scripts.similarity_test import similarity_loss
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats.stats import pearsonr
from core.region import region
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis, convex_hull_image, binary_closing, binary_opening
from core.region.mser import get_msers_, get_all_msers
from skimage.transform import resize
from gui.img_grid.img_grid_dialog import ImgGridDialog
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
import warnings
import os
from skimage.transform import resize
from core.settings import Settings as S_


WORKING_DIR = '/Users/fnaiser/Documents/chunks'
vid_path = '/Users/fnaiser/Documents/chunks/eight.m4v'
working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/eight'
AVG_AREA = 252.1
AVG_MARGIN = 25.2

def load_chunks():
    with open(WORKING_DIR+'/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)

    return chunks

def get_t_diff(r1, r2):
    t1 = r1.theta_
    t2 = r2.theta_
    if t1 < 0:
        t1 += np.pi
    if t2 < 0:
        t2 += np.pi

    t_ = max(t1, t2) - min(t1, t2)

    if t_ > np.pi/2:
        t_ = np.pi - t_

    return t_

def poly_area(p):
    return 0.5 * abs(sum((x0*y1 - x1*y0)**0.5
                         for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])

def area_np(p):
    x = p[:, 0]
    y = p[:, 1]
    n = p.shape[0]

    shift_up = np.arange(-n+1, 1)
    shift_down = np.arange(-1, n-1)

    return (x * (y.take(shift_up) - y.take(shift_down))).sum() / 2.0

def get_x(r, AVG_AREA, AVG_MARGIN):
    x = []
    x.append(r.area() / float(AVG_AREA))
    x.append(r.ellipse_major_axis_length() / r.ellipse_minor_axis_length())

    c1 = len(r.contour())
    roi_ = r.roi()

    if roi_.height() < 4 or roi_.width() < 4 or r.contour().shape[0] < 5:
        h_area = r.area()
    else:
        try:
            h_ = ConvexHull(r.contour())
            h_area = area_np(r.contour()[h_.vertices, :])
        except Exception as e:
            h_area = r.area()
            pass

    x.append(h_area / float(AVG_AREA))
    x.append((r.margin_-AVG_MARGIN)**2)
    x.append(c1 / (r.area()**0.5))

    return x

def get_svm_model():
    chunks = load_chunks()
    merged = [(1, 209), (6, 472), (4, 473), (4, 474), (4, 475), (5, 635), (5, 636), (5, 638), (4, 649), (4, 650), (4, 651), (4, 652), (4, 653), (1, 654), (1, 655), (1, 656), (1, 657), (1, 658), (5, 673), (1, 678), (1, 679), (4, 680)]

    X = []
    classes = []
    for (id, frame) in merged:
        X.append(get_x(chunks[id][frame], AVG_AREA, AVG_MARGIN))
        classes.append(1)

    for frame in range(20):
        for id in range(8):
            X.append(get_x(chunks[id][frame], AVG_AREA, AVG_MARGIN))
            classes.append(0)

    # X = np.array(X)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[:len(merged), 0], X[:len(merged), 1], X[:len(merged), 2], c='r')
    # ax.scatter(X[len(merged):, 0], X[len(merged):, 1], X[len(merged):, 2], c='b')
    # plt.show()

    clf = svm.SVC(kernel='linear', probability=True)

    print clf.fit(X, classes)
    print clf.support_vectors_
    print clf.support_
    print clf.n_support_

    return clf


def select_msers_cached(frame, use_area_filter=True):
    msers = get_all_msers(frame, vid_path, working_dir)

    # img = vid.seek_frame(frame)
    # img = np.asarray(resize(img, (img.shape[0] / 2, img.shape[1] / 2)) * 255, dtype=np.uint8)
    #
    # msers = get_msers_(img)

    groups = get_region_groups(msers)
    ids = margin_filter(msers, groups)
    ids = children_filter(msers, ids)

    return [msers[i] for i in ids]


def get_svm_model2():
    merged = set([9250, 14377, 15271, 8983, 15584, 9390, 15521, 9261, 19126, 2812, 18148, 10079, 10632, 14303, 8931, 9674, 15119, 15584, 15028, 8971, 10607, 18925, 15596, 15182, 15170, 8520, 95264645, 9300, 9312, 8086, 9473, 6357, 9335, 7775, 6049, 8174, 7743, 15028, 9097, 8723, 8735, 15205, 15218, 15482, 8711, 8558, 4645, 10607, 8983, 8971, 10211, 18123, 9090, 9102, 15482, 7775, 19126, 19273, 6102, 2812, 9322, 9335, 10691, 7743, 9008, 18992, 15763, 8758, 8770, 15877, 15271, 6394, 15230, 9539, 9560, 19273, 9570, 9581, 8846, 8832, 9102, 9113, 10665, 10678, 10665, 10654, 10678, 10691, 10654, 10644])
    with open('/Volumes/Seagate Expansion Drive/mser_svm/eight/certainty_visu.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        g = up.load()
        ccs = up.load()
        vid_path = up.load()

    X = []
    classes = []

    used_ids = {}
    for cc in ccs:
        for r in (cc['c1'] + cc['c2']):
            if r not in used_ids:
                used_ids[r] = True
                if r.id_ in merged:
                    classes.append(1)
                else:
                    classes.append(0)

                X.append(get_x(r, AVG_AREA, AVG_MARGIN))

    # X = np.array(X)
    # classes = np.array(classes)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ids = classes == 0
    # ax.scatter(X[ids, 0], X[ids, 1], X[ids, 2], c='r')
    # ids = classes == 1
    # ax.scatter(X[ids, 0], X[ids, 1], X[ids, 2], c='b')
    # plt.show()

    clf = svm.SVC(kernel='linear', probability=True)

    print clf.fit(X, classes)
    print clf.support_vectors_
    print clf.support_
    print clf.n_support_

    return clf

def test():
    app = QtGui.QApplication(sys.argv)

    dial = ImgGridDialog()
    dial.img_grid.reshape(15, element_width=100)
    vid = get_auto_video_manager('/Users/fnaiser/Documents/smallLense_colony1_1min.avi')
    im = vid.next_frame()
    im = vid.next_frame()
    im = vid.next_frame()
    im = vid.next_frame()

    msers = get_msers_(im)
    groups = get_region_groups(msers)
    ids = margin_filter(msers, groups)
    for id in ids:
        r = msers[id]
        vis = draw_points_crop(im.copy(), r.pts(), square=True)

        iml = get_image_label(vis)
        dial.img_grid.add_item(iml)

    dial.showMaximized()
    app.exec_()
    sys.exit()


if __name__ == '__main__':
    test()


    clf = get_svm_model2()

    with open('/Volumes/Seagate Expansion Drive/mser_svm/biglenses2/certainty_visu.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        g = up.load()
        ccs = up.load()
        vid_path = up.load()


    areas = []
    major_axes = []
    margins = []

    used_ids = {}
    for cc in ccs:
        for r in (cc['c1'] + cc['c2']):
            if r not in used_ids:
                used_ids[r] = True
                areas.append(r.area())
                major_axes.append(r.ellipse_major_axis_length() * 2)
                margins.append(r.margin_)

    areas = np.array(areas)
    major_axes = np.array(major_axes)
    margins = np.array(margins)

    AVG_AREA = np.median(areas)
    AVG_MAIN_A = np.median(major_axes)
    AVG_MARGIN = np.median(margins)
    print "AVG: ", AVG_AREA, AVG_MAIN_A, AVG_MARGIN


    app = QtGui.QApplication(sys.argv)

    dial = ImgGridDialog()
    dial.img_grid.reshape(15, element_width=100)
    vid = get_auto_video_manager(vid_path)

    used_ids = {}
    for cc in ccs:
        for r in (cc['c1'] + cc['c2']):
            if r not in used_ids:
                used_ids[r] = True

                vis = g.node[r]['img']

                prob = clf.predict_proba([get_x(r, AVG_AREA, AVG_MARGIN)])
                r_ = int(255*prob[0][0])
                g_ = int(255*prob[0][1])

                p = prob[0][1]
                if p < 0.001:
                    p = 0

                margin = 3
                h_, w_, _ = vis.shape
                im = np.zeros((h_ + 2*margin, w_ + 2*margin, 3), dtype=np.uint8)
                im[:,:,2] = r_
                im[:,:,1] = g_

                im[margin:margin+h_, margin:margin+w_, :] = vis
                vis = im

                cv2.putText(vis, str(p)[0:5], (1, 22), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 255, 100), 1, cv2.cv.CV_AA)

                iml = get_image_label(vis)
                dial.img_grid.add_item(iml)

    dial.showMaximized()
    app.exec_()
    sys.exit()

    # chunks = load_chunks()
    # for id in range(8):
    #     pts = draw_points_crop_binary(chunks[id][673].pts())
    #     cv2.imshow('test', np.asarray(pts*255, dtype=np.uint8))
    #     cv2.waitKey(0)
    #
    # vid = get_auto_video_manager(vid_path)
    #
    # from core.antlikeness import Antlikeness
    # from configs.colormarks2 import *
    # antlikeness = Antlikeness()
    # f_regions = {}
    # for f in range(init_frames):
    #     f_regions[f] = select_msers_cached(f)
    #
    # antlikeness.learn(f_regions, classes)
    #
    # i = 0
    # areas = []
    # major_axes = []
    # margins = []
    #
    # for f in f_regions:
    #     for r in f_regions[f]:
    #         if classes[i]:
    #             areas.append(r.area())
    #             major_axes.append(r.ellipse_major_axis_length() * 2)
    #             margins.append(r.margin_)
    #         i += 1
    #
    # areas = np.array(areas)
    # major_axes = np.array(major_axes)
    # margins = np.array(margins)
    #
    # AVG_AREA = np.median(areas)
    # AVG_MAIN_A = np.median(major_axes)
    # AVG_MARGIN = np.median(margins)
    # print "AVG: ", AVG_AREA, AVG_MAIN_A, AVG_MARGIN
    #
    #
    # clf = get_svm_model()
    #
    #
    # chunks = load_chunks()
    # app = QtGui.QApplication(sys.argv)
    #
    # dial = ImgGridDialog()
    # dial.img_grid.reshape(15, element_width=70)
    # vid = get_auto_video_manager(vid_path)
    #
    # # for f in range(200, 220) + range(460, 480) + range(630, 700):
    # for f in range(30):
    #     print f
    #     # im = vid.next_frame()
    #     im = vid.seek_frame(f)
    #     msers = select_msers_cached(f)
    #
    #     im2 = np.copy(im)
    #     im_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    #     detected = 0
    #     for m in msers:
    #         x = get_x(m, AVG_AREA, AVG_MARGIN)
    #
    #         cont = m.contour()
    #
    #         prob_prototype_represantion_being_same_id_set = clf.predict_proba([x])
    #         r_ = int(255*prob_prototype_represantion_being_same_id_set[0][0])
    #         g_ = int(255*prob_prototype_represantion_being_same_id_set[0][1])
    #
    #         a_ = antlikeness.get_prob(m)
    #
    #         if prob_prototype_represantion_being_same_id_set[0][1] < 0.1 or a_[1] < 0.2:
    #             continue
    #
    #         vis = draw_points_crop(im.copy(), cont, color=(0, g_, r_, 1), square=True)
    #         # im2 = draw_points(im2, cont, color=(0, g_, r_, 1))
    #
    #         vis = np.asarray(resize(vis, (70, 70)) * 255, dtype=np.uint8)
    #         cv2.putText(vis, str(f)+' '+str(prob_prototype_represantion_being_same_id_set[0][1])[0:5], (1, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 255, 100), 1, cv2.cv.CV_AA)
    #
    #         iml = get_image_label(vis)
    #         dial.img_grid.add_item(iml)
    #
    #     # cv2.imshow('test', im2)
    #     # cv2.imwrite(working_dir+'/'+str(f)+'.jpg', im2)
    #     # cv2.waitKey(0)
    #
    # dial.showMaximized()
    # app.exec_()
    # sys.exit()