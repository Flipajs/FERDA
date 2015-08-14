__author__ = 'fnaiser'

import pickle
import numpy as np
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
from utils.drawing.points import get_contour
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image

import ant_number_svm

WORKING_DIR = '/Users/fnaiser/Documents/chunks'


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

def get_x(r1, r2, pred):
    x = []
    # x_ = ant_number_svm.get_x(r2)
    # p = ant_num_svm.predict_proba([x_])
    # x.append(p[0][0])

    x.append(abs(r1.area() - r2.area()))
    x.append(np.linalg.norm(r1.centroid() - r2.centroid()))
    # d = np.linalg.norm(r1.centroid() + pred - r2.centroid())
    # x.append(np.linalg.norm(r1.centroid() + pred - r2.centroid()))
    t = get_t_diff(r1, r2)
    x.append(t)

    # x.append(d*((2*np.pi - t) / (2*np.pi)))

    # x.append(abs(r1.a_ - r2.a_))
    # x.append(abs(r1.b_ - r2.b_))
    # x.append(abs((r1.a_ / r1.b_) - (r2.a_ / r2.b_)))

    # c1 = len(get_contour(r1.pts()))
    # c2 = len(get_contour(r2.pts()))

    # a = c1/(float(r1.area())**0.5)
    # b = c2/(float(r1.area())**0.5)
    #
    # x.append(abs(a-b))
    # x.append(abs(c1-c2))

    # x.append(similarity_loss(r1, r2))

    # h1 = draw_points_crop_binary(r1.pts())
    # h1 = convex_hull_image(h1)
    # h1 = np.sum(h1)
    #
    # h2 = draw_points_crop_binary(r2.pts())
    # h2 = convex_hull_image(h2)
    # h2 = np.sum(h2)
    #
    # x.append(abs(h1-h2))

    x.append(abs(r1.area() - r2.area()))

    return x

# def get_x(r1, r2, pred):
#     x = []
#     x.append(abs(r1.area() - r2.area()))
#     x.append(np.linalg.norm(r1.centroid() - r2.centroid()))
#     # x.append(np.linalg.norm(r1.centroid() + pred - r2.centroid()))
#     x.append(get_t_diff(r1, r2))
#
#     x.append(abs(r1.a_ - r2.a_))
#     x.append(abs(r1.b_ - r2.b_))
#     x.append(abs((r1.a_ / r1.b_) - (r2.a_ / r2.b_)))
#
#     c1 = len(get_contour(r1.pts()))
#     c2 = len(get_contour(r2.pts()))
#
#     a = c1/(float(r1.area())**0.5)
#     b = c2/(float(r1.area())**0.5)
#
#     x.append(abs(a-b))
#     x.append(abs(c1-c2))
#     # x.append(similarity_loss(r1, r2))
#
#     # h1 = draw_points_crop_binary(r1.pts())
#     # h1 = convex_hull_image(h1)
#     # h1 = np.sum(h1)
#     #
#     # h2 = draw_points_crop_binary(r2.pts())
#     # h2 = convex_hull_image(h2)
#     # h2 = np.sum(h2)
#     #
#     # x.append(abs(h1-h2))
#
#     return x

def get_svm_model():
    chunks = load_chunks()

    num_ = 100
    ids = 8

    X = []
    classes = []

    for i in range(num_):
        for id in range(ids):
            for rest_id in range(ids):
                r1 = chunks[id][i]
                r2 = chunks[rest_id][i+1]

                if i == 0:
                    pred = np.array([0, 0])
                else:
                    pred = r1.centroid() - chunks[id][i-1].centroid()

                X.append(get_x(r1, r2, pred))
                if rest_id == id:
                    classes.append(1)
                else:
                    classes.append(0)

    # get_contour(chunks[1][207])
    # cv2.imshow()

    # frames = [208, 208, 209, 209, 471, 475, 475, 648]
    # ids = [1, 5, 1, 5, 6, 4, 6, 6]
    # amount = 50
    #
    # for f, id in zip(frames, ids):
    #     r1 = chunks[id][f]
    #     r2 = chunks[id][f+1]
    #     pred = r1.centroid() - chunks[id][f-1].centroid()
    #     x1 = get_x(r1, r2, pred)
    #
    #     print x1
    #
    #     for i in range(amount):
    #         X.append(x1)
    #         classes.append(0)

    classes = np.array(classes)

    # X = np.array(X)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ids = classes == 1
    # ax.scatter(X[ids, 0], X[ids, 1], X[ids, 2], c='r')
    # # plt.scatter(X[ids, 0], X[ids, 1], c='r')
    # ids = classes == 0
    # ax.scatter(X[ids, 0], X[ids, 1], X[ids, 2], c='b')
    # # plt.scatter(X[ids, 0], X[ids, 1], c='b')
    # plt.show()


    clf = svm.SVC(kernel='linear', probability=True, class_weight='auto')

    print clf.fit(X, classes)
    print clf.support_vectors_
    print clf.dual_coef_
    print clf.intercept_
    print
    print clf.support_
    print clf.n_support_

    # ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], clf.support_vectors_[:, 2], c='g', s=100, alpha=1.0)
    # plt.show()
    return clf


if __name__ == '__main__':
    ant_num_svm = ant_number_svm.get_svm_model()

    svm_model = get_svm_model()
    chunks = load_chunks()

    r1 = chunks[1][0]
    r2 = chunks[1][1]

    cv2.imshow('r1', np.asarray(draw_points_crop_binary(r1.pts()) * 255, dtype=np.uint8))
    cv2.imshow('r2', np.asarray(draw_points_crop_binary(r2.pts()) * 255, dtype=np.uint8))

    th = np.pi/2
    pts2_ = r2.pts()
    pts2_ -= r2.centroid()
    rot = np.array([[cos(th), -sin(th)],[sin(th), cos(th)]])
    pts2_ = np.dot(r2.pts(), rot)
    pts2_ += r2.centroid()
    pts2_ = np.asarray(pts2_, dtype=np.int32)

    cv2.imshow('r2_', np.asarray(draw_points_crop_binary(pts2_) * 255, dtype=np.uint8))
    # cv2.waitKey(0)

    x1 = get_x(r1, r2, np.array([0, 0]))

    r2.theta_ += np.pi/2
    x2 = get_x(r1, r2, np.array([0, 0]))

    p = svm_model.predict_proba([x1, x2])
    print p



    #
    # chunks = load_chunks()
    #
    # num_ = 100
    # ids = 8
    #
    # svm_model = get_svm_model()
    # for i in range(648, 650):
    #     for id in range(ids):
    #         for rest_id in range(ids):
    #             r1 = chunks[id][i]
    #             r2 = chunks[rest_id][i+1]
    #
    #             if i == 0:
    #                 pred = np.array([0, 0])
    #             else:
    #                 pred = r1.centroid() - chunks[id][i-1].centroid()
    #
    #             p = svm_model.predict_proba([get_x(r1, r2, pred)])
    #             print i, id, rest_id, p[0][1], r1.area(), r2.area()