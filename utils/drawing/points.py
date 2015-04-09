__author__ = 'fnaiser'

import pickle
import cv2
import numpy as np
from core.region import mser, mser_operations
from utils.img import get_safe_selection, get_roi
from core.settings import Settings as S_
from PyQt4 import QtGui


def process_color(c):
    """
    Returns np.array in format [B G R alpha], where BGR are in range <0, 255> and alpha is in range <0, 1>
    :param c:
    :return:
    """

    if isinstance(c, QtGui.QColor):
        alpha = c.alpha() / 255.0
        c = np.array([c.blue(), c.green(), c.red(), alpha])
    elif isinstance(c, tuple):
        c = np.array(c)

    return c


def draw_points(img, pts, color=None):
    if pts.size == 0:
        return img

    if not color:
        color = S_.visualization.default_region_color

    color = process_color(color)
    alpha = color[3]
    c = color[:3]

    valid_ids1 = np.logical_and(pts[:, 0] > 0, pts[:,0] < img.shape[0])
    valid_ids2 = np.logical_and(pts[:, 1] > 0, pts[:,1] < img.shape[1])
    ids = np.logical_and(valid_ids1, valid_ids2)
    img[pts[ids, 0], pts[ids, 1], :] = alpha * c + (1 - alpha) * img[pts[ids, 0], pts[ids, 1], :]

    return img


def draw_points_crop(img, pts, color=None, margin=0.1, square=False):
    """
    returns image with region visualization cropped around region with margin which is specified by percentage of max(height, width)
    :param img:
    :param region:
    :param color:
    :param margin:
    :return:
    """

    if pts.size == 0:
        return img

    roi = get_roi(pts)

    width = roi.width()
    height = roi.height()
    if square:
        a = max(roi.height(), roi.width())
        height = a
        width = a

    m_ = max(width, height)
    margin = m_ * margin

    y_ = roi.y() - margin
    x_ = roi.x() - margin
    height_ = height + 2 * margin
    width_ = width + 2 * margin

    im_ = np.copy(img)
    im_ = draw_points(im_, pts, color)

    crop = get_safe_selection(im_, y_, x_, height_, width_)

    return crop


def get_contour(pts):
    """
    returns np.array of [y,x] positions of inner contour for given points

    :param pts:
    :return:
    """

    roi = get_roi(pts)

    img = np.zeros((roi.height(), roi.width()), dtype=np.uint8)


    img[pts[:,0]-roi.y(), pts[:,1]-roi.x()] = 255

    ret, thresh = cv2.threshold(img, 127, 255, 0)

    # different versions of opencv... =/
    try:
        # TODO: CV_RETR_CCOMP and skip holes... http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    except:
        try:
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        except:
            pass

    cont = np.array([])
    for c in contours:
        (rows, _, _) = c.shape
        c = np.reshape(c, (rows, 2))

        c[:, [0, 1]] = c[:, [1, 0]]

        if cont.size > 0:
            cont = np.append(cont, c, axis=0)
        else:
            cont = c

    if cont.size > 0:
        cont += np.array([roi.y(), roi.x()])

    return cont



if __name__ == '__main__':
    with open('/Users/fnaiser/Documents/colormarktests/regions/1.pkl', 'rb') as f:
        regions = pickle.load(f)

    img = cv2.imread('/Users/fnaiser/Documents/colormarktests/imgs/0.png')


    regions = mser.get_msers_(img)
    groups = mser_operations.get_region_groups(regions)
    idx = mser_operations.margin_filter(regions, groups)

    for id in idx:
        # img = draw_points(img, regions[id].pts())
        cont = get_contour(regions[id].pts())

        img = draw_points(img, cont, color=(0, 0, 255, 0.5))

    #
    # cont = get_contour(regions[idx[0]].pts())
    # cont_img = draw_points_crop(img, cont)

    cv2.imshow('img', img)
    cv2.moveWindow('img', 0, 0)
    # cv2.imshow('cont', cont_img)
    # cv2.moveWindow('cont', 0, 0)
    cv2.waitKey(0)