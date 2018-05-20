__author__ = 'fnaiser'

import pickle
import numpy as np
from utils.video_manager import get_auto_video_manager
from utils.drawing.points import draw_points, draw_points_crop
import cv2
from math import sin, cos
from PyQt4 import QtGui, QtCore
import sys
from gui.img_grid.img_grid_widget import ImgGridWidget
from gui.gui_utils import get_image_label
from core.region.mser import get_msers_img
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter

WORKING_DIR = '/Users/fnaiser/Documents/chunks'

def load_chunks():
    with open(WORKING_DIR+'/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)

    return chunks


def transform_pts(r1, r2):
    pts2 = r2.pts().copy()

    pts2 -= r2.centroid()
    th = r2.theta_ - r1.theta_
    if abs(th) > np.pi/2:
        th = th + np.pi

    rot = np.array([[cos(th), -sin(th)],[sin(th), cos(th)]])
    pts2 = np.dot(pts2, rot)

    pts2 += r1.centroid()

    return pts2


def get_intersection(pts1, pts2):
    s1 = set(map(tuple, pts1))
    s2 = set(map(tuple, np.asarray(pts2, dtype=np.int32)))

    s = s1 & s2
    inter_len = len(s)

    return (len(s1) - inter_len + len(s2) - inter_len) / float(len(s1))


def similarity_loss(r1, r2):
    pts2 = transform_pts(r1, r2)
    s = get_intersection(r1.pts(), pts2)

    return s


if __name__ == '__main__':
    # chunks = load_chunks()
    vid = get_auto_video_manager(WORKING_DIR+'/eight.m4v')

    app = QtGui.QApplication(sys.argv)
    w = ImgGridWidget()
    w.reshape(cols=15, element_width=70)
    #
    # for j in range(8):
    #     print j
    #     id = j
    #     r1 = chunks[id][0]
    #     for i in range(500):
    #         r2 = chunks[id][i+1]
    #         im = vid.seek_frame(i)
    #         pts2 = transform_pts(r1, r2)
    #
    #         vis2 = draw_points(im.copy(), r1.pts(), color=(0, 0, 255, 0.5))
    #         vis2 = draw_points_crop(vis2, np.asarray(pts2, dtype=np.int32), color=(255, 0, 0, 0.5), margin=0.4, square=True)
    #         cv2.putText(vis2, str(get_intersection(r1.pts(), pts2))[0:5], (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.cv.CV_AA)
    #
    #         w.add_item(get_image_label(vis2))
    #
    #         r1 = r2

    im = vid.next_frame()
    msers = get_msers_img(im)
    groups = get_region_groups(msers)
    ids = margin_filter(msers, groups)

    # ids = area_filter(msers, ids, )
    ids = children_filter(msers, ids)

    msers = [msers[i] for i in ids]

    for i in range(len(msers)):
        r1 = msers[i]
        for j in range(len(msers)):
            r2 = msers[j]
            pts2 = transform_pts(r1, r2)

            vis2 = draw_points(im.copy(), r1.pts(), color=(0, 0, 255, 0.5))
            vis2 = draw_points_crop(vis2, np.asarray(pts2, dtype=np.int32), color=(255, 0, 0, 0.5), margin=0.4, square=True)
            cv2.putText(vis2, str(get_intersection(r1.pts(), pts2))[0:5], (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.cv.CV_AA)

            w.add_item(get_image_label(vis2))


    w.showMaximized()

    app.exec_()
    app.deleteLater()
    sys.exit()