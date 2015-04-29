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
from utils.drawing.points import get_contour
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis, convex_hull_image, binary_closing, binary_opening
from core.region.mser import get_msers_, get_all_msers
from skimage.transform import resize
from gui.img_grid.img_grid_dialog import ImgGridDialog


WORKING_DIR = '/Users/fnaiser/Documents/chunks'
vid_path = '/Users/fnaiser/Documents/chunks/eight.m4v'
working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/eight'

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

def get_x(r):
    x = []
    x.append(r.area())
    # x.append(r.a_)
    # x.append(r.b_)
    x.append(r.a_ / r.b_)

    c1 = len(get_contour(r.pts()))
    # x.append(c1)

    # h1 = draw_points_crop_binary(r.pts())
    # h1 = convex_hull_image(h1)
    # h1 = np.sum(h1)
    #
    # x.append(h1)
    x.append(r.margin_)
    # x.append(r.area())
    # x.append(r.min_intensity_)
    # x.append(c1 / (r.area()**0.5))

    return x

def get_svm_model():
    chunks = load_chunks()

    merged = [(1, 209), (6, 472), (4, 473), (4, 474), (4, 475), (5, 635), (5, 636), (5, 638), (4, 649), (4, 650), (4, 651), (4, 652), (4, 653), (1, 654), (1, 655), (1, 656), (1, 657), (1, 658)]

    X = []
    classes = []
    for (id, frame) in merged:
        X.append(get_x(chunks[id][frame]))
        classes.append(1)

    for frame in range(6):
        for id in range(8):
            X.append(get_x(chunks[id][frame]))
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


def select_msers_cached(frame):
    msers = get_all_msers(frame, vid_path, working_dir)
    groups = get_region_groups(msers)
    ids = margin_filter(msers, groups)

    ids = children_filter(msers, ids)

    return [msers[i] for i in ids]

if __name__ == '__main__':
    # chunks = load_chunks()
    # for id in range(8):
    #     pts = draw_points_crop_binary(chunks[id][2].pts())
    #     cv2.imshow('test', np.asarray(pts*255, dtype=np.uint8))
    #     cv2.waitKey(0)

    clf = get_svm_model()
    vid = get_auto_video_manager(vid_path)




    chunks = load_chunks()
    frame = 648
    for id in range(8):
        chunks[id][frame]

    app = QtGui.QApplication(sys.argv)

    dial = ImgGridDialog()
    dial.img_grid.reshape(15, element_width=70)
    vid = get_auto_video_manager(vid_path)

    for f in range(200, 220) + range(460, 480) + range(630, 700):
        # im = vid.move2_next()
        im = vid.seek_frame(f)
        msers = select_msers_cached(f)

        im2 = np.copy(im)
        im_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        detected = 0
        for m in msers:
            x = get_x(m)

            cont = get_contour(m.pts())

            prob = clf.predict_proba([x])
            r_ = int(255*prob[0][0])
            g_ = int(255*prob[0][1])
            vis = draw_points_crop(im.copy(), cont, color=(0, g_, r_, 1), square=True)
            # im2 = draw_points(im2, cont, color=(0, g_, r_, 1))

            vis = np.asarray(resize(vis, (70, 70)) * 255, dtype=np.uint8)
            cv2.putText(vis, str(f)+' '+str(prob[0][1])[0:5], (1, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 255, 100), 1, cv2.cv.CV_AA)

            iml = get_image_label(vis)
            dial.img_grid.add_item(iml)

        # cv2.imshow('test', im2)
        # cv2.imwrite(working_dir+'/'+str(f)+'.jpg', im2)
        # cv2.waitKey(0)

    dial.showMaximized()
    app.exec_()
    sys.exit()