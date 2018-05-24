__author__ = 'fnaiser'

from PyQt4 import QtCore, QtGui
import sys
from sklearn import svm
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
from core.region.mser import get_msers_img, get_all_msers
from gui.img_grid.img_grid_dialog import ImgGridDialog
from utils.drawing.points import draw_points_crop, get_contour, draw_points
from utils.video_manager import get_auto_video_manager
from gui.gui_utils import get_image_label
import numpy as np
from skimage.transform import resize
import cv2

vid_path = '/Users/fnaiser/Documents/chunks/eight.m4v'
working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/eight'
classes=np.zeros(77)
classes[0:8] = 1
classes[25:33] = 1
classes[51:59] = 1
init_frames=3

num_frames=10

# vid_path = '/Users/fnaiser/Documents/chunks/NoPlasterNoLid800.m4v'
# working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/noplast'
# classes = np.zeros(301)
# classes[3:19] = 1
# classes[8] = 0
# classes[106:122] = 1
# classes[110] = 0
# classes[202:218] = 1
# classes[207] = 0
#
# init_frames=3
# num_frames = 100

# vid_path = '/Volumes/Seagate Expansion Drive/IST - videos/smallLense_colony1.avi'
# working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/smalllense'
# classes = np.zeros(133)
# classes[2:15] = 1
# classes[44:57] = 1
# classes[91:104] = 1
# init_frames = 3

# vid_path = '/Volumes/Seagate Expansion Drive/IST - videos/bigLenses_colormarks2.avi'
# working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/biglenses2'
# classes = np.zeros(847)
# classes[0:9] = 1
# classes[218:226] = 1
# classes[430:439] = 1
# classes[637:646] = 1
# init_frames = 4

# vid_path = '/Volumes/Seagate Expansion Drive/IST - videos/bigLenses_colormarks1.avi'
# working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/biglenses1'
# classes = np.zeros(597)
# classes[0:6] = 1
# classes[191:197] = 1
# classes[383:389] = 1
# init_frames = 3

# vid_path = '/Users/fnaiser/Documents/Camera 1_biglense1.avi'
# working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/camera1'
# classes = np.zeros(1178)
# classes[0:5] = 1
# classes[384:389] = 1
# classes[792:797] = 1
# init_frames = 3

def select_msers_cached(frame):
    msers = get_all_msers(frame, vid_path, working_dir)
    groups = get_region_groups(msers)
    ids = margin_filter(msers, groups)

    ids = children_filter(msers, ids)

    return [msers[i] for i in ids]


def init():
    app = QtGui.QApplication(sys.argv)

    dial = ImgGridDialog()
    vid = get_auto_video_manager(vid_path)

    dial.img_grid.reshape(15, element_width=70)

    id = 0
    for f in range(init_frames):
        im = vid.next_frame()
        msers = select_msers_cached(f)

        for m in msers:
            vis = draw_points_crop(im.copy(), m.pts(), square=True)

            vis = np.asarray(resize(vis, (70, 70)) * 255, dtype=np.uint8)
            cv2.putText(vis, str(id), (1, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 255, 100), 1, cv2.cv.CV_AA)

            iml = get_image_label(vis)
            dial.img_grid.add_item(iml)

            id = id + 1

    dial.showMaximized()
    app.exec_()
    sys.exit()


def get_x(im_gray, m):
    cl = len(get_contour(m.pts()))
    # intensities = im_gray[m.pts()[:, 0], m.pts()[:, 1]]
    # min_i10 = np.percentile(intensities, perc)
    #
    # return [m.margin_, m.area(), cl/float(m.area()), min_i10, m.max_intensity_, m.major_axis_/m.minor_axis_]
    #
    intensities = im_gray[m.pts()[:, 0], m.pts()[:, 1]]
    min_i10 = np.percentile(intensities, perc)

    return [m.margin_, cl/m.area()**0.5, min_i10]

def get_x_minI(m):
    cl = len(get_contour(m.pts()))
    return [m.margin_, cl/m.area()**0.5, m.min_intensity_]

def get_svm_model():
    vid = get_auto_video_manager(vid_path)

    id = 0
    X = []

    for f in range(init_frames):
        im = vid.next_frame()
        # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        for m in select_msers_cached(f):
            m.id_ = id

            id += 1

            X.append(get_x_minI(m))

    clf = svm.SVC(kernel='linear', probability=True, class_weight='auto')
    smv = clf.fit(X, classes)

    return smv

if __name__ == '__main__':
    if False:
        init()
    else:
        app = QtGui.QApplication(sys.argv)

        dial = ImgGridDialog()
        vid = get_auto_video_manager(vid_path)

        id = 0
        X = []

        perc = 3

        for f in range(init_frames):
            im = vid.next_frame()
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            for m in select_msers_cached(f):
                m.id_ = id

                id += 1

                X.append(get_x(im_gray, m))

        clf = svm.SVC(kernel='linear', probability=True, class_weight='auto')

        print clf.fit(X, classes)
        print clf.support_vectors_
        print clf.support_
        print clf.n_support_

        vid = get_auto_video_manager(vid_path)
        dial.img_grid.reshape(15, element_width=70)

        for f in range(num_frames):
            im = vid.next_frame()
            msers = select_msers_cached(f)

            im2 = np.copy(im)
            im_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            detected = 0
            for m in msers:
                x = get_x(im_gray, m)

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
            print f, detected

        dial.showMaximized()
        app.exec_()
        sys.exit()