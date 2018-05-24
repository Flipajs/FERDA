__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore
from gui.img_controls.my_scene import MyScene
from gui.gui_utils import cvimg2qtpixmap
import numpy as np
from skimage.transform import resize
from utils.img import get_roi
from utils.roi import ROI, get_roi
from gui.gui_utils import get_image_label
from utils.drawing.points import draw_points_crop, draw_points
from utils.video_manager import get_auto_video_manager
from core.region.mser import get_msers_img, get_all_msers
from skimage.transform import resize
from gui.img_controls.my_view import MyView
from gui.img_controls.my_scene import MyScene
import sys
from PyQt4 import QtGui, QtCore
from gui.img_controls.gui_utils import cvimg2qtpixmap
import numpy as np
import pickle
import cv2
from core.region.fitting import Fitting

if __name__ == '__main__':
    # app = QtGui.QApplication(sys.argv)

    # with open('/Volumes/Seagate Expansion Drive/mser_svm/biglenses2/certainty_visu.pkl', 'rb') as f:
    #     up = pickle.Unpickler(f)
    #     g = up.load()
    #     regions = up.load()
    #     ccs = up.load()
    #     vid_path = up.load()

    with open('/Volumes/Seagate Expansion Drive/mser_svm/eight/certainty_visu2.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        ccs = up.load()
        vid_path = up.load()

    # with open('/Volumes/Seagate Expansion Drive/mser_svm/eight/certainty_visu2.pkl', 'wb') as f:
    #     p = pickle.Pickler(f)
    #     p.dump(ccs)
    #     p.dump(vid_path)
    # print 'SAVED'


    merged_ids = []


    vid = get_auto_video_manager(vid_path)
    im = vid.next_frame()

    h_, w_, _ = im.shape
    color1 = (0, 255, 0, 0.8)
    color1_ = (0, 255, 0, 0.2)
    color2 = (0, 0, 255, 0.8)
    color2_ = (0, 0, 255, 0.2)

    visu = np.zeros((h_, w_, 3), dtype=np.uint8)
    cv2.imshow('visu', visu)
    cv2.moveWindow('visu', 0, 0)
    for c in ccs:
        visu = np.zeros((h_, w_, 3), dtype=np.uint8)

        for c1 in c['c1']:
            draw_points(visu, c1.contour(), color=color1)
            draw_points(visu, c1.pts(), color=color1_)

        for c2 in c['c2']:
            draw_points(visu, c2.contour(), color=color2)
            draw_points(visu, c2.pts(), color=color2_)

        cv2.imshow('visu', visu)
        cv2.waitKey(0)

        # try:
        if len(c['c1']) > 0 and len(c['c2']) > 0:
            avg_area_c1 = 0
            for c1 in c['c1']:
                avg_area_c1 += c1.area()
            avg_area_c1 /= float(len(c['c1']))

            avg_area_c2 = 0
            for c2 in c['c2']:
                avg_area_c2 += c2.area()
            avg_area_c2 /= float(len(c['c2']))

            t1_ = c['c1']
            t2_ = c['c2']
            t_reversed = False
            if avg_area_c1 > avg_area_c2:
                t1_ = c['c2']
                t2_ = c['c1']
                t_reversed = True

            reg = []
            for c2 in t2_:
                if not reg:
                    reg = c2
                else:
                    reg.pts_ = np.append(reg.pts_, c2.pts_, axis=0)

            objects = []
            for c1 in t1_:
                objects.append(c1)

            f = Fitting(reg, objects, num_of_iterations=10)
            f.fit()

        # except:
        #     pass


    # app.exec_()
    # sys.exit()