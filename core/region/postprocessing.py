
from PyQt4 import QtGui, QtCore
import sys
from core.region.clustering import clustering, display_cluster_representants, draw_region
import cPickle as pickle
from sklearn.preprocessing import StandardScaler
from utils.video_manager import get_auto_video_manager
from utils.drawing.collage import create_collage_rows
from scipy.spatial.distance import cdist
import numpy as np
import cv2
from gui.img_grid.img_grid_widget import ImgGridWidget
from functools import partial
from utils.misc import print_progress
from scipy.ndimage.morphology import binary_erosion, grey_erosion, binary_dilation, grey_dilation, grey_closing, grey_opening
from utils.drawing.points import draw_points_crop_binary


def bin2uint8(im):
    return np.asarray(im, dtype=np.uint8) * 255

if __name__ == '__main__':
    from core.project.project import Project

    p = Project()
    p.load_hybrid('/Users/flipajs/Documents/wd/FERDA/Camera3')

    vm = get_auto_video_manager(p)
    frame = vm.get_frame(25)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray1', gray)

    grey2 = grey_dilation(gray, size=(3, 3))
    cv2.imshow('grey dilation', grey2)
    grey3 = grey_erosion(grey2, size=(2, 2))
    cv2.imshow('grey erosion', grey3)
    cv2.waitKey(0)

    # vs = [492, 754]
    for v in range(1, 1000):
        try:
            r = p.gm.region(v)
        except:
            continue

        if r.area() < 800:
            continue

        im1 = bin2uint8(draw_points_crop_binary(r.pts()))
        im2 = im1.copy()
        cv2.imshow('pre', im1)
        cv2.moveWindow('pre', 0, 0)

        steps = 3
        w_move_step = 100
        for i in range(steps):
            im2 = bin2uint8(binary_erosion(im2, iterations=1))
            cv2.imshow('post'+str(i), im2)
            cv2.moveWindow('post' + str(i), (i+1) * w_move_step, 0)

        im2 = bin2uint8(binary_dilation(im2, iterations=steps+1))
        cv2.imshow('dilation', im2)
        cv2.moveWindow('dilation', (steps + 2) * w_move_step, 0)

        cv2.waitKey(0)