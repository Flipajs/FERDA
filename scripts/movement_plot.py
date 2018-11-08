from __future__ import unicode_literals
from builtins import range
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
from core.region.mser import get_msers_img
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

WORKING_DIR = '/Users/fnaiser/Documents/chunks'


def load_chunks():
    with open(WORKING_DIR+'/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)

    return chunks

if __name__ == '__main__':
    chunks = load_chunks()

    n_frames = 200
    moves = []
    distances_2_nearest = []
    predict_size = []
    for f in range(n_frames):
        for id in range(8):
            prev = chunks[id][f]
            now = chunks[id][f+1]
            next = chunks[id][f+2]

            pred = now.centroid() - prev.centroid()
            move = next.centroid() - now.centroid()

            moves.append(move - pred)

            predict_size.append(np.linalg.norm(pred))

            best_d = np.inf
            for i in range(8):
                if i == id:
                    continue

                d = np.linalg.norm(chunks[id][f+1].centroid() - chunks[i][f+1].centroid())

                if d < best_d:
                    best_d = d

            distances_2_nearest.append(best_d)


    moves = np.array(moves)

    # for f in range(500):
    #     for id in range(8):
    #         i = f*8 + id
    #         if np.linalg.norm(moves[i]) > 20:
    #             print f, id, np.linalg.norm(moves[i]), np.linalg.norm(chunks[id][f+1].centroid() - chunks[id][f+2].centroid())
    #             cv2.imshow('r1', np.asarray(draw_points_crop_binary(chunks[id][f+1].pts()) * 255, dtype=np.uint8))
    #             cv2.imshow('r2', np.asarray(draw_points_crop_binary(chunks[id][f+2].pts()) * 255, dtype=np.uint8))
    #             cv2.waitKey(0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    distances_2_nearest = np.array(distances_2_nearest)
    ids = distances_2_nearest < 50

    # ax.scatter(moves[ids, 0], moves[ids, 1], distances_2_nearest[ids], c=np.linalg.norm(moves, axis=1)[ids])
    # ax.scatter(moves[ids, 0], moves[ids, 1], distances_2_nearest[ids], c=distances_2_nearest[ids])
    ax.scatter(moves[:, 0], moves[:, 1], predict_size)
    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
    plt.show()

    # plt.scatter(moves[:, 0], moves[:, 1])
    # plt.show()