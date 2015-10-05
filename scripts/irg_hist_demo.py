__author__ = 'flipajs'


from utils.video_manager import VideoManager
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
from skimage.segmentation import quickshift
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import time
import scipy

class ColorHist3d():
    def __init__(self, im, NUM_SAMPLES=32):
        self.hist_ = np.array((NUM_SAMPLES, NUM_SAMPLES, NUM_SAMPLES), dtype=np.int)
        self.im = im

        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                px = im[y, x, :]
                pos = np.round(px/float(NUM_SAMPLES))
                self.hist_[pos] += 1

def igbr_transformation(im):
    I_NORM = 766 * 3 * 2
    igbr = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.double)

    igbr[:, :, 0] = np.sum(im, axis=2) + 1
    igbr[:, :, 1] = im[:, :, 0] / igbr[:, :, 0]
    igbr[:, :, 2] = im[:, :, 1] / igbr[:, :, 0]
    igbr[:, :, 3] = im[:, :, 2] / igbr[:, :, 0]

    igbr[:, :, 0] = igbr[:, :, 0] / I_NORM

    return igbr


def show_all_pixels_in_same_bin(y, x, fig=2):
    global NUM_BINS
    global igbr
    global im

    if fig == 2:
        im_ = im
    else:
        irg = igbr[:, :, [0, 1, 3]]
        irg[:, :, 0] /= np.max(irg[:, :, 0])
        irg[:, :, 1] /= np.max(irg[:, :, 1])
        irg[:, :, 2] /= np.max(irg[:, :, 2])

        irg = np.asarray(irg*255, dtype=np.uint8)
        im_ = irg

    my_pos = np.round(im_[y, x, :] / float(NUM_BINS))

    pos = np.round(im_ / float(NUM_BINS))
    ids = np.logical_and(pos[:, :, 0] == my_pos[0], np.logical_and(pos[:, :, 1] == my_pos[1], pos[:, :, 2] == my_pos[2]))

    # positions = np.argwhere(pos == my_pos)
    # ids = pos == my_pos

    #
    # for y_ in range(im.shape[0]):
    #     for x_ in range(im.shape[1]):
    #         px = im[y_, x_, :]
    #         pos = np.round(px/float(NUM_BINS))
    #         if np.array_equal(my_pos, pos):
    #             positions.append([y_, x_])

    plt.figure(fig)
    im2 = im.copy()
    im2[ids] = [186, 255, 36]
    plt.imshow(im2)
    plt.hold(True)
    # plt.scatter(x, y, c='r')

    # positions = np.array(positions)
    # plt.scatter(positions[:, 1], positions[:, 0], c='b')
    # plt.hold(False)
    plt.show()

def OnClick(event):
    from matplotlib.backend_bases import MouseEvent

    if isinstance(event, MouseEvent):
        if event.xdata and event.ydata:
            show_all_pixels_in_same_bin(event.ydata, event.xdata)
            show_all_pixels_in_same_bin(event.ydata, event.xdata, fig=3)

if __name__ == "__main__":
    NUM_BINS = 32

    # vid = VideoManager('/Users/flipajs/Documents/wd/C210min.avi')
    vid = VideoManager('/Users/flipajs/Documents/wd/bigLense_clip.avi')

    im = vid.get_frame(0)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    igbr = igbr_transformation(im)

    plt.ion()
    fig = plt.gcf()
    cid_up = fig.canvas.mpl_connect('button_press_event', OnClick)

    plt.imshow(im)
    plt.show()

    while True:
        plt.waitforbuttonpress()
