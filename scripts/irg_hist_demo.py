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
from PyQt4 import QtGui, QtCore
import sys
from gui.arena.arena_editor import ArenaEditor


class ColorHist3d():
    def __init__(self, im, NUM_SAMPLES=32):
        self.num_samples = NUM_SAMPLES
        self.num_pxs = im.shape[0] * im.shape[1] * im.shape[2]

        pos = np.asarray(im/float(NUM_BINS), dtype=np.int)
        self.hist_bg_ = np.zeros((NUM_SAMPLES, NUM_SAMPLES, NUM_SAMPLES), dtype=np.int)

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                p = pos[i, j]

                self.hist_bg_[p[0], p[1], p[2]] += 1
                    # im[i, j, :] = [255, 186, 36]

        # self.hist_bg_ = cv2.calcHist([im], [0, 1, 2], None, [NUM_SAMPLES, NUM_SAMPLES, NUM_SAMPLES], [0, 256, 0, 256, 0, 256])
        self.hist_fg_ = np.zeros((NUM_SAMPLES, NUM_SAMPLES, NUM_SAMPLES), dtype=np.int)

        self.p_fg_ = np.zeros((NUM_SAMPLES, NUM_SAMPLES, NUM_SAMPLES), dtype=np.float)
        self.im = im

        # self.hist_bg_ =
        # pos = np.asarray(np.round(im/float(NUM_SAMPLES)), dtype=np.int)
        # self.hist_bg_[pos] += 1
        # self.hist_fg_[pos] += 1

        # for y in range(im.shape[0]):
        #     for x in range(im.shape[1]):
        #         # px = im[y, x, :]
        #         # pos = np.asarray(np.round(px/float(NUM_SAMPLES)), dtype=np.int)
        #         self.hist_[pos[y, x]] += 1

    def print_most_densed(self):
        id = np.argmax(self.hist_)
        print self.hist_[id], self.hist_[id] / float(self.num_pxs)
        # for i in range(10):
        #     num = self.hist_[ids[i]]
        #     print ids[i], num, num/float(self.num_pxs)

    def swap_bg2fg(self, pxs):
        pos = np.asarray(pxs/float(self.num_samples), dtype=np.int)
        for i in range(pxs.shape[0]):
            p = pos[i, :]

            self.hist_bg_[p[0], p[1], p[2]] -= 1
            self.hist_fg_[p[0], p[1], p[2]] += 1

    def compute_p_fg(self):
        for i in range(self.num_samples):
            for j in range(self.num_samples):
                for k in range(self.num_samples):
                    num_bg = self.hist_bg_[i, j, k]
                    num_fg = self.hist_fg_[i, j, k]
                    if num_bg + num_fg > 0:
                        self.p_fg_[i, j, k] = num_fg / float(num_bg + num_fg)
                        print i, j, k, self.p_fg_[i, j, k]

def igbr_transformation(im):
    I_NORM = 766 * 3 * 2
    igbr = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.double)

    igbr[:, :, 0] = np.sum(im, axis=2) + 1
    igbr[:, :, 1] = im[:, :, 0] / igbr[:, :, 0]
    igbr[:, :, 2] = im[:, :, 1] / igbr[:, :, 0]
    igbr[:, :, 3] = im[:, :, 2] / igbr[:, :, 0]

    igbr[:, :, 0] = igbr[:, :, 0] / I_NORM

    return igbr


def show_all_pixels_in_same_bin(y, x, fig=2, tolerance=0):
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
    a_ = np.abs(pos[:, :, 0] - my_pos[0]) <= tolerance
    b_ = np.abs(pos[:, :, 1] - my_pos[1]) <= tolerance
    c_ = np.abs(pos[:, :, 2] - my_pos[2]) <= tolerance
    ids = np.logical_and(a_, b_, c_)

    plt.figure(fig)
    im2 = im.copy()
    im2[ids] = [186, 255, 36]
    plt.imshow(im2)
    plt.hold(True)

    plt.show()

def OnClick(event):
    from matplotlib.backend_bases import MouseEvent

    if isinstance(event, MouseEvent):
        if event.xdata and event.ydata:
            show_all_pixels_in_same_bin(event.ydata, event.xdata)
            show_all_pixels_in_same_bin(event.ydata, event.xdata, fig=3)

def QImageToCvMat(incomingImage):
    '''  Converts a QImage into an opencv MAT format  '''
    incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format_RGB32)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.constBits()
    ptr.setsize(incomingImage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
    return arr

def get_ccs(im):
    import skimage
    labeled, num = skimage.measure.label(im, background=0, return_num=True)

    ccs = []
    for i in range(num):
        ccs.append(np.argwhere(labeled == i))

    return ccs

def find_dist_thresholds(ccs, data, CH3d):
    for cc in ccs:
        pxs = data[cc[:, 0], cc[:, 1], :]
        c = np.mean(cc, axis=0)

        center_color = data[c[0], c[1], :]

        dists = cdist(pxs, np.array([center_color]))
        dists_med = np.median(dists)
        # print np.min(dists), dists_med, np.max(dists), np.percentile(dists, 0.5)

        plt.figure(1)
        plt.plot(np.sort(dists, axis=0))
        plt.hold('on')

        plt.figure(4)
        ids = dists < np.percentile(dists, 70)
        coords = cc[np.reshape(ids, (ids.shape[0],)), :]
        pxs_ = data[coords[:, 0], coords[:, 1], :]

        CH3d.swap_bg2fg(pxs_)
        # im_copy[cc[:, 0], cc[:, 1], :] = [255, 0, 0]
        global im_copy
        im_copy[coords[:, 0], coords[:, 1], :] = [186, 255, 36]
        plt.imshow(im_copy)


def show_foreground(CH3d, data, im):
    global NUM_BINS

    plt.figure(5)
    import time
    s = time.time()
    pos = np.asarray(data/float(NUM_BINS), dtype=np.int)
    # pos = [data[:,:,0]/float(NUM_BINS), data[:,:,1]/float(NUM_BINS), data[:,:,2]/float(NUM_BINS)]

    # pos = np.asarray(data/float(NUM_BINS), dtype=np.int)
    # pos = np.asarray(pos, dtype=np.int)
    s = time.time()
    # ids = np.argwhere(CH3d.p_fg_[pos] > 0.5)
    # pos = np.reshape(pos, (pos.shape[0]*pos.shape[1], 3))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            p = pos[i, j]

            if CH3d.p_fg_[p[0], p[1], p[2]] > 0.5:
                im[i, j, :] = [255, 186, 36]

    #
    # vals = CH3d.p_fg_[pos[:,:,0], pos[:,:,1], pos[:,:,2]]
    # # ids = CH3d.p_fg_[pos[:,:,0], pos[:,:,1], pos[:,:,2]] > 0
    # print "argwhere takes: ", time.time() - s
    #
    # im[ids[:, 0], ids[:, 1], :] = [255, 186, 36]

    # plt.imshow(im)
    # plt.show()

    return im

def get_irg_255(im):
    igbr = igbr_transformation(im)
    irg = igbr[:, :, [0, 1, 3]]

    irg_255 = np.zeros(irg.shape)
    irg_255[:,:,0] = irg[:,:,0]/np.max(irg[:,:,0])
    irg_255[:,:,1] = irg[:,:,1]/np.max(irg[:,:,1])
    irg_255[:,:,2] = irg[:,:,2]/np.max(irg[:,:,2])
    irg_255 = np.asarray(irg_255*255, dtype=np.uint8)

    return irg_255

if __name__ == "__main__":
    NUM_BINS = 32

    vid = VideoManager('/Users/flipajs/Documents/wd/C210min.avi')
    # vid = VideoManager('/Users/flipajs/Documents/wd/bigLense_clip.avi')

    frame = 300
    im = vid.get_frame(frame)
    im_copy = im.copy()
    # h3d = cv2.calcHist([im], [0, 1, 2], None, [NUM_BINS, NUM_BINS, NUM_BINS], [0, 256, 0, 256, 0, 256])
    # print h3d

    # ch = ColorHist3d(im)
    # ch.print_most_densed()

    mask_frames = [300, 500]
    mask_name = '/Users/flipajs/Documents/wd/colormarks/c210min'
    hist_name = '/Users/flipajs/Documents/wd/colormarks/hist1.pkl'

    import cPickle as pickle

    if False:
        app = QtGui.QApplication(sys.argv)

        ex = ArenaEditor(im, None)
        ex.show()
        ex.move(-500, -500)
        ex.showMaximized()
        ex.setFocus()

        app.exec_()
        app.deleteLater()

        mask = ex.merge_images()
        mask = QImageToCvMat(mask)

        with open(mask_name+str(frame)+'.pkl', 'wb') as f:
            pp = pickle.Pickler(f, -1)
            pp.dump(mask)
    # else:
    #     with open(mask_name, 'rb') as f:
    #         up = pickle.Unpickler(f)
    #         mask = up.load()

    if True:
        if True:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            irg_255 = get_irg_255(im)
            CH3d = ColorHist3d(irg_255.copy(), NUM_SAMPLES=NUM_BINS)

            for frame in mask_frames:
                im = vid.get_frame(frame)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                with open(mask_name+str(frame)+'.pkl', 'rb') as f:
                    up = pickle.Unpickler(f)
                    mask_ = up.load()

                # mask_ = np.sum(mask_[:, :, 0:3], axis=2)
                # mask_ = mask_ > 0

                # plt.imshow(mask_)
                # plt.show()
                # plt.waitforbuttonpress(0)

                ccs = get_ccs(mask_)
                irg_255 = get_irg_255(im)
                find_dist_thresholds(ccs, irg_255.copy(), CH3d)

            CH3d.compute_p_fg()

            with open(hist_name, 'wb') as f:
                pp = pickle.Pickler(f, -1)
                pp.dump(CH3d)
        else:
            with open(hist_name, 'rb') as f:
                up = pickle.Unpickler(f)
                CH3d = up.load()

        for frame in range(0, 2001, 50):
            print frame
            im = vid.get_frame(frame)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            plt.figure(2)
            plt.imshow(im)

            igbr = igbr_transformation(im)
            irg = igbr[:, :, [0, 1, 3]]

            irg_255 = np.zeros(irg.shape)
            irg_255[:,:,0] = irg[:,:,0]/np.max(irg[:,:,0])
            irg_255[:,:,1] = irg[:,:,1]/np.max(irg[:,:,1])
            irg_255[:,:,2] = irg[:,:,2]/np.max(irg[:,:,2])
            irg_255 = np.asarray(irg_255*255, dtype=np.uint8)

            foreground = show_foreground(CH3d, irg_255.copy(), im.copy())

            # plt.show()
            # plt.waitforbuttonpress(0)

            plt.imsave('/Users/flipajs/Documents/wd/colormarks/'+str(frame)+'.png', foreground)
