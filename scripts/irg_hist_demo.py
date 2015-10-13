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
import cPickle as pickle


class ColorHist3d():
    def __init__(self, im, num_colors, num_bins=32, theta=0.1, epsilon=0.3):
        self.theta = theta
        self.epsilon = epsilon

        self.num_bins = num_bins
        self.num_pxs = im.shape[0] * im.shape[1] * im.shape[2]
        self.num_colors = num_colors
        self.BG = num_colors

        pos = np.asarray(im/float(self.num_bins), dtype=np.int)

        # num_colors + 1 for background
        self.hist_ = np.zeros((num_bins, num_bins, num_bins, num_colors+1), dtype=np.int)
        self.hist_[:,:,:,self.BG] += 1

        self.hist_labels_ = np.zeros((num_bins, num_bins, num_bins), dtype=np.int) + self.BG

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                p = pos[i, j]
                self.hist_[p[0], p[1], p[2], self.BG] += 1

    def swap_bg2color(self, pxs, color_id):
        pos = np.asarray(pxs/float(self.num_bins), dtype=np.int)
        for i in range(pxs.shape[0]):
            p = pos[i, :]

            if self.hist_[p[0], p[1], p[2], self.BG] > 1:
                self.hist_[p[0], p[1], p[2], self.BG] -= 1

            self.hist_[p[0], p[1], p[2], color_id] += 1

    def remove_bg(self, pxs):
        pos = np.asarray(pxs/float(self.num_bins), dtype=np.int)
        for i in range(pxs.shape[0]):
            p = pos[i, :]

            if self.hist_[p[0], p[1], p[2], self.BG] > 1:
                self.hist_[p[0], p[1], p[2], self.BG] -= 1

    def add_color(self, pxs, color_id):
        pos = np.asarray(pxs/float(self.num_bins), dtype=np.int)
        for i in range(pxs.shape[0]):
            p = pos[i, :]

            self.hist_[p[0], p[1], p[2], color_id] += 1

    def compute_p_fg(self):
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                for k in range(self.num_bins):
                    num_bg = self.hist_bg_[i, j, k]
                    num_fg = self.hist_fg_[i, j, k]
                    if num_bg + num_fg > 0:
                        self.p_fg_[i, j, k] = num_fg / float(num_bg + num_fg)
                        print i, j, k, self.p_fg_[i, j, k]

    def get_p_k_x(self, k, x):
        a = self.hist_[x[0], x[1], x[2], k]
        n = np.sum(self.hist_[x[0], x[1], x[2], :])

        return a / float(n)

    def get_p_x_k(self, x, k):
        a = self.hist_[x[0], x[1], x[2], k]
        if a == 0:
            return 0.0

        n = np.sum(self.hist_[:, :, :, k])

        return a / float(n)

    def assign_labels(self):
        for c_id in range(self.num_colors):
            sum_ = 0
            good_enough = []

            for i in range(self.num_bins):
                for j in range(self.num_bins):
                    for k in range(self.num_bins):
                        pkx = self.get_p_k_x(c_id, [i, j, k])
                        pxk = self.get_p_x_k([i, j, k], c_id)

                        if pkx > self.theta:
                            good_enough.append((pxk, [i, j, k]))

            good_enough = sorted(good_enough, key=lambda x: -x[0])

            sum_ = 0
            for g in good_enough:
                self.hist_labels_[g[1][0], g[1][1], g[1][2]] = c_id
                sum_ += g[0]

                if sum_ > self.epsilon:
                    break

            print "C_ID DONE: ", c_id, sum_


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

def get_mean_around(data, c):
    color = np.asarray(data[c[0], c[1]], dtype=np.int)
    color += np.array(data[c[0], c[1]-1])
    color += np.array(data[c[0]-1, c[1]])
    color += np.array(data[c[0]+1, c[1]])
    color += np.array(data[c[0], c[1]+1])

    return np.asarray(color / 5.0, dtype=np.uint8)


def find_dist_thresholds(ccs, data):
    all_pxs = np.array([], dtype=np.int).reshape(0, 3)
    picked_pxs = np.array([], dtype=np.int).reshape(0, 3)
    for cc in ccs:
        pxs = data[cc[:, 0], cc[:, 1], :]
        all_pxs = np.append(all_pxs, pxs, axis=0)
        c = np.mean(cc, axis=0)

        center_color = get_mean_around(data, c)
        # center_color = data[c[0], c[1], :]

        dists = cdist(pxs, np.array([center_color]))
        # dists_med = np.median(dists)
        # print np.min(dists), dists_med, np.max(dists), np.percentile(dists, 0.5)

        plt.figure(1)
        plt.plot(np.sort(dists, axis=0))
        plt.hold('on')

        plt.figure(4)
        ids = dists < np.percentile(dists, 70)
        coords = cc[np.reshape(ids, (ids.shape[0],)), :]
        pxs_ = data[coords[:, 0], coords[:, 1], :]

        picked_pxs = np.append(picked_pxs, pxs_, axis=0)

    return picked_pxs, all_pxs


def show_foreground(CH3d, data, im):
    global NUM_BINS

    colors = [[255, 0, 0], [0, 255, 0], [168, 37, 255],
              [55, 255, 255], [15, 135, 255], [255, 255, 0],
              [255, 107, 151], [0, 0, 0]]

    # colors = [[255, 127, 166], [255, 255, 255], [255, 253, 22],
    #           [0, 255, 57], [0, 189, 255], [255, 255, 0],
    #           [255, 107, 151], [0, 0, 0]]

    pos = np.asarray(data/float(NUM_BINS), dtype=np.int)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            p = pos[i, j]

            l = CH3d.hist_labels_[p[0], p[1], p[2]]
            im[i, j, :] = colors[l]

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


def get_color_samples_tool(vid):
    global wd

    name = 'color_samples'

    color_samples = []
    masks = []
    try:
        with open(wd+'/'+name+'.pkl', 'rb') as f:
            up = pickle.Unpickler(f)
            color_samples = up.load()
            masks = up.load()
    except IOError:
        pass
    except EOFError:
        pass

    # color_samples = []
    # for frame, mask in masks:
    #     im = vid.get_frame(frame)
    #     ccs = get_ccs(mask)
    #
    #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #     irg_255 = get_irg_255(im)
    #     sample_pxs, all_pxs = find_dist_thresholds(ccs, irg_255.copy())
    #
    #     color_samples.append((sample_pxs, all_pxs))

    frames = [500, 500, 500, 500, 500, 500, 300]
    # frames = [52, 52, 52, 52, 52]
    frames = []

    app = QtGui.QApplication(sys.argv)

    for frame in frames:
        im = vid.get_frame(frame)

        ex = ArenaEditor(im, None)
        ex.show()
        ex.move(-500, -500)
        ex.showMaximized()
        ex.setFocus()

        ex.set_paint_mode()
        ex.slider.setValue(7)

        app.exec_()
        mask = ex.merge_images()
        mask = QImageToCvMat(mask)

        mask = np.sum(mask[:, :, 0:3], axis=2)
        mask = mask > 0
        ccs = get_ccs(mask)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        irg_255 = get_irg_255(im)

        sample_pxs, all_pxs = find_dist_thresholds(ccs, irg_255.copy())

        color_samples.append((sample_pxs, all_pxs))

        masks.append((frame, mask))

    app.deleteLater()

    with open(wd+'/'+name+'.pkl', 'wb') as f:
        p_ = pickle.Pickler(f, -1)
        p_.dump(color_samples)
        p_.dump(masks)

    return color_samples

if __name__ == "__main__":
    NUM_BINS = 32

    wd = '/Users/flipajs/Documents/wd/colormarks'

    vid = VideoManager('/Users/flipajs/Documents/wd/C210min.avi')
    # vid = VideoManager('/Users/flipajs/Documents/wd/bigLense_clip.avi')

    color_samples = get_color_samples_tool(vid)

    frame = 500
    im = vid.get_frame(frame)
    im_copy = im.copy()

    if True:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if True:
            irg_255 = get_irg_255(im)
            CH3d = ColorHist3d(irg_255.copy(), 7, num_bins=NUM_BINS, theta=0.5, epsilon=0.8)

            for (picked_pxs, all_pxs), c_id in zip(color_samples, range(len(color_samples))):
                CH3d.remove_bg(all_pxs)
                CH3d.add_color(picked_pxs, c_id)
                # CH3d.swap_bg2color(c, c_id)

            CH3d.assign_labels()

            with open(wd+'/hist.pkl', 'wb') as f:
                pp = pickle.Pickler(f, -1)
                pp.dump(CH3d)
        else:
            with open(wd+'/hist.pkl', 'rb') as f:
                up = pickle.Unpickler(f)
                CH3d = up.load()

        for frame in range(0, 2001, 25):
            im = vid.get_frame(frame)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            irg_255 = get_irg_255(im)
            foreground = show_foreground(CH3d, irg_255.copy(), im.copy())

            plt.imsave(wd+'/'+str(frame)+'.png', im)
            plt.imsave(wd+'/'+str(frame)+'_c.png', foreground)
