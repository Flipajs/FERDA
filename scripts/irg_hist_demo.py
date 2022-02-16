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
from PyQt6 import QtCore, QtGui, QtWidgets
import sys
from gui.arena.arena_editor import ArenaEditor
import pickle as pickle
import time
from scipy import ndimage


class ColorHist3d():
    def __init__(self, im, num_colors, num_bins1=32, num_bins2=32, num_bins3=32, theta=0.1, epsilon=0.3):
        self.theta = theta
        self.epsilon = epsilon

        # TODO: 2x multiply num of bins
        self.num_bins1 = num_bins1
        self.num_bins2 = num_bins2
        self.num_bins3 = num_bins3

        self.num_bins_v = np.array([self.num_bins1, self.num_bins2, self.num_bins3], dtype=np.float)

        self.num_pxs = im.shape[0] * im.shape[1] * im.shape[2]
        self.num_colors = num_colors
        self.BG = num_colors

        pos = np.asarray(im / self.num_bins_v, dtype=np.int)

        # num_colors + 1 for background
        self.hist_ = np.zeros((self.num_bins1, self.num_bins2, self.num_bins3, num_colors + 1), dtype=np.int)
        self.hist_[:, :, :, self.BG] += 1

        self.hist_labels_ = np.zeros((self.num_bins1, self.num_bins2, self.num_bins3), dtype=np.int) + self.BG

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                p = pos[i, j]
                self.hist_[p[0], p[1], p[2], self.BG] += 1

    def swap_bg2color(self, pxs, color_id):
        pos = np.asarray(pxs / self.num_bins_v, dtype=np.int)
        for i in range(pxs.shape[0]):
            p = pos[i, :]

            if self.hist_[p[0], p[1], p[2], self.BG] > 1:
                self.hist_[p[0], p[1], p[2], self.BG] -= 1

            self.hist_[p[0], p[1], p[2], color_id] += 1

    def remove_bg(self, pxs):
        pos = np.asarray(pxs / self.num_bins_v, dtype=np.int)
        for i in range(pxs.shape[0]):
            p = pos[i, :]

            if self.hist_[p[0], p[1], p[2], self.BG] > 1:
                self.hist_[p[0], p[1], p[2], self.BG] -= 1

    def add_color(self, pxs, color_id):
        pos = np.asarray(pxs / self.num_bins_v, dtype=np.int)
        for i in range(pxs.shape[0]):
            p = pos[i, :]

            self.hist_[p[0], p[1], p[2], color_id] += 1

    def compute_p_fg(self):
        for i in range(self.num_bins1):
            for j in range(self.num_bins2):
                for k in range(self.num_bins3):
                    num_bg = self.hist_bg_[i, j, k]
                    num_fg = self.hist_fg_[i, j, k]
                    if num_bg + num_fg > 0:
                        self.p_fg_[i, j, k] = num_fg / float(num_bg + num_fg)
                        print(i, j, k, self.p_fg_[i, j, k])

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

            for i in range(self.num_bins1):
                for j in range(self.num_bins2):
                    for k in range(self.num_bins3):
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

            print("C_ID DONE: ", c_id, sum_)


def irgb_transformation(im):
    I_NORM = 766 * 3 * 2
    irgb = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.double)

    irgb[:, :, 0] = np.sum(im, axis=2) + 1
    irgb[:, :, 1] = im[:, :, 0] / irgb[:, :, 0]
    irgb[:, :, 2] = im[:, :, 1] / irgb[:, :, 0]
    irgb[:, :, 3] = im[:, :, 2] / irgb[:, :, 0]

    irgb[:, :, 0] = irgb[:, :, 0] / I_NORM

    return irgb


def show_all_pixels_in_same_bin(y, x, fig=2, tolerance=0):
    global NUM_BINS
    global igbr
    global im
    global num_bins_v

    if fig == 2:
        im_ = im
    else:
        irg = igbr[:, :, [0, 1, 3]]
        irg[:, :, 0] /= np.max(irg[:, :, 0])
        irg[:, :, 1] /= np.max(irg[:, :, 1])
        irg[:, :, 2] /= np.max(irg[:, :, 2])

        irg = np.asarray(irg * 255, dtype=np.uint8)
        im_ = irg

    my_pos = np.round(im_[y, x, :] / num_bins_v)

    pos = np.round(im_ / num_bins_v)
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
    incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGB32)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.constBits()
    ptr.setsize(incomingImage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
    return arr


def get_ccs(im, bg=0, min_a=1, max_a=5000):
    import skimage

    labeled, num = skimage.measure.label(im, background=bg, return_num=True)

    sizes = ndimage.sum(np.ones((im.shape[0], im.shape[1])), labeled, list(range(1, num + 1)))

    ccs = []
    for i in range(num):
        np.argwhere(labeled == i)
        ccs.append(np.argwhere(labeled == i))

    return ccs


def get_mean_around(data, c):
    color = np.asarray(data[c[0], c[1]], dtype=np.int)
    color += np.array(data[c[0], c[1] - 1])
    color += np.array(data[c[0] - 1, c[1]])
    color += np.array(data[c[0] + 1, c[1]])
    color += np.array(data[c[0], c[1] + 1])

    return np.asarray(color / 5.0, dtype=np.uint8)


def find_dist_thresholds(ccs, data, orig_img=None):
    all_pxs = np.array([], dtype=np.int).reshape(0, 3)
    picked_pxs = np.array([], dtype=np.int).reshape(0, 3)
    mean_colors = []

    for cc in ccs:
        pxs = data[cc[:, 0], cc[:, 1], :]
        all_pxs = np.append(all_pxs, pxs, axis=0)
        c = np.mean(cc, axis=0)

        center_color = get_mean_around(data, c)

        dists = cdist(pxs, np.array([center_color]))

        ids = dists < np.percentile(dists, 70)
        coords = cc[np.reshape(ids, (ids.shape[0],)), :]
        pxs_ = data[coords[:, 0], coords[:, 1], :]

        if orig_img is not None:
            # TODO: solve for multiple colormarks...
            color_representant = get_mean_around(orig_img, c)


        picked_pxs = np.append(picked_pxs, pxs_, axis=0)

    return picked_pxs, all_pxs, color_representant


def show_foreground(CH3d, data, im):
    global num_bins_v

    colors = [[255, 0, 0], [0, 255, 0], [168, 37, 255],
              [55, 255, 255], [15, 135, 255], [255, 255, 0],
              [255, 107, 151], [0, 0, 0]]

    colors = [[255, 0, 0], [0, 255, 0], [168, 37, 255],  # Red Green Purple
              [55, 255, 255], [255, 255, 0],  # Blue Yellow
              [255, 107, 151], [0, 0, 0]]  # Pink Black

    # colors = [[255, 255, 255], [55, 255, 255],  # white blue
    #           [0, 255, 0], [255, 255, 0],  # green yellow
    #           [255, 107, 151], [50, 50, 50]]  # pink bg

    # colors = [[255, 127, 166], [255, 255, 255], [255, 253, 22],
    #           [0, 255, 57], [0, 189, 255], [255, 255, 0],
    #           [255, 107, 151], [0, 0, 0]]

    pos = np.asarray(data / num_bins_v, dtype=np.int)

    s = time.time()

    labels = CH3d.hist_labels_[pos[:, :, 0], pos[:, :, 1], pos[:, :, 2]]

    colors = np.array(colors)
    im_ = colors[labels.ravel()]
    im = np.asarray(im_.reshape(im.shape[0], im.shape[1], 3), dtype=np.uint8)

    # labels = np.zeros((im.shape[0], im.shape[1]), dtype=np.int)
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         p = pos[i, j]
    #
    #         l = CH3d.hist_labels_[p[0], p[1], p[2]]
    #         im[i, j, :] = colors[l]
    #         labels[i, j] = l

    print("labelling time ", time.time() - s)

    return im, labels


def get_irg_255(im):
    irgb = irgb_transformation(im)
    irg = irgb[:, :, [0, 1, 3]]

    # irg[:, :, 0] = irg[:, :, 0] ** 0.5

    irg_255 = np.zeros(irg.shape)
    irg_255[:, :, 0] = irg[:, :, 0] / np.max(irg[:, :, 0])
    irg_255[:, :, 1] = irg[:, :, 1] / np.max(irg[:, :, 1])
    irg_255[:, :, 2] = irg[:, :, 2] / np.max(irg[:, :, 2])
    irg_255 = np.asarray(irg_255 * 255, dtype=np.uint8)

    return irg_255


def get_color_samples_tool(vid, frames=None, wd_=''):
    global wd

    name = 'color_samples'

    color_samples = []
    masks = []
    # try:
    #     with open(wd + '/' + name + '.pkl', 'rb') as f:
    #         up = pickle.Unpickler(f)
    #         color_samples = up.load()
    #         masks = up.load()
    # except IOError:
    #     pass
    # except EOFError:
    #     pass

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

    if not frames:
        frames = [500, 500, 500, 500, 500, 300]
        # f = 916
        # frames = [f, f, f, f, f, f, f, f, f]
        # frames = [52, 52, 52, 52, 52]
        # frames = []

    app = QtWidgets.QApplication(sys.argv)

    for frame in frames:
        im = vid.get_frame(frame)

        ex = ArenaEditor(im, None)
        ex.show()
        ex.move(-500, -500)
        ex.showMaximized()
        ex.setFocus()

        ex.set_paint_mode()
        ex.slider.setValue(7)

        app.exec()
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

    with open(wd_ + '/' + name + '.pkl', 'wb') as f:
        p_ = pickle.Pickler(f, -1)
        p_.dump(color_samples)
        p_.dump(masks)

    return color_samples


def process_ccs(im, labels):
    min_a = 12
    max_a = 500

    s = time.time()
    ccs = get_ccs(labels, bg=-1, min_a=min_a, max_a=max_a)
    print("get_ccs t: ", time.time() - s)

    for cc in ccs:
        if not (min_a < len(cc) < max_a):
            im[cc[:, 0], cc[:, 1], :] = [50, 50, 50]
            # labels[cc[:, 0], cc[:, 1]] = 0

    return im


if __name__ == "__main__":
    NUM_BINS1 = 16
    NUM_BINS2 = 16
    NUM_BINS3 = 16
    # NUM_BINS1 = 32
    # NUM_BINS2 = 32
    # NUM_BINS3 = 32

    num_bins_v = np.array([NUM_BINS1, NUM_BINS2, NUM_BINS3], dtype=np.float)

    wd = '/Users/flipajs/Documents/wd/C210'

    vid = VideoManager('/Users/flipajs/Documents/wd/C210min.avi')
    # vid = VideoManager('/Volumes/Seagate Expansion Drive/IST - videos/bigLenses_colormarks2.avi')
    # vid = VideoManager('/Users/flipajs/Documents/wd/bigLense_clip.avi')

    color_samples = get_color_samples_tool(vid)

    frame = 500
    im = vid.get_frame(frame)
    im_copy = im.copy()

    if True:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if True:
            irg_255 = get_irg_255(im)
            CH3d = ColorHist3d(irg_255.copy(), 6, num_bins1=NUM_BINS1, num_bins2=NUM_BINS2, num_bins3=NUM_BINS3,
                               theta=0.3, epsilon=0.9)

            for (picked_pxs, all_pxs), c_id in zip(color_samples, list(range(len(color_samples)))):
                CH3d.remove_bg(all_pxs)
                CH3d.add_color(picked_pxs, c_id)

            CH3d.assign_labels()

            with open(wd + '/hist.pkl', 'wb') as f:
                pp = pickle.Pickler(f, -1)
                pp.dump(CH3d)
        else:
            with open(wd + '/hist.pkl', 'rb') as f:
                up = pickle.Unpickler(f)
                CH3d = up.load()

        for frame in range(0, 1001, 50):
            for i in range(10):
                frame_ = frame + i

                im = vid.get_frame(frame_)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                irg_255 = get_irg_255(im)

                # irgb = irgb_transformation(im)
                # plt.subplot(2, 2, 1)
                # plt.imshow(irgb[:, :, 0], cmap='gray')
                # plt.subplot(2, 2, 2)
                # plt.imshow(irgb[:, :, 1], cmap='gray')
                # plt.subplot(2, 2, 3)
                # plt.imshow(irgb[:, :, 2], cmap='gray')
                # plt.subplot(2, 2, 4)
                # plt.imshow(irgb[:, :, 3], cmap='gray')
                # plt.show()
                # plt.waitforbuttonpress(0)

                foreground, labels = show_foreground(CH3d, irg_255.copy(), im.copy())

                plt.imsave(wd + '/' + str(frame_) + '.png', im)
                plt.imsave(wd + '/' + str(frame_) + '_c.png', foreground)
                foreground_ = process_ccs(foreground, labels)
                plt.imsave(wd + '/' + str(frame_) + '_p.png', foreground_)

                print(frame_)
