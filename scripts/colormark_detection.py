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

I_NORM = 766 * 3 * 2


def igbr_transformation(im):
    igbr = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.double)

    igbr[:, :, 0] = np.sum(im, axis=2) + 1
    igbr[:, :, 1] = im[:, :, 0] / igbr[:, :, 0]
    igbr[:, :, 2] = im[:, :, 1] / igbr[:, :, 0]
    igbr[:, :, 3] = im[:, :, 2] / igbr[:, :, 0]

    igbr[:, :, 0] = igbr[:, :, 0] / I_NORM

    return igbr


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def colormarks_labelling(image, colors, original_colors=None):
    im_ = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    dists = cdist(im_, colors)
    ids = np.argmin(dists, axis=1)
    labels = ids.reshape((image.shape[0], image.shape[1]))
    if original_colors is None:
        labels = np.asarray(colors[labels], dtype=np.uint8)
    else:
        labels = np.asarray(original_colors[labels], dtype=np.uint8)

    return labels


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print 'x = %d, y = %d' % (
        ix, iy)

    # global coords
    # coords.append((ix, iy))
    print ix, iy
    # if len(coords) == 2:
    #     fig.canvas.mpl_disconnect(cid)


if __name__ == "__main__":
    vid = VideoManager('/Users/flipajs/Documents/wd/C210min.avi')
    coords = np.array(
        [[190.55604719764017, 641.76548672566378],
         [683.29941002949852, 294.3908554572273],
         [639.877581120944, 454.86283185840716],
         [239.64159292035401, 773.91887905604722],
         [687.07522123893807, 430.3200589970503],
         [526.6032448377581, 233.9778761061948],
         [216.98672566371681, 303.83038348082607],
         [464.302359882, 828.668141593]])

    coords = np.asarray(coords, dtype=np.int)
    im = vid.seek_frame(200)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    igbr = igbr_transformation(im)
    orig_colors = im[coords[:, 1], coords[:, 0], :]
    igbr_colors = igbr[coords[:, 1], coords[:, 0], :]

    plt.ion()
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', onclick)
    for i in range(200, 1000, 100):
        im = vid.seek_frame(i)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im)
        # plt.hold(True)
        # plt.scatter(coords[:,0], coords[:,1])
        # plt.hold(False)

        # colors = np.array([[46, 34, 21], [216, 209, 217], [208, 195, 184], [54, 89, 120], [167, 140, 95],
        #                    [148, 52, 56], [168, 125, 144], [36, 58, 96], [122, 103, 110], [199, 190, 196],
        #                    [163, 123, 137]])
        # colormarks_labelling(im, colors)

        #############
        import time

        s = time.time()
        igbr = igbr_transformation(im)

        labels = colormarks_labelling(igbr, igbr_colors, orig_colors)
        print "TIME: ", time.time() - s
        plt.figure()
        plt.imshow(labels)
        plt.show()
        plt.waitforbuttonpress()

        # rows = 2
        # cols = 3
        #
        # i = 1
        # plt.figure()
        # plt.subplot(int(str(rows)+str(cols)+str(i)))
        # im_ = im.copy()
        # plt.imshow(im_)
        #
        # i=2
        # plt.subplot(int(str(rows)+str(cols)+str(i)))
        # plt.imshow(igbr[:,:,1:4])
        #
        # for i in range(3, 7):
        #     plt.subplot(int(str(rows)+str(cols)+str(i)))
        #     plt.imshow(igbr[:, :, i-3], cmap='gray')
        #
        # plt.show()
        # plt.waitforbuttonpress()
