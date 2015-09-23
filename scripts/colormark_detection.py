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


I_NORM = 766 * 3 * 2


def igbr_transformation(im):
    igbr = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.double)

    igbr[:,:,0] = np.sum(im,axis=2) + 1
    igbr[:, :, 1] = im[:,:,0] / igbr[:,:,0]
    igbr[:,:,2] = im[:,:,1] / igbr[:,:,0]
    igbr[:,:,3] = im[:,:,2] / igbr[:,:,0]

    igbr[:,:,0] = igbr[:,:,0] / I_NORM

    return igbr


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
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


def colormarks_labelling(image, colors):

    pass


if __name__ == "__main__":
    vid = VideoManager('/Users/flipajs/Documents/wd/C210min.avi')

    fig = plt.figure()
    # plt.ion()
    for i in range(0, 1000, 100):
        im = vid.seek_frame(i)

        CLUSTER_N = 10

        image = im

        # load the image and convert it from BGR to RGB so that
        # we can dispaly it with matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segments = quickshift(image, kernel_size=10, convert2lab=True, max_dist=80, ratio=0.9)

        plt.imshow(segments)
        plt.show()
        # pass
        #
        # # show our image
        # plt.figure()
        # plt.axis("off")
        # plt.imshow(image)
        #
        # # reshape the image to be a list of pixels
        # image = image.reshape((image.shape[0] * image.shape[1], 3))
        #
        # # cluster the pixel intensities
        # clt = KMeans(n_clusters = CLUSTER_N)
        # clt.fit(image)
        #
        # # build a histogram of clusters and then create a figure
        # # representing the number of pixels labeled to each color
        # hist = centroid_histogram(clt)
        # bar = plot_colors(hist, clt.cluster_centers_)
        #
        # # show our color bart
        # plt.figure()
        # plt.axis("off")
        # plt.imshow(bar)
        # plt.show()
        #
        # #
        # #
        #
        # # blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
        # # # Compute radii in the 3rd column.
        # # blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
        # #
        # # blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
        # # blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
        #
        # blobs_dog = blob_dog(image_gray, min_sigma=1, max_sigma=5, threshold=.1)
        #
        # blobs_list = [blobs_dog]
        # colors = ['red']
        # titles = ['Difference of Gaussians']
        # sequence = zip(blobs_list, colors, titles)
        #
        # for blobs, color, title in sequence:
        #     fig, ax = plt.subplots(1, 1)
        #     ax.set_title(title)
        #     ax.imshow(image, interpolation='nearest')
        #     for blob in blobs:
        #         y, x, r = blob
        #         c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        #         ax.add_patch(c)
        #
        # plt.show()
        #
        #
        #
        # igbr = igbr_transformation(im)
        #
        # rows = 2
        # cols = 3
        #
        # i = 1
        # plt.subplot(int(str(rows)+str(cols)+str(i)))
        # im_ = im.copy()
        # im_[:,:,0] = im[:,:,2]
        # im_[:,:,2] = im[:,:,0]
        # plt.imshow(im_)
        #
        # for i in range(2, 6):
        #     plt.subplot(int(str(rows)+str(cols)+str(i)))
        #     plt.imshow(igbr[:, :, i-2], cmap='gray')
        #
        # from skimage import color
        # lab = color.rgb2lab(im)
        #
        # # for i in range(6, 9):
        # #     plt.subplot(int(str(cols)+str(rows)+str(i)))
        # #     plt.imshow(lab[:, :, i-6], cmap='gray')
        #
        # plt.show()
        # plt.waitforbuttonpress()