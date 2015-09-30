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

    colors[0:4, :] = [255, 255, 255]
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

def compute_saturation(im):
    igbr = igbr_transformation(im)
    out_im = np.sum((igbr[:, :, 1:4] - np.array([1.0/3., 1.0/3., 1.0/3.]))**2, axis=2)

    return out_im

def compute_saturation_(im):
    from skimage import color
    import time
    s = time.time()
    lab = color.rgb2lab(im)
    print time.time()-s

    out_im = np.sum(lab[:,:,1:2]**2, axis=2)

    m_ = np.max(out_im) / 4.
    print m_
    out_im[out_im > m_] = m_
    print np.max(out_im)

    return out_im

def color_candidate_pixels_slow(im):
    from numpy.linalg import norm

    MIN_WHITE_DIST = 180
    MIN_GRAY_DIST = 10
    TOO_DARK = 60

    WHITE = np.array([[255, 255, 255]])

    use = []
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            px = im[y, x, :]
            if norm(px - WHITE) > MIN_WHITE_DIST:
                if np.sum(px) < TOO_DARK:
                    use.append((y, x))
                elif (norm(np.cross(WHITE, -px)) / norm(WHITE)) > MIN_GRAY_DIST:
                    use.append((y, x))

    result = np.ones((im.shape[0], im.shape[1], 3), dtype=np.uint8)*255
    for px in use:
        result[px[0], px[1], :] = im[px[0], px[1], :]

    return result

def color_candidate_pixels(im):
    s = time.time()
    im_copy = im.copy()
    print "im_copy t: ", time.time() - s

    s = time.time()
    im_ = im.reshape(im.shape[0] * im.shape[1], im.shape[2])
    print "im reshape t: ", time.time() - s

    MIN_WHITE_DIST = 180
    MIN_GRAY_DIST = 10

    s = time.time()
    dists = cdist(im_, np.array([[255, 255, 255]]), 'euclidean')
    print "distances t: ", time.time() - s

    s = time.time()
    ids = dists < MIN_WHITE_DIST
    print "thresholding t: ", time.time() - s

    not_ids = np.logical_not(ids)

    # # remove GRAY
    # from numpy.linalg import norm
    # l2 = np.array([[255, 255, 255]])
    # s = time.time()
    # gray_dists = norm(np.cross(l2, -im_[np.logical_not(ids[:, 0]), :]), axis=1) / norm(l2)
    # are_gray = gray_dists < MIN_GRAY_DIST
    #
    # ids[ids == 0] = are_gray
    # print "gray dist t: ", time.time()-s
    remove = ids.reshape((im.shape[0], im.shape[1]))
    print np.sum(remove)/float((im.shape[0] * im.shape[1]))
    #
    im_copy[remove] = [255, 255, 255]

    return im_copy

def on_key_event(event):
    global frame
    key = event.key

    # In Python 2.x, the key gets indicated as "alt+[key]"
    # Bypass this bug:
    if key.find('alt') == 0:
        key = key.split('+')[1]

    if key in ['n', 'right']:
        frame += 1
    elif key in ['N']:
        frame += 50
    elif key in ['B']:
        frame -= 50
        if frame < 0:
            frame = 0
    elif key in ['b', 'B', 'left']:
        if frame > 0:
            frame -= 1
    elif key in ['q']:
        quit_ = True

if __name__ == "__main__":
    vid = VideoManager('/Users/flipajs/Documents/wd/C210min.avi')

    plt.ion()
    # fig = plt.figure(1)
    # fig.canvas.mpl_connect('button_press_event', onclick)
    # for i in range(200, 1000, 100):
    frame = 0
    while True:
        plt.figure(1)
        im = vid.seek_frame(frame)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im)

        fig = plt.figure(2)
        fig.canvas.mpl_connect('key_press_event', on_key_event)
        # plt.title('lab')
        # out_im = color_candidate_pixels_slow(im)
        out_im = color_candidate_pixels(im)
        colors = np.array([[46, 34, 21], # ANT
                           [158, 139, 131], [175, 160, 152], [188, 176, 167], # BG
                           [27, 54, 39], # DARK GREEN
                           [58, 75, 98], [37, 71, 107], # BLUE
                           [183, 142, 174], [158, 126, 152], # PINK
                           [56, 48, 58], # PURPLE
                           [146, 47, 51], # RED
                           [167, 137, 103], # ORANGE
                           ])

        out_im = colormarks_labelling(out_im, colors)
        plt.imshow(out_im)

        # plt.figure(3)
        # plt.title('irgb')
        # out_im = compute_saturation(im)
        # plt.imshow(out_im)
        plt.show()
        plt.waitforbuttonpress()

        #
        # # plt.hold(True)
        # # plt.scatter(coords[:,0], coords[:,1])
        # # plt.hold(False)
        #
        # # colors = np.array([[46, 34, 21], [216, 209, 217], [208, 195, 184], [54, 89, 120], [167, 140, 95],
        # #                    [148, 52, 56], [168, 125, 144], [36, 58, 96], [122, 103, 110], [199, 190, 196],
        # #                    [163, 123, 137]])
        # # colormarks_labelling(im, colors)
        #
        # #############
        # import time
        #
        # s = time.time()
        # igbr = igbr_transformation(im)
        #
        # labels = colormarks_labelling(igbr, igbr_colors, orig_colors)
        # print "TIME: ", time.time() - s
        # plt.figure()
        # plt.imshow(labels)
        # plt.show()
        # plt.waitforbuttonpress()
        #
        # # rows = 2
        # # cols = 3
        # #
        # # i = 1
        # # plt.figure()
        # # plt.subplot(int(str(rows)+str(cols)+str(i)))
        # # im_ = im.copy()
        # # plt.imshow(im_)
        # #
        # # i=2
        # # plt.subplot(int(str(rows)+str(cols)+str(i)))
        # # plt.imshow(igbr[:,:,1:4])
        # #
        # # for i in range(3, 7):
        # #     plt.subplot(int(str(rows)+str(cols)+str(i)))
        # #     plt.imshow(igbr[:, :, i-3], cmap='gray')
        # #
        # # plt.show()
        # # plt.waitforbuttonpress()
