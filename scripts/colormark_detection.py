from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from builtins import zip
from builtins import range
from past.utils import old_div
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

I_NORM = 766 * 3 * 2


def igbr_transformation(im):
    igbr = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.double)

    igbr[:, :, 0] = np.sum(im, axis=2) + 1
    igbr[:, :, 1] = old_div(im[:, :, 0], igbr[:, :, 0])
    igbr[:, :, 2] = old_div(im[:, :, 1], igbr[:, :, 0])
    igbr[:, :, 3] = old_div(im[:, :, 2], igbr[:, :, 0])

    igbr[:, :, 0] = old_div(igbr[:, :, 0], I_NORM)

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

    # colors[0:5, :] = [255, 255, 255]
    if original_colors is None:
        out = np.asarray(colors[labels], dtype=np.uint8)
    else:
        out = np.asarray(original_colors[labels], dtype=np.uint8)

    # set all these to background
    labels[labels < NUM_BG_COLORS] = 0

    return out, labels


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d' % (
        ix, iy))

    # global coords
    # coords.append((ix, iy))
    print(ix, iy)
    # if len(coords) == 2:
    #     fig.canvas.mpl_disconnect(cid)

def compute_saturation(im):
    igbr = igbr_transformation(im)
    out_im = np.sum((igbr[:, :, 1:4] - np.array([old_div(1.0,3.), old_div(1.0,3.), old_div(1.0,3.)]))**2, axis=2)**0.5

    MIN_I = 100

    out_im[np.sum(im, axis=2) < MIN_I] = 0

    return out_im

def compute_saturation_(im):
    from skimage import color
    import time
    s = time.time()
    lab = color.rgb2lab(im)
    print(time.time()-s)

    out_im = np.sum(lab[:,:,1:2]**2, axis=2)

    m_ = old_div(np.max(out_im), 4.)
    print(m_)
    out_im[out_im > m_] = m_
    print(np.max(out_im))

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
                elif (old_div(norm(np.cross(WHITE, -px)), norm(WHITE))) > MIN_GRAY_DIST:
                    use.append((y, x))

    result = np.ones((im.shape[0], im.shape[1], 3), dtype=np.uint8)*255
    for px in use:
        result[px[0], px[1], :] = im[px[0], px[1], :]

    return result

def color_candidate_pixels(im):
    s = time.time()
    im_copy = im.copy()
    # print "im_copy t: ", time.time() - s

    s = time.time()
    im_ = im.reshape(im.shape[0] * im.shape[1], im.shape[2])
    # print "im reshape t: ", time.time() - s

    MIN_WHITE_DIST = 180
    MIN_GRAY_DIST = 10

    s = time.time()
    dists = cdist(im_, np.array([[255, 255, 255]]), 'euclidean')
    # print "distances t: ", time.time() - s

    s = time.time()
    ids = dists < MIN_WHITE_DIST
    # print "thresholding t: ", time.time() - s

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
    print(old_div(np.sum(remove),float((im.shape[0] * im.shape[1]))))
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

def get_area_from_integral_im(i_im, center, square_size):
    sq2 = old_div(square_size,2)
    if sq2 < center[0] < i_im.shape[0] - sq2 and sq2 < center[1] < i_im.shape[1] - sq2:
        p0 = i_im[center[0] - sq2, center[1] - sq2]
        p1 = i_im[center[0] - sq2, center[1] + sq2]
        p2 = i_im[center[0] + sq2, center[1] - sq2]
        p3 = i_im[center[0] + sq2, center[1] + sq2]

        # see http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html#integral
        return old_div((p0 + p3 - p1 - p2), square_size**2)

    return np.inf

def test_dark_neighbourhood(i_im, c):
    import itertools

    SQUARE_SIZE = 13
    MEAN_INTENSITY_THRESHOLD = 50

    # test all 8 directins
    a = [-SQUARE_SIZE, 0, SQUARE_SIZE]
    found = True
    for r in itertools.product(a, a):
        if r[0] == r[1] == 0:
            continue

        if get_area_from_integral_im(i_im, c + np.array(r), SQUARE_SIZE) < MEAN_INTENSITY_THRESHOLD:
            if found:
                return True
            else:
                found = True

    return False

def process_ccs_(im, labels, integral_im):
    from skimage.measure import label
    labels, num = label(labels, return_num=True, neighbors=4)

    rest_num = 0
    for i in range(num):
        ids = labels == i
        px_num = np.sum(ids)
        if px_num < 20:
            labels[ids] = 0
            im[ids] = [255, 255, 255]
        else:
            labels[ids] = rest_num
            rest_num += 1

    print("#CC after min area thresh: ", rest_num)
    plt.figure(3)
    plt.imshow(im)
    # plt.waitforbuttonpress()

    num = rest_num
    rest_num = 0
    for i in range(num):
        ids = labels == i
        px_num = np.sum(ids)
        if px_num > 200:
            labels[ids] = 0
            im[ids] = [255, 255, 255]
        else:
            labels[ids] = rest_num
            rest_num += 1

    print("#CC after max area thresh: ", rest_num)
    plt.figure(3)
    plt.imshow(im)
    # plt.waitforbuttonpress()

    num = rest_num
    rest_num = 0
    for i in range(num):
        ids = labels == i

        coords = np.argwhere(labels == i)
        c = np.mean(coords, axis=0)

        M00 = coords.shape[0]
        M11 = np.sum(coords[:, 0] * coords[:, 1])
        M20 = np.sum(coords[:, 0]**2)
        M02 = np.sum(coords[:, 1]**2)

        u20 = old_div(M20,float(M00)) - c[0]**2
        u02 = old_div(M02,float(M00)) - c[1]**2
        u11 = old_div(M11,float(M00)) - c[0]*c[1]

        part2 = old_div(((4*u11**2 + (u20 - u02)**2)**0.5), 2.)
        lambda1 = old_div((u20 + u02), 2.)
        lambda2 = lambda1 - part2
        lambda1 += part2

        eccentricity = (1 - old_div(lambda2,lambda1)) ** 0.5
        print(eccentricity, lambda1, lambda2)

        std_ = np.std(coords, axis=0)
        is_std_ok = True if np.sum(std_) < 9 else False

        if eccentricity > 0.8:
        # if not is_std_ok:
            labels[ids] = 0
            im[ids] = [255, 255, 255]
        else:
            labels[ids] = rest_num
            rest_num += 1

    print("#CC after std thresh: ", rest_num)
    plt.figure(3)
    plt.imshow(im)
    plt.waitforbuttonpress()

    num = rest_num
    rest_num = 0
    for i in range(num):
        ids = labels == i

        coords = np.argwhere(labels == i)
        c = np.mean(coords, axis=0)
        if not test_dark_neighbourhood(integral_im, c):
            labels[ids] = 0
            im[ids] = [255, 255, 255]
        else:
            labels[ids] = rest_num
            rest_num += 1

    print("#CC after dark neighbour thresh: ", rest_num)
    plt.figure(3)
    plt.imshow(im)
    plt.waitforbuttonpress()

    num = rest_num
    rest_num = 0
    for i in range(num):
        ids = labels == i
        px_num = np.sum(ids)
        if px_num < 10:
            labels[ids] = 0
        elif px_num > 200:
            labels[ids] = 0
        else:
            coords = np.argwhere(labels == i)
            c = np.mean(coords, axis=0)
            std_ = np.std(coords, axis=0)
            is_std_ok = True if np.sum(std_) < 9 else False
            # if is_std_ok:
            if is_std_ok and test_dark_neighbourhood(integral_im, c):
                rest_num += 1
                labels[ids] = rest_num
            else:
                labels[ids] = 0


def process_ccs(im, integral_im):
    from skimage.measure import label
    from skimage.morphology import erosion, square
    # im = erosion(im, square(2))

    labels, num = label(im, return_num=True, neighbors=4)

    rest_num = 0
    for i in range(num+1):
        ids = labels == i
        px_num = np.sum(ids)
        if px_num < 10:
            labels[ids] = 0
        elif px_num > 200:
            labels[ids] = 0
        else:
            coords = np.argwhere(labels == i)
            c = np.mean(coords, axis=0)

            M00 = coords.shape[0]
            M11 = np.sum(coords[:, 0] * coords[:, 1])
            M20 = np.sum(coords[:, 0]**2)
            M02 = np.sum(coords[:, 1]**2)

            u20 = old_div(M20,float(M00)) - c[0]**2
            u02 = old_div(M02,float(M00)) - c[1]**2
            u11 = old_div(M11,float(M00)) - c[0]*c[1]

            part2 = old_div(((4*u11**2 + (u20 - u02)**2)**0.5), 2.)
            lambda1 = old_div((u20 + u02), 2.)
            lambda2 = lambda1 - part2
            lambda1 += part2

            eccentricity = (1 - old_div(lambda2,lambda1)) ** 0.5

            is_eccentricity_ok = True if eccentricity < 0.8 else False
            if is_eccentricity_ok and test_dark_neighbourhood(integral_im, c):
                rest_num += 1
                labels[ids] = rest_num
            else:
                labels[ids] = 0

    print("rest num: ", rest_num)
    # print "Number of components:", num
    plt.figure(3)
    plt.imshow(labels, cmap='jet')



if __name__ == "__main__":
    # vid = VideoManager('/Users/flipajs/Documents/wd/C210min.avi')
    vid = VideoManager('/Users/flipajs/Documents/wd/bigLense_clip.avi')

    plt.ion()
    frame = 0
    fig = plt.figure(1)
    fig.canvas.mpl_connect('key_press_event', on_key_event)
    fig = plt.figure(2)
    fig.canvas.mpl_connect('key_press_event', on_key_event)
    fig = plt.figure(3)
    fig.canvas.mpl_connect('key_press_event', on_key_event)

    NUM_BG_COLORS = 2
    colors = np.array([[46, 34, 21], [88, 63, 65], # ANT
                           [158, 139, 131], [175, 160, 152], [188, 176, 167], [174, 94, 73], # BG
                           [27, 54, 39], # DARK GREEN
                           [37, 71, 107], # BLUE
                           [158, 126, 152], # PINK
                           [56, 48, 58], # PURPLE
                           [146, 47, 51], # RED
                           [158, 130, 90], # ORANGE
                           ])

    colors = np.array([[51, 35, 36], # ANT
                           [176, 175, 168], # BG
                           [141, 71, 99], # PINK
                           [173, 179, 174], # WHITE
                           [35, 68, 48], # DARK GREEN
                           [34, 36, 56], # DARK BLUE
                           [139, 126, 83] # ORANGE
                           ])

    old_frame = -1
    while True:
        if old_frame != frame:
            print("FRAME: ", frame)
            plt.figure(1)
            im = vid.seek_frame(frame)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            plt.imshow(im)

            fig = plt.figure(2)
            out_im = color_candidate_pixels(im)

            out_im, labels = colormarks_labelling(out_im, colors.copy())
            plt.imshow(out_im)

            # integral_im = cv2.integral(cv2.cvtColor(im , cv2.COLOR_RGB2GRAY))
            # process_ccs(labels, integral_im)

            plt.show()

            old_frame = frame

        plt.waitforbuttonpress()