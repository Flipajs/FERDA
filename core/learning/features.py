from skimage.measure import moments_central, moments_hu, moments_normalized, moments
from utils.img_manager import ImgManager
import cv2
from utils.img import get_img_around_pts, replace_everything_but_pts
import cPickle as pickle
import numpy as np
from libs.intervaltree.intervaltree import IntervalTree
from gui.learning.ids_names_widget import IdsNamesWidget
from PyQt4 import QtGui
import sys
import operator
import time
import itertools
import math
from utils.img import rotate_img, centered_crop, get_bounding_box, endpoint_rot
import warnings
from features import *


def get_hu_moments(img):
    m = moments(img)
    cr = m[0, 1] / m[0, 0]
    cc = m[1, 0] / m[0, 0]

    mu = moments_central(img, cr, cc)
    nu = moments_normalized(mu)
    hu = moments_hu(nu)

    features = [m_ for m_ in hu]

    return features

def __hu2str(vec):
    s = ''
    for i in vec:
        s += '\t{}\n'.format(i)

    return s


def features2str_var1(vec):
    s = 'area: {} cont. len: {} \n' \
        'axis major:{:.2f} min: {:.2f} ratio: {:.2f}\n'.format(vec[0], vec[1], vec[2], vec[3], vec[4])

    #
    # s = "area: " + str(vec[0]) + " cont len: " + str(vec[1]) + " major ax : " + str(vec[2]) + "minor ax: " + str(vec[3]) + \
    #     "ax ratio: " + str(vec[4]) + "\npts bin HU: "
    #
    # for i in range(7, 7+7):
    #     s = s + " " + str(vec[i])
    #
    # s += "\n"

    HU_START = 7
    HU_LEN = 7

    start = HU_START+5*HU_LEN
    s += __hu2str(vec[start:start+HU_LEN])

    return s

def get_features_var1(r, p):
    f = []
    # area
    f.append(r.area())

    # # area, modifications
    # f.append(r.area()**0.5)
    # f.append(r.area()**2)
    #
    # contour length
    f.append(len(r.contour()))

    # major axis
    f.append(r.a_)

    # minor axis
    f.append(r.b_)

    # axis ratio
    f.append(r.a_ / r.b_)

    # axis ratio sqrt
    f.append((r.a_ / r.b_)**0.5)

    # axis ratio to power of 2
    f.append((r.a_ / r.b_)**2.0)

    img = p.img_manager.get_whole_img(r.frame_)
    crop, offset = get_img_around_pts(img, r.pts())

    pts_ = r.pts() - offset

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)

    ###### MOMENTS #####
    # #### BINARY
    crop_b_mask = replace_everything_but_pts(np.ones(crop_gray.shape, dtype=np.uint8), pts_)
    f.extend(get_hu_moments(crop_b_mask))


    #### ONLY MSER PXs
    # in GRAY
    crop_gray_masked = replace_everything_but_pts(crop_gray, pts_)
    f.extend(get_hu_moments(crop_gray_masked))

    # B G R
    for i in range(3):
        crop_ith_channel_masked = replace_everything_but_pts(crop[:, :, i], pts_)
        f.extend(get_hu_moments(crop_ith_channel_masked))

    # min, max from moments head/tail
    relative_border = 2.0

    bb, offset = get_bounding_box(r, p, relative_border)
    p_ = np.array([r.a_*math.sin(-r.theta_), r.a_*math.cos(-r.theta_)])
    endpoint1 = np.ceil(r.centroid() + p_) + np.array([1, 1])
    endpoint2 = np.ceil(r.centroid() - p_) - np.array([1, 1])

    bb = rotate_img(bb, r.theta_)
    bb = centered_crop(bb, 8*r.b_, 4*r.a_)

    c_ = endpoint_rot(bb, r.centroid(), -r.theta_, r.centroid())

    endpoint1_ = endpoint_rot(bb, endpoint1, -r.theta_, r.centroid())
    endpoint2_ = endpoint_rot(bb, endpoint2, -r.theta_, r.centroid())
    if endpoint1_[1] > endpoint2_[1]:
        endpoint1_, endpoint2_ = endpoint2_, endpoint1_

    y_ = int(c_[0] - r.b_)
    y2_ = int(c_[0]+r.b_)
    x_ = int(c_[1] - r.a_)
    x2_ = int(c_[1] + r.a_)
    im1_ = bb[y_:y2_, x_:int(c_[1]), :].copy()
    im2_ = bb[y_:y2_, int(c_[1]):x2_, :].copy()

    # ### ALL PXs in crop image given margin
    # crop, offset = get_img_around_pts(img, r.pts(), margin=0.3)
    #
    # # in GRAY
    # crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # f.extend(self.get_hu_moments(crop_gray))
    #

    # B G R
    for i in range(3):
        hu1 = get_hu_moments(im1_[:, :, i])
        hu2 = get_hu_moments(im2_[:, :, i])

        f.extend(list(np.min(np.vstack([hu1, hu2]), axis=0)))
        f.extend(list(np.max(np.vstack([hu1, hu2]), axis=0)))

    return f


    crop_ = np.asarray(crop, dtype=np.int32)

    # # R G combination
    # crop_rg = crop_[:, :, 1] + crop_[:, :, 2]
    # f.extend(self.get_hu_moments(crop_rg))
    #
    # # B G
    # crop_bg = crop_[:, :, 0] + crop_[:, :, 1]
    # f.extend(self.get_hu_moments(crop_bg))
    #
    # # B R
    # crop_br = crop_[:, :, 0] + crop_[:, :, 2]
    # f.extend(self.get_hu_moments(crop_br))


def __process_crops(crops, fliplr):
    from skimage.feature import hog

    f = []

    for crop in crops:
        if fliplr:
            crop = np.fliplr(crop)

        h, w = crop.shape

        fd = hog(crop, orientations=8, pixels_per_cell=(w, h),
                            cells_per_block=(1, 1), visualise=False)

        f.extend(fd)

        fd2 = hog(crop, orientations=8, pixels_per_cell=(w/4, h),
                            cells_per_block=(1, 1), visualise=False)

        f.extend(fd2)

    return f


def get_features_var2(r, p, fliplr=False):
    img = p.img_manager.get_whole_img(r.frame_)

    crop, offset = get_img_around_pts(img, r.pts(), margin=2.0)
    crop = rotate_img(crop, r.theta_)

    margin = 3

    crop = centered_crop(crop, 2 * (r.b_ + margin), 2 * (r.a_ + margin))

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_r = crop[:, :, 2]
    crop_g = crop[:, :, 1]
    crop_b = crop[:, :, 0]

    crops = [crop_gray, crop_r, crop_b, crop_g]

    if fliplr:
        f1 = __process_crops(crops, fliplr=False)
        f2 = __process_crops(crops, fliplr=True)

        return f1, f2
    else:
        f = __process_crops(crops, fliplr=False)

        return f


def get_features_var3(r, p, fliplr=False):
    f1 = get_features_var1(r, p)
    if fliplr:
        f2_a, f2_b = get_features_var2(r, p, fliplr=fliplr)
        return f1 + f2_a, f1 + f2_b
    else:
        f2= get_features_var2(r, p, fliplr=fliplr)
        return f1 + f2