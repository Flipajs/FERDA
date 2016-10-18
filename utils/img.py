__author__ = 'fnaiser'

import numpy as np
from PIL import ImageQt
from PyQt4 import QtGui
import scipy
from skimage.transform import rescale

from utils.misc import get_settings
from core.settings import Settings as S_
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import math
import scipy.ndimage
from utils.roi import get_roi
import matplotlib as mpl


def get_safe_selection(img, y, x, height, width, fill_color=(255, 255, 255), return_offset=False):
    y = int(y)
    x = int(x)
    height = int(height)
    width = int(width)

    border = max(max(-y, -x), 0)

    channels = 1
    if len(img.shape) > 2:
        channels = img.shape[2]

    if len(fill_color) != channels:
        fill_color = 255

    h_ = img.shape[0] - (height + y)
    w_ = img.shape[1] - (width + x)

    border = max(border, max(max(-h_, -w_), 0))

    if border > 0:
        img_ = np.zeros((img.shape[0] + 2 * border, img.shape[1] + 2 * border, channels), dtype=img.dtype)
        img_ += np.asarray(fill_color, dtype=img.dtype)
        img_[border:-border, border:-border] = img
        crop = np.ones((height, width, channels), dtype=img.dtype)
        crop *= np.asarray(fill_color, dtype=img.dtype)

        y += border
        x += border
        crop = np.copy(img_[y:y + height, x:x + width, :])
    else:
        # TODO: why is height twice here?
        # crop = np.copy(img[y:y + height, x:x + height, :])
        crop = np.copy(img[y:y + height, x:x + width, :])

    if return_offset:
        return crop, np.array([y, x])

    return crop


def get_img_around_pts(img, pts, margin=0):
    roi = get_roi(pts)

    width = roi.width()
    height = roi.height()

    m_ = max(width, height)
    margin = m_ * margin

    y_ = roi.y() - margin
    x_ = roi.x() - margin
    height_ = height + 2 * margin
    width_ = width + 2 * margin

    crop = get_safe_selection(img, y_, x_, height_, width_, fill_color=(0, 0, 0))
    return crop, np.array([y_, x_])


def get_pixmap_from_np_bgr(np_image):
    img_q = ImageQt.QImage(np_image.data, np_image.shape[1], np_image.shape[0], np_image.shape[1] * 3, 13)
    pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

    return pix_map


def avg_circle_area_color(im, y, x, radius):
    """
    computes average color in circle area given by pos and radius
    :param im:
    :param pos:
    :param radius:
    :return:
    """

    c = np.zeros((1, 3), dtype=np.double)
    num_px = 0
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            d = ((w - radius) ** 2 + (h - radius) ** 2) ** 0.5
            if d <= radius:
                num_px += 1
                c += im[y - radius + h, x - radius + w, :]

    print num_px
    c /= num_px

    return [c[0, 0], c[0, 1], c[0, 2]]


def get_igbr_normalised(im):
    igbr = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.double)

    igbr[:, :, 0] = np.sum(im, axis=2) + 1
    igbr[:, :, 1] = im[:, :, 0] / igbr[:, :, 0]
    igbr[:, :, 2] = im[:, :, 1] / igbr[:, :, 0]
    igbr[:, :, 3] = im[:, :, 2] / igbr[:, :, 0]

    i_norm = (1 / get_settings('igbr_i_weight', float)) * get_settings('igbr_i_norm', float)
    igbr[:, :, 0] = igbr[:, :, 0] / i_norm

    return igbr


def prepare_for_visualisation(img, project):
    if project.other_parameters.img_subsample_factor > 1.0:
        img = np.asarray(rescale(img, 1/project.other_parameters.img_subsample_factor) * 255, dtype=np.uint8)

    return img


def prepare_for_segmentation(img, project, grayscale_speedup=True):
    if project.bg_model:
        img = project.bg_model.bg_subtraction(img)

    if grayscale_speedup:
        try:
            if project.other_parameters.use_only_red_channel:
                img = img[:,:,2].copy()
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if project.arena_model is not None:
        img = project.arena_model.mask_image(img)

    if project.mser_parameters.gaussian_kernel_std > 0:
        img = scipy.ndimage.gaussian_filter(img, sigma=project.mser_parameters.gaussian_kernel_std)

    if project.other_parameters.img_subsample_factor > 1.0:
        img = np.asarray(rescale(img, 1/project.other_parameters.img_subsample_factor) * 255, dtype=np.uint8)

    return img


def get_cropped_pts(region, return_roi=True, only_contour=True):
    roi_ = region.roi()
    if only_contour:
        pts = region.contour() - roi_.top_left_corner()
    else:
        pts = region.pts() - roi_.top_left_corner()

    if return_roi:
        return pts, roi_

    return pts

def imresize(im,sz):
    """  Resize an image array using PIL. """
    from PIL import Image

    pil_im = Image.fromarray(np.uint8(im))

    return np.array(pil_im.resize(sz))


def draw_pts(pts_):
    h_ = np.max(pts_[:, 0]) + 1
    w_ = np.max(pts_[:, 1]) + 1

    im = np.zeros((h_, w_), dtype=np.bool)
    im[pts_[:, 0], pts_[:, 1]] = True

    return im


def get_normalised_img(region, norm_size, blur_sigma=0):
    pts_ = get_cropped_pts(region)

    im = draw_pts(pts_)

    # plt.figure()
    # plt.imshow(im)

    if blur_sigma > 0:
        im = np.asarray(np.round(scipy.ndimage.gaussian_filter(im, sigma=blur_sigma)), dtype=np.bool)

    # plt.figure()
    # plt.imshow(im)
    # plt.show()

    im_normed = imresize(im, norm_size)
    im_out = np.zeros(im_normed.shape, dtype=np.uint8)

    # fill holes
    (cnts, _) = cv2.findContours(im_normed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    cv2.drawContours(im_out, [cnts], -1, 255, -1)

    return im_out


def replace_everything_but_pts(img, pts, fill_color=[0, 0, 0]):
    # TODO: 3 channels vs 1 channel (:, :, 1) and (:, :)
    if len(img.shape) == 2 or img.shape[3] == 1:
        if len(fill_color) > 1:
            fill_color = fill_color[0]

    new_img = np.zeros(img.shape, dtype=img.dtype)
    new_img[:, :] = fill_color

    new_img[pts[:, 0], pts[:, 1]] = img[pts[:, 0], pts[:, 1]]

    return new_img


class DistinguishableColors():
    def __init__(self, N, step=5, cmap='hsv'):
        self.step = step
        self.N = math.ceil(N/step) * step
        color_norm = colors.Normalize(vmin=0, vmax=self.N-1)
        self.scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)

    def get_color(self, index):
        i = self.step * (index % self.step) + index / self.step #   index/self.step + index % self.step
        return self.scalar_map.to_rgba(i)

def get_cmap(N, step):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''

    N = math.ceil(N/step) * step
    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='nipy_spectral')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color

def rotate_img(img, theta, center=None):
    s_ = max(img.shape[0], img.shape[1])

    im_ = np.zeros((s_, s_, img.shape[2]), dtype=img.dtype)
    h_ = (s_ - img.shape[0]) / 2
    w_ = (s_ - img.shape[1]) / 2

    im_[h_:h_+img.shape[0], w_:w_+img.shape[1], :] = img

    if center is None:
        center = (im_.shape[0] / 2, im_.shape[1] / 2)

    rot_mat = cv2.getRotationMatrix2D(center, -np.rad2deg(theta), 1.0)
    return cv2.warpAffine(im_, rot_mat, (s_, s_))

def centered_crop(img, new_h, new_w):
    new_h = int(new_h)
    new_w = int(new_w)

    h_ = img.shape[0]
    w_ = img.shape[1]

    y_ = (h_ - new_h) / 2
    x_ = (w_ - new_w) / 2

    if y_ < 0 or x_ < 0:
        Warning('cropped area cannot be bigger then original image!')
        return img

    return img[y_:y_+new_h, x_:x_+new_w, :].copy()

def get_bounding_box(r, project, relative_border=1.3, absolute_border=-1, img=None):
    from math import ceil

    if img is None:
        frame = r.frame()
        img = project.img_manager.get_whole_img(frame)

    roi = r.roi()

    height2 = int(ceil((roi.height() * relative_border) / 2.0))
    width2 = int(ceil((roi.width() * relative_border) / 2.0))

    if absolute_border > -1:
        height2 = absolute_border
        width2 = absolute_border

    x = r.centroid()[1] - width2
    y = r.centroid()[0] - height2

    bb = get_safe_selection(img, y, x, height2*2, width2*2)

    return bb, np.array([y, x])

def endpoint_rot(bb_img, pt, theta, centroid):
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta), math.cos(theta)]]
    )

    pt_ = pt-centroid
    pt = np.dot(rot, pt_.reshape(2, 1))
    new_pt = [int(round(pt[0][0] + bb_img.shape[0]/2)), int(round(pt[1][0] + bb_img.shape[1]/2))]

    return new_pt

def img_saturation(img, saturation_coef=2.0, intensity_coef=1.0):
    img_hsv = mpl.colors.rgb_to_hsv(img)
    img_hsv[:,:,1] *= saturation_coef
    img_hsv[:,:,2] *= intensity_coef
    img = mpl.colors.hsv_to_rgb(img_hsv)
    img = np.asarray(np.clip(img - img.min(), 0, 255), dtype=np.uint8)

    return img