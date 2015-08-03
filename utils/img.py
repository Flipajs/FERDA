__author__ = 'fnaiser'

import numpy as np
from PIL import ImageQt
from PyQt4 import QtGui
import scipy
from skimage.transform import rescale

from utils.misc import get_settings
from core.settings import Settings as S_
import cv2


def get_safe_selection(img, y, x, height, width, fill_color=(255, 255, 255)):
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
        crop = np.copy(img[y:y + height, x:x + height, :])

    return crop


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

def prepare_for_segmentation(img, project, grayscale_speedup=True):
    if project.bg_model:
        img = project.bg_model.bg_subtraction(img)

    if grayscale_speedup:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if project.arena_model:
        img = project.arena_model.mask_image(img)

    if S_.mser.gaussian_kernel_std > 0:
        img = scipy.ndimage.gaussian_filter(img, sigma=S_.mser.gaussian_kernel_std)

    if S_.mser.img_subsample_factor > 1.0:
        img = np.asarray(rescale(img, 1/S_.mser.img_subsample_factor) * 255, dtype=np.uint8)

    return img

