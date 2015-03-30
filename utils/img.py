__author__ = 'fnaiser'

import numpy as np
from PIL import ImageQt
from PyQt4 import QtGui
from utils.misc import get_settings


class ROI():
    def __init__(self, y=0, x=0, height=0, width=0):
        self.y_ = y
        self.x_ = x
        self.y_max_ = y + height
        self.x_max_ = x + width
        self.height_ = height
        self.width_ = width

    def y(self):
        return self.y_

    def x(self):
        return self.x_

    def height(self):
        return self.height_

    def width(self):
        return self.width_

    def top_left_corner(self):
        return np.array([self.y_, self.x_])

    def nearest_pt_in_roi(self, y, x):
        """
        :return: If the point is inside ROI, pt is returned. Else the nearest point from border is returned
        """

        y_ = y
        x_ = x
        if y < self.y_:
            y_ = self.y_
        elif y_ >= self.y_max_:
            y_ = self.y_max_ - 1

        if x_ < self.x_:
            x_ = self.x_
        elif x_ >= self.x_max_:
            x_ = self.x_max_ -1

        return np.array([y_, x_])

    def is_inside(self, pt):
        y = pt[0]
        x = pt[1]
        if y < self.y_:
            return False

        if y >= self.y_max_:
            return False

        if x < self.x_:
            return False

        if x >= self.x_max_:
            return False

        return True


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
        img_ += fill_color
        img_[border:-border, border:-border] = img
        crop = np.ones((height, width, channels), dtype=img.dtype)
        crop *= fill_color

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


def get_roi(pts):
    """
    Returns ROI tupple (y, x, height, width) - Region Of Interest for given points

    :param pts:
    :return:
    """
    x = np.min(pts[:, 1])
    width = np.max(pts[:, 1]) - x + 1
    y = np.min(pts[:, 0])
    height = np.max(pts[:, 0]) - y + 1

    roi = ROI(y, x, height, width)

    return roi


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
