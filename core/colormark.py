from __future__ import division
from __future__ import unicode_literals
from builtins import range
from builtins import object
from past.utils import old_div
import utils.img_temp

__author__ = 'fnaiser'

import numpy as np
from gui.gui_utils import get_settings
from core.region.mser import Mser
from core.region import mser_operations
import utils.img
import cv2


class Colormark(object):
    I_NORM = (255 * 3 + 1) * 3

    def __init__(self, color, pos=None):
        self.color_ = color
        self.color_igbr_ = None
        self.pts_ = None
        self.pos_ = pos
        self.dmap_ = None
        self.avg_intensity_ = None
        self.darkest_neighbour_intensity_ = None

    def color_igbr(self):
        """
        Returns color in Igbr space,

        :param i_norm:
        :return: np.array with 4 elements
        """
        if self.color_igbr_:
            return self.color_igbr_

        color = np.array(self.color_, dtype=np.float)
        s = np.sum(color) + 1
        i_norm = (old_div(1,get_settings('igbr_i_weight', float))) * get_settings('igbr_i_norm', float)
        c = np.array([old_div(s,(i_norm)), old_div(color[0],s), old_div(color[1],s), old_div(color[2],s)])

        self.color_igbr_ = c

        return np.copy(self.color_igbr_)

    def update_color(self, color):
        self.color_ = color
        self.color_igbr_ = None

    def set_pts(self, pts):
        self.pts_ = np.copy(pts)

    def set_pos(self, pos):
        self.pos = np.copy(pos)

    def set_dmap(self, dmap):
        self.dmap_ = np.copy(dmap)

    def set_avg_intensity(self, avg_intensity):
        self.avg_intensity_ = avg_intensity

    def set_darkest_neighbour_intensity(self, neigh_intensity):
        self.darkest_neighbour_intensity_ = neigh_intensity

def get_colormark_mser(num_px):
    max_area_ = get_settings('colormarks_mser_max_area', int)
    min_area_ = get_settings('colormarks_mser_min_area', int)
    min_margin_ = get_settings('colormarks_mser_min_margin', int)
    mser = Mser(max_area=max_area_, min_area=min_area_, min_margin=min_margin_)

    return mser


def get_colormark(img, color, position, radius, colormark_radius=-1):
    """
    if colormark_radius=-1, the auto detected settings from initialization will be used.
    :param img:
    :param color:
    :param position:
    :param radius:
    :param colormark_radius:
    :return:
    """
    num_px = (2*radius)**2
    mser = get_colormark_mser(num_px)

    y_ = position[0] - radius
    x_ = position[1] - radius

    c = Colormark(color)

    img_crop = utils.img.get_safe_selection(img, y_, x_, radius*2, radius*2)
    igbr = utils.img_temp.get_igbr_normalised(img_crop)

    dist_im = np.linalg.norm(igbr - c.color_igbr(), axis=2)
    dist_im /= np.max(dist_im)
    dist_im = np.asarray(dist_im * 255, dtype=np.uint8)
    regions = mser.process_image(dist_im)
    groups = mser_operations.get_region_groups(regions)
    ids = mser_operations.margin_filter(regions, groups)
    regions = [regions[i] for i in ids]

    if len(regions) == 0:
        return None, -1, -1, dist_im

    avg_intensity = [old_div(np.sum(dist_im[p.pts()[:, 0], p.pts()[:, 1]]), p.area()) for p in regions]
    avg_radius = get_settings('colormarks_avg_radius', float)
    darkest_neighbour = [darkest_neighbour_square(img_crop, r.centroid(), avg_radius*2) for r in regions]

    val = np.array(avg_intensity) + np.array(darkest_neighbour)
    order = np.argsort(val)

    ids = np.asarray(order[0:1], dtype=np.int32)

    selected_r = [regions[id] for id in ids]

    offset = np.array([y_, x_])
    c.set_pts(selected_r[0].pts() + offset)
    c.set_avg_intensity(avg_intensity[ids[0]])
    c.set_darkest_neighbour_intensity(avg_intensity[ids[0]])
    c.set_pos(selected_r[0].centroid()+offset)

    if get_settings('colormarks_debug'):
        c.set_dmap(dist_im)

    return c


def darkest_neighbour_square(im, pt, square_size):
    # if id == 0:
    #     position = 'top-left'
    # elif id == 1:
    #     position = 'top'
    # elif id == 2:
    #     position = 'top-right'
    # elif id == 3:
    #     position = 'left'
    # elif id == 4:
    #     position = 'right'
    # elif id == 5:
    #     position = 'bottom-left'
    # elif id == 6:
    #     position = 'bottom'
    # elif id == 7:
    #     position = 'bottom-right'

    squares = []
    start = [pt[0] - square_size - (old_div(square_size, 2)), pt[1] - square_size - (old_div(square_size, 2))]

    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue

            crop = utils.img.get_safe_selection(im, start[0] + square_size * i, start[1] + square_size * j, square_size,
                                                square_size, fill_color=(255, 255, 255))
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            s = np.sum(crop_gray)

            squares.append(s)

    id = np.argmin(squares)


    return old_div(squares[id], square_size ** 2)




