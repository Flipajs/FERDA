__author__ = 'flipajs'

import numpy as np


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

    def bottom_right_corner(self):
        return np.array([self.y_max_, self.x_max_])

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

    def is_inside(self, pt, strict=True):
        y = pt[0]
        x = pt[1]
        if y < self.y_:
            return False

        if y > self.y_max_ or strict and y == self.y_max_:
            return False

        if x < self.x_:
            return False

        if x > self.x_max_ or strict and x == self.x_max_:
            return False

        return True

    def corner_pts(self):
        return np.array([
            [self.y_, self.x_],
            [self.y_, self.x_max_],
            [self.y_max_, self.x_max_],
            [self.y_max_, self.x_]
        ])


def get_roi(pts):
    """
    Returns ROI class - Region Of Interest for given points

    :param pts:
    :return:
    """
    x = np.min(pts[:, 1])
    width = np.max(pts[:, 1]) - x + 1
    y = np.min(pts[:, 0])
    height = np.max(pts[:, 0]) - y + 1

    roi = ROI(y, x, height, width)

    return roi