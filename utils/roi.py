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

    def __str__(self):
        s = ""
        s += "y: {}".format(self.y_)
        s += "\nx: {}".format(self.x_)
        s += "\nheight: {}".format(self.height_)
        s += "\nwidth: {}".format(self.width_)

        return s

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

    def is_inside(self, pt, tolerance=0):
        y = pt[0]
        x = pt[1]
        if y < self.y_ - tolerance:
            return False

        if y > self.y_max_ + tolerance:
            return False

        if x < self.x_ - tolerance:
            return False

        if x > self.x_max_ + tolerance:
            return False

        return True

    def corner_pts(self):
        return np.array([
            [self.y_, self.x_],
            [self.y_, self.x_max_],
            [self.y_max_, self.x_max_],
            [self.y_max_, self.x_]
        ])

    def slices(self):
        return [slice(self.y_, self.y_max_), slice(self.x_, self.x_max_)]

    def safe_roi(self, img, border=30):
        y_ = max(0, self.y_-border)
        y_max_ = min(img.shape[0], self.y_max_+border)

        x_ = max(0, self.x_-border)
        x_max_ = min(img.shape[1], self.x_max_+border)
        return img[[slice(y_, y_max_), slice(x_, x_max_)]].copy()

    def expand(self, border):
        return ROI(self.y_ - border,
                   self.x_ - border,
                   self.height_ + 2*border,
                   self.width_ + 2*border)

    def safe_expand(self, border, image):
        y = max(0, self.y_ - border)
        x = max(0, self.x_ - border)
        h = min(image.shape[0] - y - 1, self.height_ + 2 * border)
        w = min(image.shape[1] - x - 1, self.width_ + 2 * border)
        return ROI(y,
                   x,
                   h,
                   w)

    def is_intersecting(self, roi2):
        """
        returns True even when they intersects by edge
        Args:
            roi2:

        Returns:

        """
        return not(self.x_ > roi2.x() + roi2.width() or
                   self.x_max_ < roi2.x() or
                   self.y_ > roi2.y() + roi2.height() or
                   self.y_max_ < roi2.y()
                   )


    def is_intersecting_expanded(self, roi2, offset):
        return self.expand(offset).is_intersecting(roi2)

    def union(self, roi):
        y_ = min(self.y_, roi.y_)
        x_ = min(self.x_, roi.x_)
        height = max(self.y_max_, roi.y_max_) - y_
        width = max(self.x_max_, roi.x_max_) - x_

        return ROI(y_, x_, height, width)

def get_roi(pts=None):
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


def get_roi_rle(rle):
    x = np.inf
    y = np.inf
    xx = 0
    yy = 0

    for row in rle:
        x = min(x, row['col1'])
        xx = max(xx, row['col2'])

        y = min(y, row['line'])
        yy = max(yy, row['line'])

    return ROI(y, x, yy - y + 1, xx - x + 1)
