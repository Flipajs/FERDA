import cv2
import numpy as np


class Shape(object):
    @classmethod
    def from_region(cls, region):
        pass

    def to_region(self):
        pass

    @classmethod
    def from_dict(cls, region_dict, frame=None):
        pass

    def __init__(self, frame=None):
        self.frame = frame

    def __str__(self):
        pass

    def to_dict(self):
        return self.__dict__.copy()

    @property
    def area(self):
        return cv2.contourArea(self.to_poly())

    def to_poly(self):
        pass

    def get_overlap(self, another_object):
        area, poly = cv2.intersectConvexConvex(self.to_poly(), another_object.to_poly())
        return area

    def is_close(self, another_object, thresh_px=None):
        return np.linalg.norm(self.xy - another_object.xy) < thresh_px

    def is_outside_bounds(self, x1, y1, x2, y2):
        pass

    def to_array(self):
        pass

    def rotate(self, angle_deg_cw, rotation_center_xy=None):
        pass

    def move(self, delta_xy):
        pass

    def draw(self, ax=None, label=None, color=None):
        pass

    def __add__(self, other):
        """
        Return mean object.

        :param other: Ellipse
        :return: mean Ellipse
        """
        pass
