from __future__ import division
from __future__ import unicode_literals
from builtins import object
from past.utils import old_div
import cv2
import numpy as np

from core.region.region import Region
from core.region.ep import p2e, e2p, column


class Ellipse(object):
    @classmethod
    def from_region(cls, region):
        yx = region.centroid()
        tmp = cls(yx[1], yx[0], -np.rad2deg(region.theta_), 2 * region.major_axis_, 2 * region.minor_axis_,
                  region.frame())
        return tmp

    def to_region(self):
        r = Region(is_origin_interaction=True, frame=self.frame)
        r.centroid_ = self.xy[::-1]
        r.theta_ = -np.deg2rad(self.angle_deg)
        r.major_axis_ = old_div(self.major, 2)
        r.minor_axis_ = old_div(self.minor, 2)
        return r

    @classmethod
    def from_dict(cls, region_dict):
        return cls(region_dict['0_x'], region_dict['0_y'], region_dict['0_angle_deg_cw'], region_dict['0_major'],
                   region_dict['0_minor'])

    def __init__(self, x=None, y=None, angle_deg=None, major=None, minor=None, frame=None):
        self.x = x
        self.y = y
        self.angle_deg = angle_deg  # positive means clockwise (image axes)
        self.major = major  # == 2 * semi major axis
        self.minor = minor  # == 2 * semi minor axis
        self.frame = frame

    def __str__(self):
        return('Ellipse xy ({x:.1f},{y:.1f}), angle {angle_deg:.1f} deg, major {major:.1f} px, minor {minor:.1f} px, '
               'frame {frame}'.format(**self.__dict__))

    def to_dict(self):
        return ({
            '0_x': self.x,
            '0_y': self.y,
            '0_angle_deg_cw': self.angle_deg,
            '0_major': self.major,
            '0_minor': self.minor,
        })

    @property
    def xy(self):
        return np.array((self.x, self.y))

    @xy.setter
    def xy(self, xy):
        self.x = xy[0]
        self.y = xy[1]

    @property
    def area(self):
        return cv2.contourArea(self.to_poly())

    def to_poly(self):
        return cv2.ellipse2Poly((int(self.x), int(self.y)), (int(old_div(self.major, 2.)), int(old_div(self.minor, 2.))),
                                int(self.angle_deg), 0, 360, 30)

    def get_overlap(self, el_region):
        if isinstance(el_region, Ellipse):
            el_region = el_region.to_poly()
        area, poly = cv2.intersectConvexConvex(self.to_poly(), el_region)
        #         poly = poly.reshape((-1, 2))
        return area

    def to_array(self):
        return np.array([self.x, self.y, self.angle_deg, self.major, self.minor, self.frame])

    def rotate(self, angle_deg_cw, rotation_center_xy=None):
        if rotation_center_xy is None:
            rotation_center_xy = self.xy
        self.angle_deg += angle_deg_cw
        rot = cv2.getRotationMatrix2D(tuple(rotation_center_xy), -angle_deg_cw, 1.)
        self.xy = p2e(np.vstack((rot, (0, 0, 1))).dot(e2p(column(self.xy)))).flatten()
        return self

    def move(self, delta_xy):
        self.xy += delta_xy
        return self

    def get_point(self, angle_deg):
        """
        Get point on the ellipse.

        The position is approximate.

        :param angle_deg: angle between the point and the major axis
        :return: xy; ndarray, shape=(2, )
        """
        assert self.minor > 2
        assert self.major > 2, ''
        pts = cv2.ellipse2Poly(tuple(self.xy.astype(int)), (int(old_div(self.major, 2.)), int(old_div(self.minor, 2.))),
                               int(self.angle_deg), int(round(angle_deg)) - 2, int(round(angle_deg)) + 2, 1)
        return pts.mean(axis=0)

    def get_vertices(self):
        """
        Get ellipse vertices: the endpoints of the major axis.

        :return: positive endpoint xy, negative endpoint xy
        """
        # returns head, tail

        p_ = np.array([self.major / 2. * np.cos(np.deg2rad(self.angle_deg)),
                       self.major / 2. * np.sin(np.deg2rad(self.angle_deg))])
        endpoint_pos = self.xy + p_
        endpoint_neg = self.xy - p_
        # endpoint_pos = np.ceil(self.xy + p_) + np.array([1, 1])
        # endpoint_neg = np.ceil(self.xy - p_) - np.array([1, 1])

        return endpoint_pos, endpoint_neg

    def draw(self, ax=None, label=None, color=None):
        import matplotlib.pylab as plt
        from matplotlib.patches import Ellipse
        if ax is None:
            ax = plt.gca()
        if color is None:
            color = 'r'
        ax.add_patch(Ellipse(self.xy, self.major, self.minor, self.angle_deg,
                             facecolor='none', edgecolor=color,
                             label=label, linewidth=1))
        plt.scatter(self.x, self.y, c=color)

    def __add__(self, other):
        """
        Return mean ellipse.

        :param other: Ellipse
        :return: mean Ellipse
        """
        assert self.frame == other.frame
        mean = np.vstack((self.to_array(), other.to_array())).mean(axis=0)
        el = Ellipse(*mean)
        el.frame = int(el.frame)
        return el