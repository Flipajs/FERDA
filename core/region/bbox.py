import cv2
import numpy as np

from core.region.region import Region
from core.region.shape import Shape
from core.region.ep import p2e, e2p, column
from utils.angles import angle_absolute_error_direction_agnostic, angle_absolute_error


class BBox(Shape):
    @classmethod
    def from_region(cls, region):
        yx = region.centroid()
        tmp = cls(yx[1], yx[0], -np.rad2deg(region.theta_), 2 * region.major_axis_, 2 * region.minor_axis_,
                  region.frame())
        return tmp

    @classmethod
    def from_planar_object(cls, another_object):
        xmin, ymin, width, height = cv2.boundingRect(another_object.to_poly())
        xmax = xmin + width
        ymax = ymin + height
        return cls(xmin, ymin, xmax, ymax)

    @classmethod
    def from_dict(cls, region_dict, frame=None):
        return cls(region_dict['0_x'], region_dict['0_y'], region_dict['0_angle_deg_cw'], region_dict['0_major'],
                   region_dict['0_minor'], frame)

    def __init__(self, xmin=None, ymin=None, xmax=None, ymax=None, frame=None):
        super(BBox, self).__init__(frame)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __str__(self):
        return('BBox xymin ({xmin:.1f},{ymin:.1f}) xymax ({xmax:.1f},{ymax:.1f}), '\
               'width height ({width:.1f},{height:.1f}), frame {frame}'.format(
            width=self.width, height=self.height, **self.__dict__))

    @property
    def xy(self):
        return np.array((self.xmin + self.width / 2, self.ymin + self.height / 2))

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    def to_poly(self):
        return [(self.xmin, self.ymin), (self.xmin, self.ymax), (self.xmax, self.ymax), (self.xmax, self.ymin)]

    def is_outside_bounds(self, xmin, ymin, xmax, ymax):
        return self.xmin < xmin or self.ymin < ymin or self.xmax > xmax or self.ymax > ymax

    def to_array(self):
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax, self.frame])

    def rotate(self, angle_deg_cw, rotation_center_xy=None):
        assert False
        if rotation_center_xy is None:
            rotation_center_xy = self.xy
        self.angle_deg += angle_deg_cw
        rot = cv2.getRotationMatrix2D(tuple(rotation_center_xy), -angle_deg_cw, 1.)
        self.xy = p2e(np.vstack((rot, (0, 0, 1))).dot(e2p(column(self.xy)))).flatten()
        return self

    def move(self, delta_xy):
        self.xmin += delta_xy[0]
        self.xmax += delta_xy[0]
        self.ymin += delta_xy[1]
        self.ymax += delta_xy[1]
        return self

    def draw(self, ax=None, label=None, color=None):
        import matplotlib.pylab as plt
        from matplotlib.patches import Rectangle
        if ax is None:
            ax = plt.gca()
        if color is None:
            color = 'r'
        ax.add_patch(Rectangle((self.xmin, self.ymin), self.width, self.height,
                             facecolor='none', edgecolor=color,
                             label=label, linewidth=1))

