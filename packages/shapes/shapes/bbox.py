import cv2
import numpy as np
import copy

from shapes.shape import Shape
from shapes.ep import p2e, e2p, column


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
        d = region_dict
        if 'x' in d and 'y' in d and 'width' in d and 'height' in d:
            return cls(d['x'], d['y'], d['x'] + d['width'], d['y'] + d['height'], frame)

    @classmethod
    def from_xywh(cls, x, y, width, height, frame=None):
        return cls(x, y, x + width, y + height, frame)

    @classmethod
    def from_xycenter_wh(cls, x_center, y_center, width, height, frame=None):
        return cls(x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2, frame)

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

    def is_strictly_outside_bounds(self, xmin, ymin, xmax, ymax):
        return self.iou(BBox(xmin, ymin, xmax, ymax)) == 0

    def is_strictly_outside_bbox(self, bbox):
        return self.is_strictly_outside_bounds(*bbox.to_array()[:4])

    def is_partially_outside_bounds(self, xmin, ymin, xmax, ymax):
        return self.iou(BBox(xmin, ymin, xmax, ymax)) > 0 and not self.is_inside_bounds(xmin, ymin, xmax, ymax)

    def is_partially_outside_bbox(self, bbox):
        return self.is_partially_outside_bounds(*bbox.to_array()[:4])

    def is_inside_bounds(self, xmin, ymin, xmax, ymax):
        return self.xmin > xmin and self.ymin > ymin and self.xmax < xmax and self.ymax < ymax

    def is_inside_bbox(self, bbox):
        return self.is_inside_bounds(*bbox.to_array()[:4])

    def cut(self, viewport_bbox):
        if self.is_strictly_outside_bbox(viewport_bbox):
            return None
        elif self.is_inside_bbox(viewport_bbox):
            return self
        else:
            assert self.is_partially_outside_bbox(viewport_bbox)
            return self.intersection(viewport_bbox)

    def intersection(self, other):
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        if ymin >= ymax or xmin >= xmax:
            return None
        else:
            assert self.frame == other.frame
            return BBox(xmin, ymin, xmax, ymax, self.frame)

    def to_array(self):
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax, self.frame])

    @property
    def area(self):
        return self.width * self.height

    def iou(self, bbox):
        # source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

        # determine the (x, y)-coordinates of the intersection rectangle
        intersection = self.intersection(bbox)

        if intersection is None:
            return 0

        # compute the area of intersection rectangle
        # interArea = max(0, inter_xmax - inter_xmin + 1) * max(0, inter_ymax - inter_ymin + 1)
        # interArea = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        interArea = intersection.area

        # compute the area of both the prediction and ground-truth
        # rectangles
        # boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        # boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        return interArea / float(self.area + bbox.area - interArea)

    def __sub__(self, other):
        return np.linalg.norm(self.xy - other.xy)

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
        if label is not None:
            plt.annotate(label, self.xy) # , xytext=(0, -self.height / 2), textcoords='offset pixels')

    def draw_to_image(self, img, label=None, color=None):
        if color is None:
            color = (0, 0, 255)
        round_tuple = lambda x: tuple([int(round(num)) for num in x])
        cv2.rectangle(img, round_tuple((self.xmin, self.ymin)),
                      round_tuple((self.xmax, self.ymax)), color)
        if label is not None:
            font_size = 1
            font_thickness = 1
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            text_size, _ = cv2.getTextSize(label, font_face, font_size, font_thickness)
            cv2.putText(img, label, round_tuple((self.xy[0] - (text_size[0] / 2), self.ymin - text_size[1])),
                        font_face, font_size, (255, 255, 255), font_thickness)
