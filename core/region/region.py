__author__ = 'fnaiser'

import numpy as np
import math
from utils.img import get_roi

class Region():
    """ This class encapsulates set of points. It computes and stores statistics like moments, contour of region etc.
        It takes list of list in format [[y1, x1], [y2, x2]... ]
        Or dict as an output from mser algorithm where the point representation is saved as 'rle' in Run Lenght encoding.

    """

    def __init__(self, data=None, frame=-1, id=-1):
        self.pts_ = None
        self.centroid_ = np.array([-1, -1])
        self.label_ = -1
        self.margin_ = -1
        self.min_intensity_ = -1
        self.max_intensity_ = -1

        self.sxx_ = -1
        self.syy_ = -1
        self.sxy_ = -1

        self.major_axis_ = -1
        self.minor_axis_ = -1

        self.a_ = -1
        self.b_ = -1

        # in radians
        self.theta_ = -1

        self.parent_label_ = -1

        if isinstance(data, dict):
            self.from_dict_(data)
        else:
            self.from_pts_(data)

        self.roi_ = None
        self.frame_ = frame
        self.id_ = id
        self.contour_ = None
        self.is_virtual = False

    def from_dict_(self, data):
        pts = []
        if 'rle' in data:
            for row in data['rle']:
                pts.extend([[row['line'], c] for c in range(row['col1'], row['col2'] + 1)])
        else:
            raise Exception('wrong data format',
                            'Wrong data format in from_dict_ in region.points.py. Expected dictionary with "rle" key.')

        self.centroid_ = np.array([data['cy'], data['cx']])
        self.pts_ = np.array(pts)
        self.label_ = data['label']
        self.margin_ = data['margin']
        self.min_intensity_ = data['minI']
        self.max_intensity_ = data['maxI']

        # image moments
        self.sxx_ = data['sxx']
        self.syy_ = data['syy']
        self.sxy_ = data['sxy']

        self.major_axis_, self.minor_axis_ = compute_region_axis_(self.sxx_, self.syy_, self.sxy_)

        ########## stretching the axes
        a = self.major_axis_
        b = self.minor_axis_

        axis_ratio = a / float(b)
        self.b_ = math.sqrt(len(self.pts_) / (axis_ratio * math.pi))
        self.a_ = self.b_ * axis_ratio
        ########

        self.theta_ = get_orientation(self.sxx_, self.syy_, self.sxy_)
        self.parent_label_ = data['parent_label']

    def from_pts_(self, data):
        self.pts_ = np.array(data)

    def area(self):
        return len(self.pts_)

    def label(self):
        return self.label_

    def margin(self):
        return self.margin_

    def pts(self):
        return self.pts_

    def pts_copy(self):
        return np.copy(self.pts_)

    def centroid(self):
        return np.copy(self.centroid_)

    def set_centroid(self, centroid):
        self.centroid_ = centroid

    def roi(self):
        if not hasattr(self, 'roi_') or not self.roi_:
            self.roi_ = get_roi(self.pts())

        return self.roi_

    def contour(self):
        from utils.drawing.points import get_contour
        if not hasattr(self, 'contour_') or self.contour_ is None:
            self.contour_ = get_contour(self.pts_)

        return self.contour_


def compute_region_axis_(sxx, syy, sxy):
    la = (sxx + syy) / 2
    lb = math.sqrt(4 * sxy * sxy + (sxx - syy) * (sxx - syy)) / 2

    lambda1 = math.sqrt(la+lb)
    lambda2 = math.sqrt(la-lb)

    return lambda1, lambda2


def get_orientation(sxx, syy, sxy):
    theta = 0.5*math.atan2(2*sxy, (sxx - syy))

    #it must be reversed around X because in image is top left corner [0, 0] and it is not very intuitive
    theta = -theta
    if theta < 0:
        theta += math.pi

    return theta



if __name__ == '__main__':
    print "test"