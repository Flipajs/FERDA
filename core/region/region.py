__author__ = 'fnaiser'

import numpy as np


class Region():
    """ This class encapsulates set of points. It computes and stores statistics like moments, contour of region etc.
        It takes list of list in format [[y1, x1], [y2, x2]... ]
        Or dict as an output from mser algorithm where the point representation is saved as 'rle' in Run Lenght encoding.

    """

    def __init__(self, data=None):
        self.pts_ = None
        self.centroid_ = np.array([-1, -1])
        self.label_ = -1
        self.margin_ = -1

        if isinstance(data, dict):
            self.from_dict_(data)
        else:
            self.from_pts_(data)

    def from_dict_(self, data):
        pts = []
        if 'rle' in data:
            for row in data['rle']:
                pts.extend([[row['line'], c] for c in range(row['col1'], row['col2'] + 1)])
        else:
            raise Exception('wrong data format',
                            'Wrong data format in from_dict_ in region.region.py. Expected dictionary with "rle" key.')

        self.centroid_ = np.array([data['cy'], data['cx']])
        self.pts_ = np.array(pts)
        self.label_ = data['label']
        self.margin_ = data['margin']

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



if __name__ == '__main__':
    print "test"