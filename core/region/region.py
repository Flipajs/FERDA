__author__ = 'fnaiser'

from utils.video_manager import get_auto_video_manager
import numpy as np
import math
from utils.roi import get_roi


class Region():
    """ This class encapsulates set of points. It computes and stores statistics like moments, contour of region etc.
        It takes list of list in format [[y1, x1], [y2, x2]... ]
        Or dict as an output from mser algorithm where the point representation is saved as 'rle' in Run Lenght encoding.

    """

    def __init__(self, data=None, frame=-1, id=-1):
        self.id_ = id
        self.pts_ = None
        self.pts_rle_ = None
        self.centroid_ = np.array([-1, -1])
        self.label_ = -1
        self.margin_ = -1
        self.min_intensity_ = -1
        self.intensity_percentile = -1
        self.max_intensity_ = -1
        self.area_ = None

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


    def __str__(self):
        s = repr(self)+" t: "+str(self.frame_)+" area: "+str(self.area())+" centroid: ["+str(round(self.centroid_[0], 2))+", "+\
            str(round(self.centroid_[1], 2))+"] a: {:.3}".format(self.a_) + " b: {:.3}".format(self.b_)+" sxx: {:.4}".format(self.sxx_) +\
            " syy: {:.4} ".format(self.syy_) + " sxy: {:.4}".format(self.sxy_) + " margin: " + str(self.margin_)

        return s

    def id(self):
        return self.id_

    def pts_from_rle_(self, data):
        i = 0

        # do pts allocation if possible
        # if self.area_:
        try:
            pts = np.zeros((self.area(), 2), dtype=np.int)

            for row in data:
                for c in xrange(row['col1'], row['col2'] + 1):
                    pts[i, 0] = row['line']
                    pts[i, 1] = c

                    i += 1
        except:
            pts = []

            for row in data:
                for c in xrange(row['col1'], row['col2'] + 1):
                    pts.append([row['line'], c])
                    i += 1

            pts = np.array(pts)

        return pts

    def from_dict_(self, data):
        self.area_ = data['area']
        if 'rle' in data:
            self.pts_rle_ = data['rle']
            self.pts_ = self.pts_from_rle_(self.pts_rle_)
        else:
            raise Exception('wrong data format',
                            'Wrong data format in from_dict_ in region.points.py. Expected dictionary with "rle" key.')

        self.centroid_ = np.array([data['cy'], data['cx']])
        # self.pts_ = np.array(pts)
        self.label_ = data['label']
        self.margin_ = data['margin']
        self.min_intensity_ = data['minI']
        if 'intensity_percentile' in data:
            self.intensity_percentile = data['intensity_percentile']

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
        if not self.area_:
            self.area_ = len(self.pts())

        return self.area_

    def label(self):
        return self.label_

    def margin(self):
        return self.margin_

    def pts(self):
        if self.pts_ is None:
            self.pts_ = self.pts_from_rle_(self.pts_rle_)

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
            self.contour_ = get_contour(self.pts())

        return self.contour_

    def contour_without_holes(self):
        from utils.drawing.points import get_contour_without_holes

        return get_contour_without_holes(self.pts())

    def frame(self):
        return self.frame_

    def vector_on_major_axis_projection_head_unknown(self, region):
        """
        projection of movement vector onto main axis vector in range <0, pi/2> due to the head orientation uncertainty
        Returns:

        """

        p1, _ = get_region_endpoints(self)

        u = p1 - self.centroid()
        v = self.centroid() - region.centroid()
        u_d = np.linalg.norm(u)
        if u_d == 0:
            return 0

        a = np.dot(v, u/u_d)

        return abs(a)

        # c = np.dot(u, v) / np.norm(u) / np.norm(v)  # -> cosine of the angle
        # # <0, pi>
        # angle = np.arccos(np.clip(c, -1, 1))  # if you really want the angle
        #
        # if angle > np.pi/2:
        #     angle -= np.pi/2

    def is_ignorable(self, r2, max_dist):
        """


        Args:
            r2:
            max_dist:

        Returns:

        """
        # # term1 for speedup...
        # term1 = np.linalg.norm(self.centroid() - r2.centroid()) < max_dist
        # b = not term1 and not self.roi().is_intersecting_expanded(r2.roi(), max_dist)

        # term1 = np.linalg.norm(self.centroid() - r2.centroid()) < max_dist
        # b = not term1 and not self.roi().is_intersecting_expanded(r2.roi(), max_dist)
        return not self.roi().is_intersecting_expanded(r2.roi(), max_dist)

    def eccentricity(self):
        return 1-(self.b_ / self.a_)**0.5


    def get_phi(self, r2):
        """
        angle between movement vector and major axis <0, pi>
        Args:
            r2:

        Returns:

        """

        u = self.centroid() - r2.centroid()
        u_ = np.linalg.norm(u)
        p1, _ = get_region_endpoints(self)
        v = self.centroid() - p1
        v_ = np.linalg.norm(v)

        if u_ < 2.0 or v_ < 2.0:
            return 0

        c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)  # -> cosine of the angle
        # <0, pi>
        phi = np.arccos(np.clip(c, -1, 1))  # clip to prevent numerical imprecision

        return phi

    def is_inside(self, pt, tolerance=0):
        tolerance = int(tolerance)
        from utils.drawing.points import draw_points_crop_binary
        if self.roi().is_inside(pt, tolerance=tolerance):
            pt_ = np.asarray(np.round(pt - self.roi().top_left_corner()), dtype=np.uint)
            # TODO + tolerance margin, and fix offset
            im = draw_points_crop_binary(self.pts())

            y1 = int(max(0, pt_[0] - tolerance))
            y2 = int(min(pt_[0] + tolerance + 1, im.shape[0]))
            x1 = int(max(0, pt_[1] - tolerance))
            x2 = int(min(pt_[1] + tolerance + 1, im.shape[1]))
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if im[y, x]:
                        return True

        return False


def encode_RLE(pts):
    """
    returns list of dictionaries
    {'col1': , 'col2':, 'line':}
    """
    import time
    t = time.time()
    roi = get_roi(pts)
    result = np.zeros((roi.height(), roi.width()), dtype="uint8")

    pts2 = pts - roi.top_left_corner()
    result[pts2[:,0], pts2[:,1]] = 1

    offset_x = roi.top_left_corner()[1]
    offset_y = roi.top_left_corner()[0]

    rle = []
    area = 0
    for i in range(0, result.shape[0]):
        run = {}
        running = False
        for j in range(0, result.shape[1]):
            if result[i][j] == 1 and not running:
                run['col1'] = j + offset_x
                run['line'] = i + offset_y
                running = True
            if result[i][j] == 0 and running:
                run['col2'] = j - 1 + offset_x
                rle.append(run)
                run = {}
                running = False

            area += 1
        if running:
            run['col2'] = j - 1 + offset_x
            rle.append(run)

    """
    print time.time() - t

    for run in rle:
        print "line %s: [%s - %s]" % (run["line"], run["col1"], run["col2"])
    """
    return rle

def get_region_endpoints(r):
    # returns head, tail

    p_ = np.array([r.a_ * math.sin(-r.theta_), r.a_ * math.cos(-r.theta_)])
    endpoint1 = np.ceil(r.centroid() + p_) + np.array([1, 1])
    endpoint2 = np.ceil(r.centroid() - p_) - np.array([1, 1])

    return endpoint1, endpoint2

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
    import cPickle as pickle
    import numpy as np
    f = open('/home/dita/PycharmProjects/c5regions.pkl', 'r+b')
    up = pickle.Unpickler(f)
    regions = up.load()
    f.close()

    r = regions[0]
    pts = r.pts_copy()
    encode_RLE(pts)
