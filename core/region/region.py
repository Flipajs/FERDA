import math
import numbers
from utils.roi import get_roi, get_roi_rle, ROI
from utils.img import createLineIterator
import numpy as np


class Region(object):
    """ This class encapsulates set of points. It computes and stores statistics like moments, contour of region etc.
        It takes list of list in format [[y1, x1], [y2, x2]... ]
        Or dict as an output from mser algorithm where the point representation is saved as 'rle' in Run Length encoding.

    """

    def __init__(self, data=None, frame=-1, id=-1, is_origin_interaction=False):
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

        # TODO: a_, b_ should be deprecated and reduced by major/minor_axis
        self.a_ = -1
        self.b_ = -1

        # radians, 0 rad to right/west,
        # positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner),
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

        # TODO: refactor + method...
        # TODO: deprecated
        self.is_origin_interaction_ = is_origin_interaction

    def __str__(self):
        return '{}\nid: {}\nframe: {}\narea: {}\ncentroid: [{:.2f}, {:.2f}]\n' \
               'major axis: {:.2f}\nminor axis: {:.2f}\nmargin: {}'.format(
            repr(self), self.id_, self.frame_, self.area(), self.centroid_[0], self.centroid_[1],
            self.major_axis_, self.minor_axis_, self.margin_)

    def __hash__(self):
        return hash(self.id_)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self, flatten=False):
        state = self.__dict__.copy()
        if 'pts_rle_' in state:
            del state['pts_rle_']
        if 'pts_' in state:
            del state['pts_']
        if flatten:
            state['centroid_x'] = self.centroid_[1]
            state['centroid_y'] = self.centroid_[0]
            del state['centroid_']
        else:
            state['centroid_'] = state['centroid_'].tolist()
        state['min_intensity_'] = float(state['min_intensity_'])
        state['max_intensity_'] = float(state['max_intensity_'])
        state['intensity_percentile'] = float(state['intensity_percentile'])
        if flatten and self.roi_ is not None:
            state.update({'roi_' + k: v for k, v in self.roi().as_dict().items()})
            del state['roi_']
        return state

    def __setstate__(self, state):
        if 'centroid_' in state:
            state['centroid_'] = np.array(state['centroid_'])
        else:
            state['centroid_'] = np.array([state['centroid_y'], state['centroid_x']])
            del state['centroid_y']
            del state['centroid_x']
        if 'roi_x' in state:
            state['roi_'] = ROI(state['roi_y'], state['roi_x'], state['roi_height'], state['roi_width'])
            del state['roi_y']
            del state['roi_x']
            del state['roi_height']
            del state['roi_width']
            for k, v in state.items():
                if isinstance(v, (numbers.Number, np.number)) and np.isnan(v):
                    state[k] = None
        self.__dict__.update(state)

    @property
    def angle_deg_ccw(self):
        """
        Return region main axis angle.

        Angle orientation on image axes (origin on top left):
        positive values mean counter-clockwise rotation.

        :return: angle in degrees
        """
        return np.rad2deg(self.theta_)

    @property
    def angle_deg_cw(self):
        """
        Return region main axis angle.

        Angle orientation on image axes (origin on top left):
        positive values mean clockwise rotation.

        :return: angle in degrees
        """
        return -np.rad2deg(self.theta_)

    def is_origin_interaction(self):
        try:
            return self.is_origin_interaction_
        except AttributeError:
            return False

    def id(self):
        return self.id_

    def pts_from_rle_(self, data):
        i = 0

        # do pts allocation if possible
        # for some reason, in rare circumstances area from MSER is wrong
        try:
            pts = np.zeros((self.area(), 2), dtype=np.int)

            for row in data:
                d = row['col2'] + 1 - row['col1']

                pts[i:i+d, 0] = row['line']
                pts[i:i+d, 1] = range(row['col1'], row['col2'] + 1)
                i += d

                # for c in xrange(row['col1'], row['col2'] + 1):
                    # pts[i, 0] = row['line']
                    # pts[i, 1] = c
                    # i += 1
        except (IndexError, AttributeError) as e:
            print e
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
            # self.pts_ = self.pts_from_rle_(self.pts_rle_)
        elif 'pts' in data:
            self.pts_ = data['pts']
        else:
            raise Exception('wrong data format',
                            'Wrong data format in from_dict_ in region.points.py. Expected dictionary with "rle" or "pts" keys.')

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
        self.minor_axis_ = math.sqrt(self.area_ / (axis_ratio * math.pi))
        self.major_axis_ = self.minor_axis_ * axis_ratio
        ########

        self.theta_ = get_orientation(self.sxx_, self.syy_, self.sxy_)
        self.parent_label_ = data['parent_label']

    def from_pts_(self, data):
        self.pts_ = np.array(data)

    def area(self):
        if not self.area_:
            self.area_ = len(self.pts())

        return self.area_

    def ellipse_major_axis_length(self):
        """

        Returns: major axis length [px] for an ellipse approximation of the region

        """
        return self.major_axis_

    def ellipse_minor_axis_length(self):
        """

        Returns: minor axis length [px] for an ellipse approximation of the region

        """
        return self.minor_axis_

    def label(self):
        return self.label_

    def margin(self):
        return self.margin_

    def pts(self):
        """
        Return region points (contour + area).

        :return: yx coordinates; array, shape=(n, 2)
        """
        if self.pts_ is None:
            assert self.pts_rle_ is not None
            self.pts_ = self.pts_from_rle_(self.pts_rle_)

        return self.pts_

    def draw_mask(self, img):
        yx = self.pts()
        if issubclass(img.dtype.type, bool) or issubclass(img.dtype.type, np.bool_):
            img[yx[:, 0], yx[:, 1]] = True
        elif issubclass(img.dtype.type, numbers.Integral):
            img[yx[:, 0], yx[:, 1]] = 255
        elif issubclass(img.dtype.type, numbers.Real):
            img[yx[:, 0], yx[:, 1]] = 1.
        else:
            assert False, 'Not supported dtype.'
        return img

    def pts_copy(self):
        return np.copy(self.pts())

    def centroid(self):
        "in order y, x"
        return np.copy(self.centroid_)

    def set_centroid(self, centroid):
        self.centroid_ = centroid

    def roi(self):
        if not hasattr(self, 'roi_') or not self.roi_:
            if self.pts_ is None:
                self.roi_ = get_roi_rle(self.pts_rle_)
            else:
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
        return 1- (self.minor_axis_ / self.major_axis_) ** 0.5


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

        # to deal with head orientation uncertainty
        phi = min(phi, np.pi - phi)
        return phi

    def is_inside(self, pt, tolerance=0):
        tolerance = int(tolerance)
        from utils.drawing.points import draw_points_crop_binary

        try:
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
        except:
            import warnings
            warnings.warn("region id: {}".format(self.id_))

            pass

        return False

    def get_border_point(self, angle_rad, starting_point_yx=None, shift_px=0):
        '''

        :param angle_rad:
        :param starting_point_yx:
        :param shift_px: move border point along the line from the starting point, positive means outside of region
        :return:
        '''
        if starting_point_yx is None:
            starting_point_yx = self.centroid_[::-1]
        point_theta_xy = starting_point_yx[::-1] + 4 * self.major_axis_ * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        point_theta_xy = np.round(point_theta_xy).astype(int)

        mask = np.zeros(self.pts().max(axis=0) + 2, dtype=np.uint8)  # we need at least 1 background pixel around
                                                                     # actual shape
        # mask[self.pts()[:, 0] + 1, self.pts()[:, 1] + 1] = 1  # why was there + 1 ?
        mask[self.pts()[:, 0], self.pts()[:, 1]] = 1

        # import matplotlib.pylab as plt
        # plt.imshow(mask)
        # plt.plot(point_theta_xy[0], point_theta_xy[1], '+')
        # plt.annotate('center', starting_point_yx, (10, 0), textcoords='offset pixels')
        # plt.plot(starting_point_yx[0], starting_point_yx[1], '+')

        # find touch point on the ant border
        line = createLineIterator(np.round(starting_point_yx).astype(int), point_theta_xy, mask)
        i = np.nonzero(line[:, 2] == 0)[0][0]
        index = np.clip(i + shift_px, 0, len(line) - 1)
        # if index != i + shift_px:
        #     print('get_border_point shift_px clipped')
        border_point_xy = line[index, 0:2]

        # plt.plot(border_point_xy[0], border_point_xy[1], '+')

        return border_point_xy

    def ellipse_area_ratio(self):
        return (4*self.major_axis_ * self.minor_axis_) / float(self.area())

    def get_img(self, image_manager, **kwargs):
        return image_manager.get_crop(self.frame(), self.roi(), **kwargs)

import h5py

class RegionH5(Region):
    region_pts_h5 = None
    region_pts = None
    n_items = 0

    @classmethod
    def init_h5_store(cls, filename, num_items=None):
        cls.region_pts_h5 = h5py.File(filename, mode='r')  # TODO 'a'
        if not 'region_pts' in cls.region_pts_h5:
            assert num_items is not None
            dt = h5py.special_dtype(vlen=np.dtype('int32'))
            cls.region_pts_h5.create_dataset('region_pts', (num_items,), dtype=dt, maxshape=(None,))  # , compression='gzip'
        cls.region_pts = cls.region_pts_h5['region_pts']

    @classmethod
    def create_h5_store(cls, filename, num_items):
        cls.region_pts_h5 = h5py.File(filename, mode='w')
        dt = h5py.special_dtype(vlen=np.dtype('int32'))
        cls.region_pts_h5.create_dataset('region_pts', (num_items,), dtype=dt, maxshape=(None,))  # , compression='gzip'
        cls.region_pts = cls.region_pts_h5['region_pts']

    @classmethod
    def from_region(cls, region):
        regionH5 = cls()
        regionH5.__dict__.update(region.__dict__)
        if regionH5.region_pts is not None:
            try:
                regionH5.set_pts(region.pts())
                regionH5.pts_ = None
                regionH5.pts_rle_ = None
            except:
                pass
        return regionH5

    def pts(self):
        if self.region_pts is not None:
            return self.region_pts[self.id()].reshape((-1, 2))
        else:
            return super(RegionH5, self).pts()

    def set_pts(self, value):
        if self.region_pts is not None:
            self.region_pts[self.id()] = value.flatten()
        else:
            self.pts_ = value


def encode_RLE(pts, return_area=True):
    """
    returns list of dictionaries
    {'col1': , 'col2':, 'line':}
    """
    import time
    t = time.time()
    roi = get_roi(pts)
    result = np.zeros((roi.height(), roi.width()), dtype="uint8")

    pts2 = pts - roi.top_left_corner()
    result[pts2[:, 0], pts2[:, 1]] = 1

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
                area += run['col2'] - run['col1'] + 1
                rle.append(run)
                run = {}
                running = False

        if running:
            run['col2'] = j - 1 + offset_x
            area += run['col2'] - run['col1'] + 1
            rle.append(run)

    """
    print time.time() - t

    for run in rle:
        print "line %s: [%s - %s]" % (run["line"], run["col1"], run["col2"])
    """

    if return_area:
        return rle, area

    return rle

def get_region_endpoints(r):
    # returns head, tail

    p_ = np.array([r.major_axis_ * math.sin(-r.theta_), r.major_axis_ * math.cos(-r.theta_)])
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
