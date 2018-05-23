import numpy as np
import math
import cv2


def p2e(projective):
    """
    Convert 2d or 3d projective to euclidean coordinates.

    :param projective: projective coordinate(s)
    :type projective: numpy.ndarray, shape=(3 or 4, n)

    :return: euclidean coordinate(s)
    :rtype: numpy.ndarray, shape=(2 or 3, n)
    """
    assert type(projective) == np.ndarray
    assert projective.ndim == 1 or (projective.ndim == 2 and (projective.shape[0] == 4) or (projective.shape[0] == 3))
    if projective.ndim == 1:
        return (projective / projective[-1])[0:-1]
    else:
        return (projective / projective[-1, :])[0:-1, :]


def e2p(euclidean):
    """
    Convert 2d or 3d euclidean to projective coordinates.

    :param euclidean: projective coordinate(s)
    :type euclidean: numpy.ndarray, shape=(2 or 3, n)

    :return: projective coordinate(s)
    :rtype: numpy.ndarray, shape=(3 or 4, n)
    """
    assert type(euclidean) == np.ndarray
    assert euclidean.ndim == 1 or (euclidean.ndim == 2 and (euclidean.shape[0] == 3 or euclidean.shape[0] == 2))
    if euclidean.ndim == 1:
        return np.append(euclidean, 1)
    else:
        return np.vstack((euclidean, np.ones((1, euclidean.shape[1]))))


class TransformableRegion:
    def __init__(self, image):
        self.img = image
        #
        # self.border_px = 0
        self.use_background = True
        self.transformation = np.eye(3)
        self.mask = None
        self.region = None

    def compose(self, other):
        assert self.img.shape == other.img.shape
        assert self.use_background ^ other.use_background
        if self.use_background:
            img = self.get_img().copy()
            masked_img = other.get_img()
            mask = other.get_mask()
        else:
            img = other.get_img().copy()
            masked_img = self.get_img()
            mask = self.get_mask()
        img[mask] = masked_img[mask]
        return img

    def set_region(self, region):
        self.region = region

    def set_mask(self, mask):
        self.mask = mask

    def set_img(self, img):
        self.img = img

    def get_mask(self):
        assert np.all(self.transformation[2, :] == (0, 0, 1))
        assert self.mask is not None
        return cv2.warpAffine(self.mask, self.transformation[:2], self.mask.shape[::-1]).astype(np.bool)

    def set_region_points_mask(self, n_dilations=0):
        assert self.img is not None
        assert self.region is not None
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        pts = self.region.pts()
        x = y = 0
        self.mask[pts[:, 0] - y + 1, pts[:, 1] - x + 1] = 1
        if n_dilations != 0:
            self.mask = cv2.dilate(self.mask, kernel=np.ones((3, 3), np.uint8), iterations=n_dilations)

    def set_elliptic_mask(self, major_multiplier=4, minor_multiplier=6):
        assert self.img is not None
        assert self.region is not None
        self.mask = np.zeros(shape=self.img.shape[:2], dtype=np.uint8)
        cv2.ellipse(self.mask, tuple(self.region.centroid_[::-1].astype(int)),
                    (int(major_multiplier * self.region.major_axis_),
                     int(minor_multiplier * self.region.minor_axis_)),
                    -int(math.degrees(self.region.theta_)), 0, 360, 255, -1)

    def set_border(self, border_px):
        self.border_px = border_px
        return self

    def rotate(self, ccw_angle_deg, rotation_center_yx=None):
        if rotation_center_yx is None:
            rotation_center_yx = np.array((0, 0))

        rot = cv2.getRotationMatrix2D(tuple(rotation_center_yx[::-1]),
                                      ccw_angle_deg, 1.)
        self.transformation = np.vstack((rot, (0, 0, 1))).dot(self.transformation)
        return self

    def scale(self, factor, center_yx=None):
        if center_yx is None:
            center_yx = self.region.centroid()
        trans = cv2.getRotationMatrix2D(center_yx[::-1], 0., factor)
        self.transformation = np.vstack((trans, (0, 0, 1))).dot(self.transformation)
        return self

    def move(self, delta_yx):
        move_trans = np.array([[1., 0., delta_yx[1]],
                               [0., 1., delta_yx[0]],
                               [0., 0., 1.]])
        self.transformation = move_trans.dot(self.transformation)
        return self

    def get_transformed_coords(self, coords_xy):
        assert coords_xy.shape == (2, ) or (coords_xy.ndim == 2 and coords_xy.shape[0] == 2)  # (2, ) or (2, n)
        return p2e(self.transformation.dot(e2p(coords_xy)))

    def get_transformed_angle(self, ccw_angle_deg):
        #TODO
        angle1 = (ccw_angle_deg - math.degrees(math.atan(self.transformation[1, 0] / self.transformation[0, 0]))) % 360
        angle2 = (ccw_angle_deg - math.degrees(math.atan2(self.transformation[1, 0], self.transformation[0, 0]))) % 360
        assert round(angle1, 2) == round(angle2, 2)
        return angle1

    def get_img(self):
        assert np.all(self.transformation[2, :] == (0, 0, 1))
        return cv2.warpAffine(self.img, self.transformation[:2], self.img.shape[:2][::-1])

