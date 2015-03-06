__author__ = 'fnaiser'

import pickle

import numpy as np
from scipy import ndimage
import cv2

from utils.geometry import check_roi
from core.region.region import Region


class DistanceMap():
    """
    This class encapsulates distance map image, which is used to speed up search for nearest point for given coordinates.
    """

    def __init__(self, pts):
        [self.y_min, self.x_min] = np.min(pts, axis=0)
        [self.y_max, self.x_max] = np.max(pts, axis=0)

        self.contour_img_ = np.ones((self.y_max - self.y_min + 1, self.x_max - self.x_min + 1), dtype=np.bool)
        self.contour_img_[pts[:, 0] - self.y_min, pts[:, 1] - self.x_min] = False

        self.d_map, self.d_map_labels = ndimage.distance_transform_edt(self.contour_img_, return_indices=True)

    def get_nearest_point(self, pt):
        """
        Returns distance and nearest point coordinates on object measured from given pt.
        In case when the pt is out of ROI of distance image, the distance is inaccurate. But this still holds: estimated_dist >= real_dist
        :param pt:
        :return: distance, [y, x]
        """

        offset = np.array([self.y_min, self.x_min])
        pt = np.array(pt)
        pt_ = check_roi(pt, self.y_min, self.x_min, self.y_max, self.x_max)
        y_, x_ = pt_ - offset
        nearest_pt = self.d_map_labels[:, y_, x_]
        dist = self.d_map[y_, x_]

        # If the point is outside ROI - the distance to distance map border will be added.
        dist += np.linalg.norm(pt - pt_)

        return dist, nearest_pt + offset

    def get_contour_img(self):
        """
        As it must be created, it might be useful to just reuse it somewhere else.
        :return: self.contour_img_
        """

        return self.contour_img_


if __name__ == '__main__':
    with open('/Volumes/Seagate Expansion Drive/regions-merged/472.pkl', 'rb') as f:
        data = pickle.load(f)

    reg = Region(data['region'])
    dm_region = DistanceMap(reg.pts())
    im = np.asarray(255*dm_region.contour_img_, dtype=np.uint8)
    cv2.imshow('contour', im)
    dm_im = dm_region.d_map
    dm_im = dm_region.d_map / np.max(dm_im)

    dm_im = np.asarray(dm_im*255, dtype=np.uint8)
    print dm_region.x_min, dm_region.y_min, dm_region.x_max, dm_region.y_max

    print [490, 205], dm_region.get_nearest_point([490, 205])
    print [480, 200], dm_region.get_nearest_point([480, 200])
    print [460, 200], dm_region.get_nearest_point([460, 200])
    print [500, 230], dm_region.get_nearest_point([500, 230])
    print [500, 260], dm_region.get_nearest_point([500, 260])

    cv2.imshow('dmap', dm_im)
    cv2.waitKey(0)