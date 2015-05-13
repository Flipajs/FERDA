__author__ = 'fnaiser'

import pickle

import numpy as np
from scipy import ndimage
import cv2
from core.region.region import Region
from utils.drawing.points import get_contour, get_roi


class DistanceMap():
    """
    This class encapsulates distance map image, which is used to speed up search for nearest point for given coordinates.
    """

    def __init__(self, pts, only_cont=False):
        pts = np.asarray(pts, dtype=np.int32)

        self.roi = get_roi(pts)

        if not only_cont:
            self.pt_img_ = np.zeros((self.roi.height(), self.roi.width()), dtype=np.bool)
            self.pt_img_[pts[:,0] - self.roi.y(), pts[:,1] - self.roi.x()] = True

        self.contour_img_ = np.ones((self.roi.height(), self.roi.width()), dtype=np.bool)

        if only_cont:
            self.cont_pts_ = pts
        else:
            self.cont_pts_ = get_contour(pts)

        self.contour_img_[self.cont_pts_[:, 0] - self.roi.y(), self.cont_pts_[:, 1] - self.roi.x()] = False

        self.d_map, self.d_map_labels = ndimage.distance_transform_edt(self.contour_img_, return_indices=True)

    def get_nearest_point(self, pt):
        """
        Returns distance and nearest point coordinates on object measured from given pt.
        In case when the pt is out of ROI of distance image, the distance is inaccurate. But this still holds: estimated_dist >= real_dist
        :param pt:
        :return: distance, [y, x]
        """

        offset = self.roi.top_left_corner()
        pt = np.array(pt)
        pt_ = self.roi.nearest_pt_in_roi(pt[0], pt[1])

        y_, x_ = pt_ - offset
        nearest_pt = self.d_map_labels[:, y_, x_]
        dist = self.d_map[y_, x_]

        # If the point is outside ROI - the distance to distance map border will be added.
        dist += np.linalg.norm(pt - pt_)

        return nearest_pt + offset, dist

    def get_contour_img(self):
        """
        As it must be created, it might be useful to just reuse it somewhere else.
        :return: self.contour_img_
        """

        return self.contour_img_

    def cont_pts(self):
        return self.cont_pts_

    def is_inside_object(self, pt):
        if self.roi.is_inside(pt):
            if self.pt_img_[pt[0] - self.roi.y(), pt[1] - self.roi.x()]:
                return True

        return False


if __name__ == '__main__':
    with open('/Volumes/Seagate Expansion Drive/regions-merged/74.pkl', 'rb') as f:
        data = pickle.load(f)

    reg = Region(data['region'])
    dm_region = DistanceMap(reg.pts())
    im = np.asarray(255*dm_region.contour_img_, dtype=np.uint8)
    cv2.imshow('contour', im)
    dm_im = dm_region.d_map
    dm_im = dm_region.d_map / np.max(dm_im)

    dm_im = np.asarray(dm_im*255, dtype=np.uint8)
    # print dm_region.x_min, dm_region.y_min, dm_region.x_max, dm_region.y_max

    print [490, 205], dm_region.get_nearest_point([490, 205])
    print [480, 200], dm_region.get_nearest_point([480, 200])
    print [460, 200], dm_region.get_nearest_point([460, 200])
    print [500, 230], dm_region.get_nearest_point([500, 230])
    print [500, 260], dm_region.get_nearest_point([500, 260])

    cv2.imshow('dmap', dm_im)
    cv2.waitKey(0)