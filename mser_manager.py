__author__ = 'flipajs'

import pickle
from utils import geometry
import split_by_contours
import numpy as np
import cv2


class MSERManager:
    def __init__(self, data=None):
        self._data = data
        self._pts = None
        self._contour = None
        self._roi = None

        return

    def set_data(self, data):
        self._data = data

    def contour(self):
        if not self._roi:
            self._roi = geometry.rle_roi(self._data['rle'])

        if not self._contour:
            self._contour = self._contour = rle_compute_contour(
                self._data['rle'], self._roi
            )

        return self._contour

    def area(self):
        return self._data['area']

    def centroid(self):
        return [self._data['cx'], self._data['cy']]

    def cx(self):
        return self._data['cx']

    def cy(self):
        return self._data['cy']

    def sxx(self):
        return self._data['sxx']

    def syy(self):
        return self._data['syy']

    def sxy(self):
        return self._data['sxy']

    def margin(self):
        return self._data['margin']

    def label(self):
        return self._data['label']


def rle_compute_contour(rle, roi=None):
    if not roi:
        roi = geometry.rle_roi(rle)

    cols, rows = geometry.roi_size(roi)
    min_c = roi[0][0]
    min_r = roi[0][1]
    img = np.zeros((rows, cols), dtype=np.uint8)

    for r in rle:
        row = r['line'] - min_r
        col1 = r['col1'] - min_c
        col2 = r['col2'] - min_c
        img[row][col1:col2+1] = 255

    cv2.imshow("img", img)
    ret, img_thresholded = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        img_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = contours[0]
    #cnt = []
    #
    #for c in contours:
    #    for pt in c:
    #        cnt.append(pt)

    img_cont = np.zeros((rows, cols), dtype=np.uint8)
    pts = []
    for p in contours:
        img_cont[p[0][1]][p[0][0]] = 255
        pts.append([p[0][0] + min_c, p[0][1] + min_r])

    cv2.imshow("test", img_cont)
    cv2.waitKey(0)

    return pts, img, img_cont, min_r, min_c


def load_region_data(id_ = 17):
    f = open('/home/flipajs/dump/collision_editor/regions/209.pkl', "rb")
    regions = pickle.load(f)

    f.close()

    return regions[id_]


def main():
    reg = load_region_data(9)
    test = MSERManager(reg)

    test.contour()


if __name__ == '__main__':
    main()