__author__ = 'filip@naiser.cz'
import numpy as np
import math
import sys

def rotate(pts, theta_radians, center=[0, 0], method='', roi=None):
    """Rotates given list of points which is list of lists [x1, x2]
    ([[1, 1], [2, 3], [10, 1]).

    center must be in format [c1, c2]
    method should be 'back_projection' when the ROI points are
    transformed and then ROI in destination is established
    from which points are transformed back asking whether
    there is or isn't point

    """

    if method == 'back_projection':
        pts = _rotate_back_projection(pts, theta_radians, center, roi)
    else:
        pts = _rotate(pts, theta_radians, center)

    return pts


def rotation_matrix(theta, degrees=False):
    """ returns np.array 2x2 rotation matrix
    """

    if degrees:
        theta = math.radians(theta)

    rot = np.array(
        [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta), math.cos(theta)]]
    )

    return rot


def angle_from_matrix(R):
    t1 = math.acos(R[0, 0])
    t2 = math.asin(R[1, 0])

    return t1*np.sign(t2)


def _rotate(pts, th, center, rot=None):
    if rot is None:
        rot = rotation_matrix(th)

    cx = center[0]
    cy = center[1]

    new_pts = [None]*len(pts)
    for i in range(len(pts)):
        pt = pts[i]
        np_pt = np.array([pt[0] - cx, pt[1] - cy])
        np_pt = np.dot(rot, np_pt.reshape(2, 1))

        new_pts[i] = [int(round(np_pt[0][0] + cx)),
                      int(round(np_pt[1][0] + cy))]

    return new_pts


def _rotate_back_projection(pts, th, center, roi):
    if not roi:
        roi = pts_roi(pts)

    roi_corners_ = roi_corners(roi)
    roi_corners_img = _rotate(roi_corners_, th, center)
    roi_img = pts_roi(roi_corners_img)

    inv_rot_matrix = rotation_matrix(-th)
    new_pts = []

    x_range = roi_x_range(roi_img)
    y_range = roi_y_range(roi_img)

    from utils.drawing.points import draw_points_crop_binary
    im = draw_points_crop_binary(pts)

    for i in x_range:
        for j in y_range:
            pt = _rotate([[i, j]], -th, center, inv_rot_matrix)
            # TODO: old format roi [[min_x, min_y], [...]]
            y = pt[0][0] - roi[0][0]
            x = pt[0][1] - roi[0][1]
            if 0 <= y < im.shape[0]:
                if 0 <= x < im.shape[1]:
                    if im[y, x]:
                        new_pts.append([i, j])

    return new_pts


def roi_x_range(roi):
    return list(range(roi[0][0], roi[1][0] + 1))


def roi_y_range(roi):
    return list(range(roi[0][1], roi[1][1] + 1))


def roi_corners(roi):
    """
    roi_corners = [top_left, top_right, bottom_left, bottom_right]
    """

    return [
        roi[0],
        [roi[1][0], roi[0][1]],
        [roi[0][0], roi[1][1]],
        roi[1]
    ]


def count_centroid(pts):
    """ returns array of centroid computed from given points
    in format [x, y]
    """
    s = [sum(x) for x in zip(*pts)]
    centroid = [s[0]/len(pts), s[1]/len(pts)]

    return centroid


def pts_roi(pts):
    """ for given pts (list of lists in format [x, y]) returns
    ROI - Region Of Interest in format [[min_x, min_y], [max_x, max_y]]

    returns None if something goes wrong
    """
    min_x = sys.maxsize
    min_y = sys.maxsize
    max_x = 0
    max_y = 0
    for pt in pts:
        if pt[0] < min_x:
            min_x = pt[0]
        if pt[1] < min_y:
            min_y = pt[1]
        if pt[0] > max_x :
            max_x = pt[0]
        if pt[1] > max_y:
            max_y = pt[1]

    return [[min_x, min_y], [max_x, max_y]]


def rle_roi(rle, rle_ordered=True):
    """ computes ROI for RLE (Run-Length Encoding) formatted region
    rle mus be array of dicts {'line', 'col1', 'col2'}

    ROI - Region Of Interest in format [[min_x, min_y], [max_x, max_y]]
    returns None if something goes wrong
    """
    min_x = sys.maxsize
    min_y = sys.maxsize
    max_x = 0
    max_y = 0

    if rle_ordered:
        min_y = rle[0]['line']
        max_y = rle[-1]['line']

        for r in rle:
            if min_x > r['col1']:
                min_x = r['col1']
            if max_x < r['col2']:
                max_x = r['col2']
    else:
        for r in rle:
            if min_x > r['col1']:
                min_x = r['col1']
            if min_y > r['line']:
                min_y = r['line']
            if max_x < r['col2']:
                max_x = r['col2']
            if max_y < r['line']:
                max_y = r['line']

    # due col1:col2 may be 1:1... thus min_x is 1, max_x is 1
    return [[min_x, min_y], [max_x, max_y]]


def roi_size(roi):
    """
    returns number of cols and rows
    """
    cols = roi[1][0] - roi[0][0] + 1
    rows = roi[1][1] - roi[0][1] + 1

    return cols, rows


def get_region_group_overlaps(rt1, rt2):
    import cv2
    roi_union = rt1[0].roi()

    for r in rt1 + rt2:
        roi_union = roi_union.union(r.roi())

    print(roi_union)

    h = roi_union.y_ + roi_union.height_
    w = roi_union.x_ + roi_union.width_

    im = np.zeros((h, w, 3), dtype=np.uint8)

    for i, r in enumerate(rt1):
        pts = r.pts()
        im[pts[:,0], pts[:, 1], 1] = 150

    for i, r in enumerate(rt2):
        pts = r.pts()
        im[pts[:,0], pts[:, 1], 2] = 150

    cv2.imshow('test', im)
    cv2.waitKey(0)