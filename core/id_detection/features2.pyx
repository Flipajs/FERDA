cimport numpy as np
from features import get_img_around_pts
import cv2

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def get_idtracker_features(r, p, debug=False):
    # import time

    max_d = 50
    # max_i = 100

    # zebrafish settings
    min_i = 0
    max_i = 210
    max_c = 50


    # # Camera3 Settings
    # max_d = 70
    #
    # min_i = 20
    # max_i = 90
    # max_c = 40


    img = p.img_manager.get_whole_img(r.frame_)
    crop, offset = get_img_around_pts(img, r.pts())

    # t1 = time.time()
    intensity_map_ = np.zeros((max_d, max_i + 1 - min_i), dtype=DTYPE_t)
    contrast_map_ = np.zeros((max_d, max_c + 1))

    pts = r.pts() - offset


    ids1_ = []
    ids2_ = []

    n_p = len(pts)
    for i in range(n_p):
        ids1_.extend([i for _ in xrange(n_p - (i+1))])
        ids2_.extend(range(i+1, n_p))

    ids1_ = np.array(ids1_)
    ids2_ = np.array(ids2_)
    pts_ = np.array(pts)

    x1_ = pts_[ids1_, :]
    x2_ = pts_[ids2_, :]

    d_ = np.linalg.norm(x1_ - x2_, axis=1) -1

    x1_ = x1_[d_ < max_d]
    x2_ = x2_[d_ < max_d]
    # -1 because 0 never occurs
    d_ = d_[d_ < max_d]

    crop = np.asarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), dtype=np.int)

    x1_i = crop[x1_[:, 0], x1_[:, 1]]
    x2_i = crop[x2_[:, 0], x2_[:, 1]]

    i_ = x1_i + x2_i
    c_ = np.abs(x1_i - x2_i)

    for d in range(max_d):
        ids_ = np.logical_and(d_ <= d, (d-1) < d_)

        i__ = i_[ids_]
        if len(i__) == 0:
            continue

        for i in range(max(i__.min(), min_i), min(i__.max(), max_i)+1):
            intensity_map_[d, i - min_i] += np.sum(i__ == i)

        c__ = c_[ids_]
        for c in range(c__.min(), min(c__.max(), max_c)+1):
            contrast_map_[d, c] += np.sum(c__ == c)

    # print time.time() - t1

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(intensity_map_, aspect='auto')
        plt.figure()
        plt.imshow(contrast_map_, aspect='auto')


    return list(np.ravel(intensity_map_)), list(np.ravel(contrast_map_))