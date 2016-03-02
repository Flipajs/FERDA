from sklearn.ensemble import RandomForestClassifier
from core.project.project import Project
from core.graph.region_chunk import RegionChunk
from skimage.measure import moments_central, moments_hu, moments_normalized, moments
from utils.img_manager import ImgManager
import cv2
from utils.img import get_img_around_pts, replace_everything_but_pts
import cPickle as pickle
import numpy as np
import optparse

import os, sys
# TODO: REMOVE
from libs.mondrianforest.mondrianforest import MondrianForest, parser_add_common_options, parser_add_mf_options, process_command_line
from libs.mondrianforest.mondrianforest_utils import precompute_minimal


def get_training_data(p, get_features, first_n=-1, offset=0):
    X = []
    y = []

    vertices = p.gm.get_vertices_in_t(0)

    chunks = []
    for v in vertices:
        chunks.append(p.chm[p.gm.g.vp['chunk_start_id'][p.gm.g.vertex(v)]])

    id_ = 0
    for ch in chunks:
        r_ch = RegionChunk(ch, p.gm, p.rm)
        i = 0
        for t in xrange(offset, p.gm.end_t):
            if first_n > -1 and i == first_n:
                break

            r = r_ch.region_in_t(t)
            if not r.is_virtual:
                f_ = get_features(r, p)
                X.append(f_)
                y.append(id_)

                i += 1

        id_ += 1

    return X, y


def get_hu_moments(img):
    m = moments(img)
    cr = m[0, 1] / m[0, 0]
    cc = m[1, 0] / m[0, 0]

    mu = moments_central(img, cr, cc)
    nu = moments_normalized(mu)
    hu = moments_hu(nu)

    features = [m_ for m_ in hu]

    return features


def get_features1(r, p):
    f = []
    # area
    f.append(r.area())

    # major axis
    f.append(r.a_)

    # minor axis
    f.append(r.b_)

    # axis ratio
    f.append(r.a_ / r.b_)

    # axis ratio sqrt
    f.append((r.a_ / r.b_)**0.5)

    # axis ratio to power of 2
    f.append((r.a_ / r.b_)**2.0)

    img = p.img_manager.get_whole_img(r.frame_)
    crop, offset = get_img_around_pts(img, r.pts())

    pts_ = r.pts() - offset

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)

    ###### MOMENTS #####
    #### BINARY

    #### ONLY MSER PXs
    # in GRAY
    crop_gray_masked = replace_everything_but_pts(crop_gray, pts_)
    f.extend(get_hu_moments(crop_gray_masked))

    # B G R
    for i in range(3):
        crop_ith_channel_masked = replace_everything_but_pts(crop[:, :, i], pts_)
        f.extend(get_hu_moments(crop_ith_channel_masked))


    ### ALL PXs in crop image given margin
    crop, offset = get_img_around_pts(img, r.pts(), margin=0.3)

    # in GRAY
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    f.extend(get_hu_moments(crop_gray))

    # B G R
    for i in range(3):
        f.extend(get_hu_moments(crop[:, :, i]))

    return f


if __name__ == '__main__':
    p = Project()
    p.load('/Users/flipajs/Documents/wd/GT/Cam1/cam1.fproj')
    p.img_manager = ImgManager(p)

    if True:
        with open(p.working_directory+'/temp/rfc.pkl', 'rb') as f:
            p_ = pickle.Unpickler(f)
            X = p_.load()
            y = p_.load()
            rfc = p_.load()
            X2 = p_.load()
            y2 = p_.load()


        # settings = process_command_line()
        #
        # data = {'n_dim': 100}
        #
        # data = {'x_train': np.array(X), 'y_train': np.array(y), 'n_class': 6, \
        #     'n_dim': len(X[0]), 'n_train': len(X), 'is_sparse': False}
        #
        # mf = MondrianForest(settings, data)
        # param, cache = precompute_minimal(data, settings)
        # mf.fit(data, range(len(X)), settings, param, cache)

    else:
        X, y = get_training_data(p, get_features1, first_n=500)

        rfc = RandomForestClassifier()
        rfc.fit(X, y)

        mf = MondrianForest()

        print rfc.score(X, y)
        X2, y2 = get_training_data(p, get_features1, first_n=200, offset=1000)

        with open(p.working_directory+'/temp/rfc.pkl', 'wb') as f:
            p_ = pickle.Pickler(f)
            p_.dump(X)
            p_.dump(y)
            p_.dump(rfc)
            p_.dump(X2)
            p_.dump(y2)

    results = rfc.predict_proba(X2)
    y_results = rfc.predict(X2)

    idx = np.array(y_results) == np.array(y2)

    mismatches_proba = results[np.logical_not(idx)]
    matches_proba = results[idx]

    print np.sum(idx), len(idx)

    print "MISMATCHES"
    print ("mean: %.3f medain: %.3f STD: %.3f, median of max: %.3f") % \
          (np.mean(mismatches_proba),
           np.median(mismatches_proba),
           np.std(mismatches_proba),
           np.median(np.max(mismatches_proba, axis=1)))

    print "MATCHES"
    print ("mean: %.3f medain: %.3f STD: %.3f, median of max: %.3f") % \
          (np.mean(matches_proba),
           np.median(matches_proba),
           np.std(matches_proba),
           np.median(np.max(matches_proba, axis=1)))

    # print results, y2