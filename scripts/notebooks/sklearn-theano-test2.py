import numpy as np
import matplotlib.pyplot as plt
from sklearn_theano.datasets import load_sample_image
from sklearn_theano.feature_extraction import OverfeatLocalizer, OverfeatTransformer
from sklearn.mixture import GMM
import cv2
import time
import cPickle as pickle
from utils.img import get_img_around_pts, replace_everything_but_pts

from core.graph.region_chunk import RegionChunk

SAMPLING = 3

def process_tracklet(t, p, cnn):
    X = []
    r_ch = RegionChunk(t, p.gm, p.rm)
    i = 0
    for r in r_ch.regions_gen():
        # if i > 100:
        #     continue
        if not (i % SAMPLING):
            i += 1
            continue

        if not r.is_virtual:
            import math
            from utils.img import rotate_img, centered_crop, get_bounding_box, endpoint_rot, get_safe_selection
            relative_border = 3.0

            bb, offset = get_bounding_box(r, p, relative_border)
            p_ = np.array([r.a_ * math.sin(-r.theta_), r.a_ * math.cos(-r.theta_)])
            endpoint1 = np.ceil(r.centroid() + p_) + np.array([1, 1])
            endpoint2 = np.ceil(r.centroid() - p_) - np.array([1, 1])

            bb = rotate_img(bb, r.theta_)
            bb = centered_crop(bb, 8 * r.b_, 4 * r.a_)


            bb = cv2.resize(bb, (0, 0), fx=2.5, fy=4)

            # reshape to fit CNN shape
            W_ = 231
            H_ = 231

            im_ = np.zeros((231, 231, 3))
            # if bb.shape[0] > 231 or bb.shape[1] > 231:
            #     # todo
            #     raise Exception("Not implemented yet")
            # else:
            y = (bb.shape[0]-231)/2
            x = (bb.shape[1]-231)/2
            im_ = get_safe_selection(bb, y, x, H_, W_, fill_color=(0, 0, 0))


            cv2.imshow('im', np.fliplr(im_))
            cv2.waitKey(0)

            X.append(im_)
            # flip for augmentation...
            X.append(np.fliplr(im_))
            i += 1

    return cnn.transform(X)

def get_features(p, tracklets, cnn):
    features = {}
    i = 0
    for arr in tracklets.itervalues():
        for t_id in arr:
            t = p.chm[t_id]
            X = process_tracklet(t, p, cnn)

            i += 1
            features[t.id()] = X
            print t.id()

    return features

if __name__ == "__main__":
    # wd = '/Users/flipajs/Documents/wd/GT/Cam1_'
    # tracklets = {0: [6, 226, 116, 153],
    #             1: [14, 227, 113, 163],
    #             2: [3, 229, 108, 145],
    #             3: [13, 209, 105, 152],
    #             4: [12, 156, 112],
    #             5: [11, 214, 94]}

    wd = '/Users/flipajs/Documents/wd/zebrafish0'
    tracklets = {0: [5, 15],
                 1: [3, 16],
                 2: [4, 47],
                 3: [1, 45],
                 4: [2]}

    if False:
        from core.project.project import Project
        p = Project()
        p.load(wd)

        oft = OverfeatTransformer()

        features = get_features(p, tracklets, oft)
        with open(wd+'/temp/sklearn-theano.pkl', 'wb') as f:
            pickle.dump(features, f)
    else:
        with open(wd+'/temp/sklearn-theano.pkl', 'rb') as f:
            features = pickle.load(f)

        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', n_estimators=10)

        data = [[] for i in range(len(tracklets))]

        for id_, arr in tracklets.iteritems():
            for t_id_ in arr:
                data[id_].extend(list(features[t_id_]))

            data[id_] = np.array(data[id_])
            print data[id_].shape

        for_learning = 300

        X = []
        y = []
        for a_id, X_ in enumerate(data):
            X_ = X_[-for_learning:, :]

            print a_id, X_.shape
            if len(y) == 0:
                X = np.array(X_)
                y = np.array([a_id] * len(X_))
            else:
                X = np.vstack([X, np.array(X_)])
                y = np.append(y, [a_id] * len(X_))

        rfc.fit(np.array(X), y)

        for a_id, X_ in enumerate(data):
            X_ = X_[:-for_learning, :]

            print a_id, X_.shape
            probs = rfc.predict_proba(X_)

            maxs = np.argmax(probs, 1)
            print "maxs med: ", np.median(maxs)
            # print probs
            print np.mean(probs, 0)
            probs_ = np.mean(probs, 0)
            i_ = np.argmax(probs_)
            m_ = probs_[i_]
            probs_[i_] = 0
            m2_ = np.max(probs_)
            print i_, m_ / (m_ + m2_)

            print
            print


        #
        # # X = load_sample_image("cat_and_dog.jpg")
        # X2 = cv2.imread('/Users/flipajs/Desktop/Screen Shot 2016-10-10 at 11.06.20.png')
        # X3 = cv2.imread('/Users/flipajs/Desktop/Screen Shot 2016-10-10 at 11.06.25.png')
        #
        # oft = OverfeatTransformer()
        #
        # # f = oft.transform(X)
        # t = time.time()
        # f2 = oft.transform(np.array([X2, X3]))
        # print time.time() - t
        # t = time.time()
        # f3 = oft.transform(X3)
        # print time.time() - t
        #
        # print f2.shape
        # print f3.shape
        #
        # print "test"