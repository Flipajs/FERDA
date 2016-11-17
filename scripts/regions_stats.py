from core.project.project import Project
from core.graph.region_chunk import RegionChunk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from skimage.segmentation import random_walker
from skimage.data import binary_blobs
import skimage
import math
from functools import partial
import cPickle as pickle
import json
from core.learning.features import get_hu_moments
import hickle
from utils.video_manager import get_auto_video_manager
import cv2
from utils.drawing.points import draw_points
from utils.drawing.collage import create_collage_rows
from PyQt4.QtGui import QColor
from utils.img import rotate_img, get_bounding_box, centered_crop
from core.learning.features import get_features_var2, get_crop
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


EXP = 'exp1'


def plotNdto3d(data, labels, core_samples_mask, indices=[0, 1, 2], ax_labels=['', '', ''], title=''):
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = data[class_member_mask & core_samples_mask]
        ax.plot(xy[:, indices[0]], xy[:, indices[1]], xy[:, indices[2]], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=14)

        xy = data[class_member_mask & ~core_samples_mask]
        ax.plot(xy[:, indices[0]], xy[:, indices[1]], xy[:, indices[2]], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)


    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    ax.set_zlabel(ax_labels[2])
    plt.title(title)

def display_head_pairs(project):
    print "displaying pairs..."
    pairs = hickle.load('/Users/flipajs/Desktop/temp/pairs/pairs.pkl')
    print "loaded.."
    from utils.video_manager import get_auto_video_manager
    import cv2
    from utils.drawing.points import draw_points

    BORDER = 150
    COLS = 7
    IT_H = 500
    IT_W = 700

    D2_COEF = 5

    vm = get_auto_video_manager(project)

    major_axes = [project.gm.region(x[0][0]).a_ for x in pairs]
    major_axes_mean = np.mean(major_axes)

    print "major axes mean", major_axes_mean

    # sort by d2
    pairs = sorted(pairs, key=lambda x: -x[2])
    # d1 > 0.5major_axes_mean
    pairs = filter(lambda x: project.gm.region(x[0][0]).vector_on_major_axis_projection_head_unknown(project.gm.region(x[0][1])) > major_axes_mean, pairs)
    d2s = [x[2] for x in pairs]

    plt.hist(d2s, bins=20)
    plt.ion()
    plt.show()

    pairs = filter(lambda x: x[2] > D2_COEF*major_axes_mean, pairs)

    hickle.dump(pairs, '/Users/flipajs/Desktop/temp/pairs/head_pairs.pkl')

    return

    # new_pairs = []
    # for ((v1, v2), d1, d2) in pairs:
    #     if d1 > 20 and d2 > 200:
    #         new_pairs.append(((v1, v2), d1, d2))

    print "NEW LEN", len(pairs)

    i = 0
    part = 0
    data = []
    for ((v1, v2), d1, d2) in pairs:

        if v1 is None or v2 is None:
            continue

        r1 = project.gm.region(v1)
        r2 = project.gm.region(v2)

        if r1.frame() + 1 != r2.frame():
            print "FRAMES? ", r1.frame(), r2.frame()

        im1 = vm.get_frame(r1.frame()).copy()
        im2 = vm.get_frame(r2.frame()).copy()

        draw_points(im1, r1.pts())
        draw_points(im2, r2.pts())
        draw_points(im2, r1.pts(), color=QColor(0, 255, 255, 40))

        im1 = r1.roi().safe_roi(im1, border=BORDER)
        im2 = r1.roi().safe_roi(im2, border=BORDER)

        im = np.hstack((im1, im2))
        data.append(im)

        if i % (100) == 0 and i > 0:
            collage = create_collage_rows(data, COLS, IT_H, IT_W)
            cv2.imwrite('/Users/flipajs/Desktop/temp/pairs/' + str(part) + '.jpg', collage)
            data = []
            part += 1

        i += 1

    collage = create_collage_rows(data, COLS, IT_H, IT_W)
    cv2.imwrite('/Users/flipajs/Desktop/temp/pairs/' + str(part) + '.jpg', collage)

def display_regions(project, arr=None, labels=None):
    print "display regions"

    COLS = 15
    IT_W = 100
    IT_H = 100

    vm = get_auto_video_manager(project)

    if arr is None:
    # with open('/Users/flipajs/Desktop/temp/clustering/data1.pkl', 'rb') as f:
        d = hickle.load('/Users/flipajs/Desktop/temp/clustering/data1.pkl')

        labels = d['labels']
        arr = np.array(d['arr'])

    unique_labels = set(labels)
    for class_, k in enumerate(unique_labels):
        class_member_mask = (labels == k)
        a_ = arr[class_member_mask]

        data = []
        part = 0
        for i, v1 in enumerate(a_):
            if i%1000 == 0:
                print i

            if v1 is None:
                continue

            r1 = project.gm.region(v1)

            im1 = vm.get_frame(r1.frame()).copy()

            draw_points(im1, r1.pts())

            im = im1[r1.roi().slices()].copy()
            data.append(im)

            if i % (15*300) == 0 and i > 0:
                collage = create_collage_rows(data, COLS, IT_H, IT_W)
                cv2.imwrite('/Users/flipajs/Desktop/temp/clustering/' + str(class_) + '_' + str(part)+ '.jpg', collage)
                part += 1
                data = []

                print "TEST"

        collage = create_collage_rows(data, COLS, IT_H, IT_W)
        cv2.imwrite('/Users/flipajs/Desktop/temp/clustering/' + str(class_) + '_' + str(part) + '.jpg', collage)

def prepare_pairs(project):
    print "preparing pairs..."
    d = hickle.load('/Users/flipajs/Desktop/temp/clustering/labels.pkl')
    labels = d['labels']
    arr = d['arr']

    vs = set(arr[labels == 0])

    pairs = []
    for v in arr[labels == 0]:
        r1 = project.gm.region(v)
        best_v = None
        best_d = np.inf
        second_best_d = np.inf

        for v_out in project.gm.g.vertex(v).out_neighbours():
            r2 = p.gm.region(v_out)

            if r1.frame() + 1 != r2.frame():
                continue

            d = np.linalg.norm(r1.centroid() - r2.centroid())

            if d < best_d:
                second_best_d = best_d
                best_d = d
                best_v = v_out

            elif d < second_best_d:
                second_best_d = d

        if best_v is not None and int(best_v) in vs:
            pairs.append(((int(v), int(best_v)), best_d, second_best_d))

    hickle.dump(pairs, '/Users/flipajs/Desktop/temp/pairs/pairs.pkl')

def __get_mu_moments_pick(img):
    from core.learning.features import get_mu_moments
    nu = get_mu_moments(img)

    return list(nu[np.logical_not(np.isnan(nu))])

def head_features(r, swap=False):
    # normalize...
    from utils.geometry import rotate
    from utils.drawing.points import draw_points_crop_binary
    import cv2
    from skimage.measure import moments_central, moments_hu, moments_normalized, moments

    pts = np.array(rotate(r.pts(), -r.theta_, r.centroid(), method='back_projection'))
    img = draw_points_crop_binary(pts)
    img = np.asarray(img, dtype=np.uint8)

    if swap:
        img = np.fliplr(img)

    features = []
    nu = __get_mu_moments_pick(img)
    features.extend(nu)
    nu = __get_mu_moments_pick(img[:, :img.shape[1] / 2])
    features.extend(nu)
    nu = __get_mu_moments_pick(img[:, img.shape[1] / 2:])
    features.extend(nu)

    # cv2.imshow('test', img*255)
    # cv2.waitKey(0)

    return features

def head_detector_features(p, display=False):
    from core.region.region import get_region_endpoints

    pairs = hickle.load('/Users/flipajs/Desktop/temp/pairs/head_pairs.pkl')

    BORDER = 20
    COLS = 10
    IT_H = 50
    IT_W = 100

    data_heads = []
    data_swap = []
    data = []
    i = 0
    part = 0
    for (v1, v2), _, _ in pairs:
        r = p.gm.region(v1)
        r2 = p.gm.region(v2)
        f1 = head_features(r)

        f2 = head_features(r, swap=True)

        f1_, f2_ = get_features_var2(r, p, fliplr=True)
        f1.extend(f1_)
        f2.extend(f2_)
        #
        p1, p2 = get_region_endpoints(r)
        #
        # # if p1 is a head, than knowing that the ant moved in direction of main axis at least by major_axis_mean...
        if np.linalg.norm(r2.centroid() - p1) > np.linalg.norm(r2.centroid() - p2):
            r.theta_ += np.pi
            if r.theta_ > 2*np.pi:
                r.theta_ -= 2*np.pi

            f1, f2 = f2, f1
        #
        data_heads.append(np.array(f1))
        data_swap.append(np.array(f2))

        if display:
            crop = get_crop(r, p)
            data.append(crop)

            if i % (500) == 0 and i > 0:
                collage = create_collage_rows(data, COLS, IT_H, IT_W)
                cv2.imwrite('/Users/flipajs/Desktop/temp/pairs/' + EXP + '/heads_train' + str(part) + '.jpg', collage)
                data = []
                part += 1

        i += 1

    collage = create_collage_rows(data, COLS, IT_H, IT_W)
    cv2.imwrite('/Users/flipajs/Desktop/temp/pairs/' + EXP + '/heads_train' + str(part) + '.jpg', collage)


    hickle.dump(data_heads, '/Users/flipajs/Desktop/temp/pairs/'+EXP+'/head_data.pkl')
    hickle.dump(data_swap, '/Users/flipajs/Desktop/temp/pairs/'+EXP+'/head_data_swap.pkl')

def head_detector_classify(p):
    from sklearn.ensemble import RandomForestClassifier

    data_head = hickle.load('/Users/flipajs/Desktop/temp/pairs/'+EXP+'/head_data.pkl')
    data_swap = hickle.load('/Users/flipajs/Desktop/temp/pairs/'+EXP+'/head_data_swap.pkl')
    rfc = RandomForestClassifier()

    X = np.vstack((np.array(data_head), np.array(data_swap)))
    y = np.hstack((np.zeros((len(data_head), ), dtype=np.int), np.ones((len(data_swap), ), dtype=np.int)))
    rfc.fit(X, y)

    print rfc.feature_importances_

    d = hickle.load('/Users/flipajs/Desktop/temp/clustering/labels.pkl')
    labels = d['labels']
    arr = d['arr']

    BORDER = 20
    COLS = 10
    IT_H = 50
    IT_W = 100

    data = []
    part = 0
    i = 0
    for v in arr[labels == 0]:
        r = p.gm.region(v)

        f = head_features(r)
        f.extend(get_features_var2(r, p))

        probs = rfc.predict_proba(np.array([f]))[0]

        if probs[1] > 0.5:
            r.theta_ += np.pi
            if r.theta_ > 2*np.pi:
                r.theta_ -= 2*np.pi

        crop = get_crop(r, p, margin=10)
        # bb, offset = get_bounding_box(r, p)
        # bb = rotate_img(bb, r.theta_)
        # bb = centered_crop(bb, 8 * r.b_, 4 * r.a_)
        # crop = bb

        cv2.putText(crop, str(probs[0]), (10, 10),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, (255, 255, 255))

        data.append(crop)

        if i % (500) == 0 and i > 0:
            collage = create_collage_rows(data, COLS, IT_H, IT_W)
            cv2.imwrite('/Users/flipajs/Desktop/temp/pairs/'+EXP+'/heads' + str(part) + '.jpg', collage)
            data = []
            part += 1

        i += 1

    collage = create_collage_rows(data, COLS, IT_H, IT_W)
    cv2.imwrite('/Users/flipajs/Desktop/temp/pairs/'+EXP+'/heads' + str(part) + '.jpg', collage)


def get_movement_descriptor(p, v1, v2, v3):
    from math import atan2
    from utils.geometry import rotate

    r1 = p.gm.region(v1)
    r2 = p.gm.region(v2)
    r3 = p.gm.region(v3)

    v1 = r2.centroid() - r1.centroid()
    v2 = r3.centroid() - r2.centroid()

    theta = atan2(v1[0], v[1])

    d = np.linalg.norm(v1)
    v2 = rotate(v2, theta)

    return (d, v2[0], v2[1])

# def prepare_triplets(p):
#     print "preparing pairs..."
#     d = hickle.load('/Users/flipajs/Desktop/temp/clustering/labels.pkl')
#     labels = d['labels']
#     arr = d['arr']
#
#     vs = set(arr[labels == 0])
#
#     pairs = []
#     for v in arr[labels == 0]:
#         r1 = project.gm.region(v)
#         best_v = None
#         best_d = np.inf
#         second_best_d = np.inf
#
#         for v_out in project.gm.g.vertex(v).out_neighbours():
#             r2 = p.gm.region(v_out)
#
#             if r1.frame() + 1 != r2.frame():
#                 continue
#
#             d = np.linalg.norm(r1.centroid() - r2.centroid())
#
#             if d < best_d:
#                 second_best_d = best_d
#                 best_d = d
#                 best_v = v_out
#
#             elif d < second_best_d:
#                 second_best_d = d
#
#         if best_v is not None and int(best_v) in vs:
#             pairs.append(((int(v), int(best_v)), best_d, second_best_d))
#
#     hickle.dump(pairs, '/Users/flipajs/Desktop/temp/pairs/pairs.pkl')


def get_max_dist(project):
    pairs = hickle.load('/Users/flipajs/Desktop/temp/pairs/pairs.pkl')

    max_dist = 0
    max_v1 = None
    max_v2 = None

    for (v1, v2), d1, _ in pairs:
        if d1 > max_dist:
            max_dist = d1
            max_v1 = v1
            max_v2 = v2

    r1 = project.gm.region(max_v1)
    r2 = project.gm.region(max_v2)

    if r1.frame() + 1 != r2.frame():
        print "FRAMES? ", r1.frame(), r2.frame()

    vm = get_auto_video_manager(project)

    im1 = vm.get_frame(r1.frame()).copy()
    im2 = vm.get_frame(r2.frame()).copy()

    draw_points(im1, r1.pts())
    draw_points(im2, r2.pts())
    draw_points(im2, r1.pts(), color=QColor(0, 255, 255, 40))

    im = np.hstack((im1, im2))
    cv2.imshow('im', im)
    cv2.waitKey(0)

    return max_dist

if __name__ == '__main__':
    p = Project()
    p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_playground')

    from core.region.region_manager import RegionManager
    p.rm = RegionManager('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp', db_name='part0_rm.sqlite3')
    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        g_ = up.load()
        # relevant_vertices = up.load()
        # chm_ = up.load()

    p.gm.g = g_

    # prepare_pairs(p)
    # display_head_pairs(p)

    # head_detector_features(p)
    # head_detector_classify(p)

    print "MAX DIST: {}".format(get_max_dist(p))

    i = 0
    if False:
        arr = []
        data = []
        areas = []
        major_axes = []

        r_data = []
        r_arr = []

        second_dists = []

        for v in p.gm.g.vertices():
            r = p.gm.region(v)

            from utils.drawing.points import draw_points_crop_binary
            bimg = draw_points_crop_binary(r.pts())
            hu_m = get_hu_moments(np.asarray(bimg, dtype=np.uint8))
            r_data.append([r.area(), r.a_, r.b_, hu_m[0], hu_m[1]])
            r_arr.append(int(v))

            i += 1

        data = np.array(data)
        label_names = np.array(['area', 'area_t1 - area_t2', 'best distance', 'second best distance', 'axis ratio'])

        data = np.array(r_data)
        label_names = np.array(['area', 'major axis', 'minor axis', 'hu1', 'hu2'])

        arr = np.array(r_arr)

        X = StandardScaler().fit_transform(data)
        db = DBSCAN(eps=0.1, min_samples=int(len(data)*0.001)).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        print('Estimated number of clusters: %d' % n_clusters_)

        # plotNdto3d(data, labels, core_samples_mask, [0, 1, 2], label_names[[0, 1, 2]])
        # plotNdto3d(data, labels, core_samples_mask, [0, 2, 3], label_names[[0, 2, 3]])
        # plotNdto3d(data, labels, core_samples_mask, [0, 2, 4], label_names[[0, 2, 4]])

        # with open('/Users/flipajs/Desktop/temp/clustering/data1.pkl', 'wb') as f:
        hickle.dump({'arr': arr, 'labels': labels}, '/Users/flipajs/Desktop/temp/clustering/labels.pkl')
            # pickle.dump({'data': data, 'arr': arr, 'labels': labels, 'core_samples_mask': core_samples_mask}, f)

    # display_regions(p, arr, labels)
    # plt.show()