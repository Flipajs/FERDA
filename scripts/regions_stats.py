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
from sklearn import svm, preprocessing
from itertools import izip


EXP = 'exp1'
DEFAULT_H_DENSITY = 1e-10


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

def display_regions(project, arr=None, labels=None,
                    in_f_name='/Users/flipajs/Desktop/temp/clustering/data1.pkl',
                    f_name='/Users/flipajs/Desktop/temp/clustering/'):
    print "display regions"

    COLS = 15
    IT_W = 100
    IT_H = 100

    vm = get_auto_video_manager(project)

    if arr is None:
        try:
        # with open('/Users/flipajs/Desktop/temp/clustering/data1.pkl', 'rb') as f:
            d = hickle.load(in_f_name)

            labels = d['labels']
            arr = np.array(d['arr'])
        except:
            with open(in_f_name) as f:
                d = pickle.load(f)

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
                cv2.imwrite(f_name + str(class_) + '_' + str(part)+ '.jpg', collage)
                part += 1
                data = []

                print "TEST"

        collage = create_collage_rows(data, COLS, IT_H, IT_W)
        cv2.imwrite(f_name + str(class_) + '_' + str(part) + '.jpg', collage)

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

def fix_head(p, r, rfc):
    f = head_features(r)
    f.extend(get_features_var2(r, p))

    probs = rfc.predict_proba(np.array([f]))[0]

    if probs[1] > 0.5:
        r.theta_ += np.pi
        if r.theta_ > 2 * np.pi:
            r.theta_ -= 2 * np.pi

def fix_heads(p, frames):
    with open('/Users/flipajs/Desktop/temp/pairs/'+EXP+'/head_rfc.pkl', 'rb') as f:
        rfc = pickle.load(f)

    for v in p.gm.g.vertices():
        r = p.gm.region(v)

        if r.frame() not in frames:
            continue

        f = head_features(r)
        f.extend(get_features_var2(r, p))

        probs = rfc.predict_proba(np.array([f]))[0]

        if probs[1] > 0.5:
            r.theta_ += np.pi
            if r.theta_ > 2 * np.pi:
                r.theta_ -= 2 * np.pi



def head_detector_classify(p):
    from sklearn.ensemble import RandomForestClassifier

    data_head = hickle.load('/Users/flipajs/Desktop/temp/pairs/'+EXP+'/head_data.pkl')
    data_swap = hickle.load('/Users/flipajs/Desktop/temp/pairs/'+EXP+'/head_data_swap.pkl')
    rfc = RandomForestClassifier()

    X = np.vstack((np.array(data_head), np.array(data_swap)))
    y = np.hstack((np.zeros((len(data_head), ), dtype=np.int), np.ones((len(data_swap), ), dtype=np.int)))
    rfc.fit(X, y)

    with open('/Users/flipajs/Desktop/temp/pairs/'+EXP+'/head_rfc.pkl', 'wb') as f:
        pickle.dump(rfc, f)

    return

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
    r1 = p.gm.region(v1)
    r2 = p.gm.region(v2)
    r3 = p.gm.region(v3)

    v1 = r2.centroid() - r1.centroid()
    v2 = r3.centroid() - r2.centroid()

    return get_movement_descriptor_(v1, v2)

def get_movement_descriptor_(v1, v2):
    from math import atan2
    from utils.geometry import rotate

    theta = atan2(v1[0], v1[1])

    d = np.linalg.norm(v1)
    v2 = rotate([v2], theta)[0]

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


def filter_edges(project, max_dist):
    to_remove = []

    g = project.gm.g
    print "avg degree before {}".format(np.mean([v.out_degree() for v in g.vertices()]))

    for (v1, v2) in g.edges():
        r1 = project.gm.region(v1)
        r2 = project.gm.region(v2)

        if r1.is_ignorable(r2, max_dist):
            to_remove.append((v1, v2))

    print "#edges: {}, will be removed: {}".format(g.num_edges(), len(to_remove))
    for (v1, v2) in to_remove:
        g.remove_edge(g.edge(v1, v2))

    degrees = [v.out_degree() for v in g.vertices()]
    print "avg degree after {}".format(np.mean(degrees))

    # plt.hist(degrees)
    # plt.show()
    #

    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_modified.pkl', 'wb') as f:
        pickle.dump(g, f)


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

def hist_query(h, edges, it):
    ids = []
    for i in range(3):
        id_ = np.argmax(it[i] < edges[i]) - 1

        if 0 <= id_ < len(edges[i]):
            ids.append(id_)
        else:
            return 0

    return h[ids[0], ids[1], ids[2]]

def get_movement_histogram(p):
    # with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_modified.pkl', 'rb') as f:
    #     g = pickle.load(f)
    #     _ = pickle.load(f)
    #     chm = pickle.load(f)

    # p.gm.g = g

    d = hickle.load('/Users/flipajs/Desktop/temp/clustering/labels.pkl')
    labels = d['labels']
    arr = d['arr']

    data = []
    data2 = []
    cases = []

    # for v in arr[labels==0]:
    #     v = p.gm.g.vertex(v)
    #
    #     if v.out_degree() == 1:
    #         for w in v.out_neighbours():
    #             if w.in_degree() == 1 and w.out_degree() == 1:
    #                 for x in w.out_neighbours():
    #                     if x.in_degree() == 1:
    #                         data.append(get_movement_descriptor(p, v, w, x))
    #             elif w.in_degree() == 1 and w.out_degree() > 1:
    #                 data2.append([])
    #                 cases.append([])
    #                 for x in w.out_neighbours():
    #                     data2[-1].append(get_movement_descriptor(p, v, w, x))
    #                     cases[-1].append(map(int, (v, w, x)))

    for ch in p.chm.chunk_gen():
        for i in range(ch.length() - 2):
            v = ch[i]
            w = ch[i+1]
            x = ch[i+2]

            data.append(get_movement_descriptor(p, v, w, x))

    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/movement_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    data = np.array(data)

    H, edges = np.histogramdd(data, bins=10)
    data = data[::5, :]

    ax.scatter(data[:, 1], data[:, 2], data[:, 0], c=data[:, 0]/data[:, 0].max())

    data2 = np.array(data2)
    from itertools import izip

    cases_p = []
    cases_n = []

    for it, case in izip(data2, cases):
        it = np.array(it)

        # if np.linalg.norm(it[0, :] - it[1, :]) > 60:
        #     continue

        vals = []
        for i in range(it.shape[0]):
            val = hist_query(H, edges, it[i, :], default=DEFAULT_H_DENSITY)
            vals.append(val)

        if max(vals) > DEFAULT_H_DENSITY:
            cases_p.append((vals, case))
            # print case
            # print vals
        else:
            cases_n.append((vals, case))
            continue

        ax.plot(it[:, 1], it[:, 2], it[:, 0], marker='v', c='m')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('d')

    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/case_p.pkl', 'wb') as f:
        pickle.dump(cases_p, f)

    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/case_n.pkl', 'wb') as f:
        pickle.dump(cases_n, f)

    # plt.ion()
    plt.show()


def observe_cases(project, type='case_p'):
    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/'+type+'.pkl') as f:
        cases = pickle.load(f)

    from utils.video_manager import get_auto_video_manager
    import cv2
    from utils.drawing.points import draw_points
    from itertools import izip

    BORDER = 150
    COLS = 1
    IT_H = 500
    IT_W = 900

    vm = get_auto_video_manager(project)

    i = 0
    part = 0
    data = []
    for vals, case in cases:
        for val, (v, x, w) in izip(vals, case):
            c = QColor(0, 255, 0, 70)
            if val <= 1e-10:
                c = QColor(255, 0, 0, 70)
                # continue

            r1 = project.gm.region(v)
            r2 = project.gm.region(x)
            r3 = project.gm.region(w)

            im1 = vm.get_frame(r1.frame()).copy()
            im2 = vm.get_frame(r2.frame()).copy()
            im3 = vm.get_frame(r3.frame()).copy()

            draw_points(im1, r1.pts(), color=c)
            draw_points(im2, r2.pts(), color=c)
            draw_points(im3, r3.pts(), color=c)
            # draw_points(im2, r1.pts(), color=QColor(0, 255, 255, 40))

            im1 = r1.roi().safe_roi(im1, border=BORDER)
            im2 = r1.roi().safe_roi(im2, border=BORDER)
            im3 = r1.roi().safe_roi(im3, border=BORDER)

            im = np.hstack((im1, im2, im3))

            cv2.putText(im, str(val), (50, 50),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (255, 255, 255))
            data.append(im)

            if i % (50) == 0 and i > 0:
                collage = create_collage_rows(data, COLS, IT_H, IT_W)
                cv2.imwrite('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/'+type+'_'+ str(part) + '.jpg', collage)
                data = []
                part += 1

            i += 1

    collage = create_collage_rows(data, COLS, IT_H, IT_W)
    cv2.imwrite('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/'+type+'_' + str(part) + '.jpg', collage)


def expand_based_on_movement_model(p):
    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/movement_data.pkl', 'rb') as f:
        data = pickle.load(f)

    data = np.array(data)

    H, edges = np.histogramdd(data, bins=10)
    THRESH = 100.0

    # when merge... it will change size
    ch_keys = p.chm.chunks_.keys()
    for t_id in ch_keys:
        if t_id not in p.chm.chunks_:
            continue

        t = p.chm[t_id]

        if t.length() > 1:
            while True:
                v = t.start_vertex(p.gm)
                if v.in_degree() > 0:
                    options = []

                    for v2 in v.in_neighbours():
                        val = hist_query(H, edges, get_movement_descriptor(p, v2, t[0], t[1]))
                        options.append((val + 1, v2))

                    options = sorted(options, key=lambda x: -x[0])

                    if len(options) > 1:
                        ratio = options[0][0] / options[1][0]
                    else:
                        ratio = options[0][0]

                    if ratio > THRESH:
                        v2 = options[0][1]
                        t.append_left(v2, p.gm)
                    else:
                        break
                else:
                    break

            while True:
                v = t.end_vertex(p.gm)
                if v.out_degree() > 0:
                    options = []

                    for v2 in v.out_neighbours():
                        val = hist_query(H, edges, get_movement_descriptor(p, t[-2], t[-1], v2))
                        options.append((val + 1, v2))

                    options = sorted(options, key=lambda x: -x[0])

                    if len(options) > 1:
                        ratio = options[0][0] / options[1][0]
                    else:
                        ratio = options[0][0]

                    if ratio > THRESH:
                        v2 = options[0][1]
                        t.append_right(v2, p.gm)
                    else:
                        break
                else:
                    break

    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_tracklets_expanded.pkl', 'wb') as f:
        pic = pickle.Pickler(f)
        pic.dump(p.gm.g)
        pic.dump([])
        pic.dump(p.chm)

def simple_tracklets(p):
    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_modified.pkl', 'rb') as f:
        g = pickle.load(f)

    p.gm.g = g
    p.gm.update_nodes_in_t_refs()

    d = hickle.load('/Users/flipajs/Desktop/temp/clustering/labels.pkl')
    labels = d['labels']
    vertices_ids = np.array(d['arr'])

    from core.graph.chunk_manager import ChunkManager
    p.chm = ChunkManager()


    singles_ids = list(vertices_ids[labels==0])

    print "BEFORE:"
    print "#vertices: {} #edges: {}".format(p.gm.g.num_vertices(), p.gm.g.num_edges())

    singles_set = set(singles_ids)

    for v in singles_ids:
        # if already removed (added into tracklet, continue...)
        if v not in singles_set:
            continue

        v = p.gm.g.vertex(v)

        new_t_vertices_ = []

        # expand backward
        v_ = v
        do_break = False
        while not do_break:
            if v_.in_degree() == 1:
                for v2 in v_.in_neighbours():
                    if v2.out_degree() > 1:
                        do_break = True
                        break

                    new_t_vertices_.append(v2)

                v_ = v2
            else:
                break

        new_t_vertices_ = list(reversed(new_t_vertices_))
        new_t_vertices_.append(v)

        # expand forward
        v_ = v
        do_break = False
        while not do_break:
            if v_.out_degree() == 1:
                for v2 in v_.out_neighbours():
                    if v2.in_degree() > 1:
                        do_break = True
                        break

                    new_t_vertices_.append(v2)

                v_ = v2
            else:
                break

        new_t_vertices_ = map(int, new_t_vertices_)

        if len(new_t_vertices_) > 1:
            for v in new_t_vertices_:
                singles_set.discard(v)

            ch, _ = p.chm.new_chunk(new_t_vertices_, p.gm)
            if ch.length() == 1:
                print "WTF"

    print "BEFORE:"
    print "#vertices: {} #edges: {}".format(p.gm.g.num_vertices(), p.gm.g.num_edges())
    print "#chunks: {}".format(len(p.chm))

    for ch in p.chm.chunk_gen():
        if ch.length() == 1:
            print ch

    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_tracklets.pkl', 'wb') as f:
        pic = pickle.Pickler(f)
        pic.dump(p.gm.g)
        pic.dump([])
        pic.dump(p.chm)


def display_classification(project, ids, labels):
    print "display regions"
    F_NAME = 'singles_classif'

    COLS = 15
    IT_W = 100
    IT_H = 100

    vm = get_auto_video_manager(project)

    unique_labels = set(labels)
    for class_, k in enumerate(unique_labels):
        class_member_mask = (labels == k)
        ids_ = ids[class_member_mask]

        data = []
        part = 0
        for i, v1 in enumerate(ids_):
            if i % 1000 == 0:
                print i

            if v1 is None:
                continue

            r1 = project.gm.region(v1)

            im1 = vm.get_frame(r1.frame()).copy()

            draw_points(im1, r1.pts())

            im = im1[r1.roi().slices()].copy()
            data.append(im)

            if i % (15 * 300) == 0 and i > 0:
                collage = create_collage_rows(data, COLS, IT_H, IT_W)
                cv2.imwrite('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/' + F_NAME + str(class_) + '_' + str(part) + '.jpg',
                            collage)
                part += 1
                data = []

                print "TEST"

        collage = create_collage_rows(data, COLS, IT_H, IT_W)
        cv2.imwrite('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/' + F_NAME + str(class_) + '_' + str(part) + '.jpg', collage)

def singles_classifier(p):
    d = hickle.load('/Users/flipajs/Desktop/temp/clustering/labels.pkl')
    labels = d['labels']
    arr = np.array(d['arr'])
    data = d['data']

    unique_labels = set(labels)

    singles_labels = set([0, 4])
    non_singles_labels = set([1, 2, 3, 5])

    X = []
    y = []

    for class_, k in enumerate(unique_labels):
        if class_ in singles_labels:
            c_ = 1
        elif class_ in non_singles_labels:
            c_ = 0
        else:
            continue

        class_member_mask = (labels == k)
        a_ = arr[class_member_mask]
        data_ = data[class_member_mask]

        X.extend(data_)
        y.extend([c_ for _ in range(len(a_))])

    X = np.array(X)
    y = np.array(y)

    print "NUM #singles: {} #not singles: {}".format(np.sum(y), len(y) - np.sum(y))

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    clf = svm.SVC(kernel='poly', degree=3, probability=True)
    clf.fit(X, y)

    class_member_mask = (labels == -1)
    X2 = data[class_member_mask]
    region_ids = arr[class_member_mask]

    X2 = scaler.transform(X2)
    probs = clf.predict_proba(X2)

    labels = probs[:, 1] > 0.99
    print len(labels), np.sum(labels)

    display_classification(p, region_ids, labels)

def solve_nearby_passings(p):
    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/movement_data.pkl', 'rb') as f:
        data = pickle.load(f)

    data = np.array(data)

    H, edges = np.histogramdd(data, bins=10)
    data = data[::5, :]

    DEFAULT_H_DENSITY = 1e-10

    cases_p = []
    cases_n = []

    for it, case in izip(data, cases):
        it = np.array(it)

        # if np.linalg.norm(it[0, :] - it[1, :]) > 60:
        #     continue

        vals = []
        for i in range(it.shape[0]):
            val = hist_query(H, edges, it[i, :], default=DEFAULT_H_DENSITY)
            vals.append(val)

        if max(vals) > DEFAULT_H_DENSITY:
            cases_p.append((vals, case))
            # print case
            # print vals
        else:
            cases_n.append((vals, case))
            continue

        ax.plot(it[:, 1], it[:, 2], it[:, 0], marker='v', c='m')

def add_single_chunks(p, frames):
    p.chm.reset_itree(p.gm)

    for n in p.gm.g.vertices():
        r = p.gm.region(n)
        if r.frame() not in frames or p.gm.get_chunk(n) is not None:
            continue

        if not p.gm.g.vp['active'][n]:
            continue

        p.chm.new_chunk([int(n)], p.gm)

    p.chm.reset_itree(p.gm)

    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_tracklets_expanded.pkl', 'wb') as f:
        pic = pickle.Pickler(f)
        pic.dump(p.gm.g)
        pic.dump([])
        pic.dump(p.chm)

def build_tracklets_from_others(p):
    # with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/movement_data.pkl', 'rb') as f:
    #     data = pickle.load(f)

    for v in p.gm.g.vertices():
        if p.gm.get_chunk(v) is None:
            if p.gm.one2one_check(v):
                v2 = p.gm.out_v(v)

                if p.gm.get_chunk(v2) is None:
                    p.chm.new_chunk([int(v), int(v2)], p.gm)

    p.chm.reset_itree(p.gm)

    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_tracklets_expanded.pkl', 'wb') as f:
        pic = pickle.Pickler(f)
        pic.dump(p.gm.g)
        pic.dump([])
        pic.dump(p.chm)


def assign_costs(p, frames):
    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_modified.pkl', 'rb') as f:
        g = pickle.load(f)

    p.gm.g = g


    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/movement_data.pkl', 'rb') as f:
        data = pickle.load(f)

    data = np.array(data)

    H, edges = np.histogramdd(data, bins=10)

    for v in p.gm.g.vertices():
        r = p.gm.region(v)
        if r.frame() not in frames:
            continue

        for v2 in v.out_neighbours():
            for e in v.out_edges():
                v3 = e.target()
                val = hist_query(H, edges, get_movement_descriptor(p, v, v2, v3))
                old_val = p.gm.g.ep['score'][e]
                new_val = max(val + 1.001, old_val)
                p.gm.g.ep['score'][e] = new_val

    # with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_modified.pkl', 'wb') as f:
    #     pickle.dump(p.gm.g, f)


def display_n_representatives(p, label=0, N=30):
    d = hickle.load('/Users/flipajs/Desktop/temp/clustering/labels.pkl')

    labels = d['labels']
    arr = np.array(d['arr'])
    data = d['data']

    scaler = StandardScaler()

    X = scaler.fit_transform(data)

    X_ = X[labels==label,:]
    arr = arr[labels==label]

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=N, random_state=0).fit(X_)

    labels = kmeans.labels_
    data = []

    vm = get_auto_video_manager(p)

    for k in range(N):
        class_member_mask = (labels == k)
        a_ = arr[class_member_mask]

        v1 = a_[0]

        r1 = p.gm.region(v1)

        im1 = vm.get_frame(r1.frame()).copy()

        draw_points(im1, r1.pts())

        im = im1[r1.roi().slices()].copy()
        data.append(im)

    collage = create_collage_rows(data, 7, 100, 100)
    cv2.imshow('collage', collage)
    cv2.waitKey(0)

    # with open('/Users/flipajs/Desktop/temp/clustering/sub_0_labels.pkl', 'wb') as f:
    #     pickle.dump({'data': X_, 'labels': kmeans.labels_, 'arr': arr}, f)


if __name__ == '__main__':
    p = Project()
    p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_playground')

    # display_n_representatives(p, label=-1, N=50)

    if True:

        from core.region.region_manager import RegionManager
        p.rm = RegionManager('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp', db_name='part0_rm.sqlite3')
        with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0.pkl', 'rb') as f:
            up = pickle.Unpickler(f)
            g_ = up.load()
            # relevant_vertices = up.load()
            # chm_ = up.load()

        p.gm.g = g_
        p.gm.rm = p.rm

        # prepare_pairs(p)
        # display_head_pairs(p)

        # head_detector_features(p)
        # head_detector_classify(p)

        max_dist = 94.59
        # max_dist = get_max_dist(p)
        print "MAX DIST: {}".format(max_dist)

        # filter_edges(p, max_dist)

        # get_assignment_histogram(p)

        # get_movement_histogram(p)
        # observe_cases(p)
        # observe_cases(p, type='case_n')

        # singles_classifier(p)

        assign_costs(p, set(range(1000)))
        from core.graph.solver import Solver
        from core.graph.chunk_manager import ChunkManager
        solver = Solver(p)
        p.chm = ChunkManager()

        THRESH = 100.

        confirm_later = []
        for v in p.gm.g.vertices():
            if v.out_degree() == 0:
                continue

            r = p.gm.region(v)
            if r.frame() > 1000:
                continue

            if p.gm.one2one_check(v):
                e = p.gm.out_e(v)
                confirm_later.append((e.source(), e.target()))
            else:
                pairs = []
                for e in v.out_edges():
                    pairs.append((e, p.gm.g.ep['score'][e]))

                pairs = sorted(pairs, key=lambda x: -x[1])
                if len(pairs) == 1 or pairs[0][1] / pairs[1][1] > THRESH:
                    best_s = 1
                    for e in pairs[0][0].target().in_edges():
                        if e == pairs[0][0]:
                            continue

                        s = p.gm.g.ep['score'][e]
                        if s > best_s:
                            best_s = s

                    if pairs[0][1] / best_s > THRESH:
                        confirm_later.append((pairs[0][0].source(), pairs[0][0].target()))

        solver.confirm_edges(confirm_later)

        p.chm.reset_itree(p.gm)

        with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_tracklets_expanded.pkl', 'wb') as f:
            pic = pickle.Pickler(f)
            pic.dump(p.gm.g)
            pic.dump([])
            pic.dump(p.chm)


        # simple_tracklets(p)
        # solve_nearby_passings(p)

        # with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_tracklets.pkl', 'rb') as f:
        #     up = pickle.Unpickler(f)
        #     g = up.load()
        #     _ = up.load()
        #     chm = up.load()
        #
        # p.chm = chm
        # p.gm.g = g
        # p.gm.update_nodes_in_t_refs()
        #
        # lengths = np.array([t.length() for t in p.chm.chunk_gen()])
        #
        # print "#chunks: {}".format(len(p.chm))
        # print "LENGTHS mean: {} median: {}, max: {}, sum: {} coverage: {:.2%}".format(np.mean(lengths), np.median(lengths),
        #                                                                               lengths.max(), np.sum(lengths), np.sum(lengths)/(4500*6.0))
        #
        # # get_movement_histogram(p)
        #
        # # expand_based_on_movement_model(p)
        # #
        # # lengths = np.array([t.length() for t in p.chm.chunk_gen()])
        # #
        # # print "#chunks: {}".format(len(p.chm))
        # # print "LENGTHS mean: {} median: {}, max: {}, sum: {} coverage: {:.2%}".format(np.mean(lengths), np.median(lengths),
        # #                                                                               lengths.max(), np.sum(lengths),
        # #                                                                               np.sum(lengths) / (4500 * 6.0))
        #
        # expand_based_on_movement_model(p)
        # add_single_chunks(p, set(range(100)))

        p.chm.reset_itree(p.gm)
        # TODO: hack... no check for 1 to 1 assignment
        # build_tracklets_from_others(p)


        lengths = np.array([t.length() for t in p.chm.chunk_gen()])

        print "#chunks: {}".format(len(p.chm))
        print "LENGTHS mean: {} median: {}, max: {}, sum: {} coverage: {:.2%}".format(np.mean(lengths), np.median(lengths), lengths.max(), np.sum(lengths), np.sum(lengths)/(4500*6.0))
        # plt.hist(lengths, bins=500)
        # plt.show()


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
            hickle.dump({'arr': arr, 'labels': labels, 'data': data}, '/Users/flipajs/Desktop/temp/clustering/labels.pkl')
                # pickle.dump({'data': data, 'arr': arr, 'labels': labels, 'core_samples_mask': core_samples_mask}, f)

        # display_regions(p, arr, labels)
        # plt.show()