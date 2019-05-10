# from __future__ import print_function
import cPickle as pickle
from itertools import izip

import cv2
from libs.hickle import hickle
# import hickle
import numpy as np
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from core.config import config
from core.graph.chunk_manager import ChunkManager
from core.graph.region_chunk import RegionChunk
from core.graph.solver import Solver
from core.id_detection.features import get_hog_features, get_crop
from core.project.project import Project
# from core.region.clustering import prepare_region_cardinality_samples
from utils.drawing.collage import create_collage_rows
from utils.drawing.points import draw_points
from tqdm import tqdm
from utils.video_manager import get_auto_video_manager
import logging

logger = logging.getLogger(__name__)


EXP = 'exp1'
DEFAULT_H_DENSITY = 1e-10


def plotNdto3d(data, labels, core_samples_mask, indices=[0, 1, 2], ax_labels=['', '', ''], title=''):
    import matplotlib.pyplot as plt
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


def display_pairs(p, pairs, file_name, cols=7, item_height=100, item_width=200, border=20):
    vm = get_auto_video_manager(p)

    part = 0
    i = 0
    data = []
    for r1, r2 in pairs:
        if r1.frame() + 1 != r2.frame():
            print("FRAMES? ", r1.frame(), r2.frame())

        im1 = vm.get_frame(r1.frame()).copy()
        im2 = vm.get_frame(r2.frame()).copy()

        draw_points(im1, r1.pts())
        draw_points(im2, r2.pts())
        draw_points(im2, r1.pts(), color=(0, 255, 255, 40))

        im1 = r1.roi().safe_roi(im1, border=border)
        cv2.putText(im1, str(r1.frame()), (10, 10),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, (255, 255, 255))

        im2 = r1.roi().safe_roi(im2, border=border)

        im = np.hstack((im1, im2))
        data.append(im)

        if i % (100) == 0 and i > 0:
            collage = create_collage_rows(data, cols, item_height, item_width)
            cv2.imwrite(p.working_directory+'/temp/'+file_name + str(part) + '.jpg', collage)
            data = []
            part += 1

        i += 1

    collage = create_collage_rows(data, cols, item_height, item_width)
    cv2.imwrite(p.working_directory+'/temp/'+ file_name + str(part) + '.jpg', collage)



def display_head_pairs(project):
    import matplotlib.pyplot as plt
    print ("displaying pairs...")
    pairs = hickle.load('/Users/flipajs/Desktop/temp/pairs/pairs.pkl')
    print ("loaded..")
    from utils.video_manager import get_auto_video_manager

    BORDER = 150
    COLS = 7
    IT_H = 500
    IT_W = 700

    D2_COEF = 5

    vm = get_auto_video_manager(project)

    major_axes = [project.gm.region(x[0][0]).ellipse_major_axis_length() for x in pairs]
    major_axes_mean = np.mean(major_axes)

    print ("major axes mean", major_axes_mean)

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

    print ("NEW LEN", len(pairs))

    i = 0
    part = 0
    data = []
    region_pairs = []
    for ((v1, v2), d1, d2) in pairs:
        if v1 is None or v2 is None:
            continue

        r1 = project.gm.region(v1)
        r2 = project.gm.region(v2)

        region_pairs.append((r1, r2))

    display_pairs(region_pairs)


def display_clustering_results(project, vertices=None, labels=None, cols=15, it_w=100, it_h=100, max_rows=100):
    vm = get_auto_video_manager(project)

    if vertices is None:
        with open(project.working_directory+'/temp/region_cardinality_samples.pkl') as f:
            up = pickle.Unpickler(f)
            _ = up.load()
            vertices = up.load()
            vertices = np.array(vertices)
            labels = up.load()

    unique_labels = set(labels)
    for class_, k in enumerate(unique_labels):
        class_member_mask = (labels == k)
        a_ = vertices[class_member_mask]

        data = []
        part = 0
        for i, v1 in enumerate(a_):
            if v1 is None:
                continue

            r1 = project.gm.region(v1)

            im1 = vm.get_frame(r1.frame()).copy()

            draw_points(im1, r1.pts())

            im = im1[r1.roi().slices()].copy()
            data.append(im)

            if (i + 1) % (cols*max_rows) == 0:
                collage = create_collage_rows(data, cols, it_h, it_w)
                cv2.imwrite(project.working_directory+'/temp/clustering_' + str(class_) + '_' + str(part)+ '.jpg', collage)
                part += 1
                data = []

        collage = create_collage_rows(data, cols, it_h, it_w)
        cv2.imwrite(project.working_directory+'/temp/clustering_' + str(class_) + '_' + str(part) + '.jpg', collage)

def prepare_pairs(project):
    print ("__________________________")
    print ("preparing pairs...")
    with open(project.working_directory+'/temp/region_cardinality_samples.pkl') as f:
        up = pickle.Unpickler(f)
        _ = up.load()
        vertices = up.load()
        labels = up.load()

    vs = set(vertices[labels == 0])

    pairs = []
    filtered_v = vertices[labels == 0]
    v_num = len(filtered_v)
    i = 0
    for v in tqdm(filtered_v):
        r1 = project.gm.region(v)
        best_v = None
        best_d = np.inf
        second_best_d = np.inf

        for v_out in project.gm.g.vertex(v).out_neighbors():
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

        i += 1

    print ("saving...")
    with open(project.working_directory+'/temp/pairs.pkl', 'wb') as f:
        pickle.dump(pairs, f, -1)
    print ("---------------------------------")


def __get_mu_moments_pick(img):
    from core.id_detection.features import get_mu_moments
    nu = get_mu_moments(img)

    return list(nu[np.logical_not(np.isnan(nu))])

def head_features(r, swap=False):
    # normalize...
    from utils.geometry import rotate
    from utils.drawing.points import draw_points_crop_binary

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

        f1_, f2_ = get_hog_features(r, p, fliplr=True)
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
    f.extend(get_hog_features(r, p))

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
        f.extend(get_hog_features(r, p))

        probs = rfc.predict_proba(np.array([f]))[0]

        if probs[1] > 0.5:
            r.theta_ += np.pi
            if r.theta_ > 2 * np.pi:
                r.theta_ -= 2 * np.pi



def head_detector_classify(p):
    data_head = hickle.load('/Users/flipajs/Desktop/temp/pairs/'+EXP+'/head_data.pkl')
    data_swap = hickle.load('/Users/flipajs/Desktop/temp/pairs/'+EXP+'/head_data_swap.pkl')
    rfc = RandomForestClassifier()

    X = np.vstack((np.array(data_head), np.array(data_swap)))
    y = np.hstack((np.zeros((len(data_head), ), dtype=np.int), np.ones((len(data_swap), ), dtype=np.int)))
    rfc.fit(X, y)

    with open('/Users/flipajs/Desktop/temp/pairs/'+EXP+'/head_rfc.pkl', 'wb') as f:
        pickle.dump(rfc, f)

    return

    print (rfc.feature_importances_)

    d = hickle.load('/Users/flipajs/Desktop/temp/prepare_region_cardinality_samples/labels.pkl')
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
        f.extend(get_hog_features(r, p))

        probs = rfc.predict_proba(np.array([f]))[0]

        if probs[1] > 0.5:
            r.theta_ += np.pi
            if r.theta_ > 2*np.pi:
                r.theta_ -= 2*np.pi

        crop = get_crop(r, p, margin=10)
        # bb, offset = get_bounding_box(r, p)
        # bb = rotate_img(bb, r.theta_)
        # bb = centered_crop(bb, 8 * r.ellipse_minor_axis_length(), 4 * r.ellipse_major_axis_length())
        # crop_ = bb

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
#     d = hickle.load('/Users/flipajs/Desktop/temp/prepare_region_cardinality_samples/labels.pkl')
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
#         for v_out in project.gm.g.vertex(v).out_neighbors():
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
    print ("avg degree before {}".format(np.mean([v.out_degree() for v in g.vertices()])))

    for (v1, v2) in g.edges():
        r1 = project.gm.region(v1)
        r2 = project.gm.region(v2)

        if r1.is_ignorable(r2, max_dist):
            to_remove.append((v1, v2))

    print ("#edges: {}, will be removed: {}".format(g.num_edges(), len(to_remove)))
    for (v1, v2) in to_remove:
        g.remove_edge(g.edge(v1, v2))

    degrees = [v.out_degree() for v in g.vertices()]
    print ("avg degree after {}".format(np.mean(degrees)))

    # plt.hist(degrees)
    # plt.show()
    #

    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_modified.pkl', 'wb') as f:
        pickle.dump(g, f)


def get_max_dist(project):
    print ("____________________________")
    print ("Estimating max distance")
    with open(project.working_directory+'/temp/pairs.pkl') as f:
        pairs = pickle.load(f)

    max_dist = 0
    max_v1 = None
    max_v2 = None

    num_pairs = len(pairs)
    i = 0
    for (v1, v2), d1, _ in tqdm(pairs):
        if d1 > max_dist:
            max_dist = d1
            max_v1 = v1
            max_v2 = v2

    print ()
    r1 = project.gm.region(max_v1)
    r2 = project.gm.region(max_v2)

    if r1.frame() + 1 != r2.frame():
        print ("FRAMES? ", r1.frame(), r2.frame())

    vm = get_auto_video_manager(project)

    im1 = vm.get_frame(r1.frame()).copy()
    im2 = vm.get_frame(r2.frame()).copy()

    draw_points(im2, r2.pts())
    draw_points(im2, r1.pts(), color=(0, 255, 255, 40))

    print ("MAX DIST: {:.1f}".format(max_dist))
    cv2.imshow('max distance visualisation', im2)
    cv2.imshow('im1', im1)
    cv2.waitKey(5)

    print ("-----------------------------")
    return max_dist

def get_max_dist2(project):
    print ("____________________________")
    print ("Estimating max distance")

    reg = project.gm.region

    safe_dists = []
    pairs = []

    num_v = project.gm.g.num_vertices()
    for v in tqdm(project.gm.g.vertices(), total=project.gm.g.num_vertices()):
        if v.out_degree() < 2:
            continue

        best_e, _ = project.gm.get_2_best_out_edges(v)

        distances = []
        for e in best_e:
            d = np.linalg.norm(reg(e.source()).centroid() - reg(e.target()).centroid())
            distances.append(d)

        if distances[1] / distances[0] > 2:
            safe_dists.append(distances[0])
            pairs.append((best_e[0].source(), best_e[0].target()))

    print ()
    id_ = np.argmax(safe_dists)
    max_dist = safe_dists[id_]
    max_v1, max_v2 = pairs[id_]

    r1 = project.gm.region(max_v1)
    r2 = project.gm.region(max_v2)

    if r1.frame() + 1 != r2.frame():
        print ("FRAMES? ", r1.frame(), r2.frame())

    vm = get_auto_video_manager(project)

    im1 = vm.get_frame(r1.frame()).copy()
    im2 = vm.get_frame(r2.frame()).copy()

    draw_points(im2, r2.pts())
    draw_points(im2, r1.pts(), color=(0, 255, 255, 40))

    print ("MAX DIST: {:.1f}".format(max_dist))
    cv2.imshow('max distance visualisation', im2)
    cv2.imshow('im1', im1)
    cv2.waitKey(5)

    print ("-----------------------------")
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
    import matplotlib.pyplot as plt
    # with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_modified.pkl', 'rb') as f:
    #     g = pickle.load(f)
    #     _ = pickle.load(f)
    #     chm = pickle.load(f)

    # p.gm.g = g

    d = hickle.load('/Users/flipajs/Desktop/temp/prepare_region_cardinality_samples/labels.pkl')
    labels = d['labels']
    arr = d['arr']

    data = []
    data2 = []
    cases = []

    # for v in arr[labels==0]:
    #     v = p.gm.g.vertex(v)
    #
    #     if v.out_degree() == 1:
    #         for w in v.out_neighbors():
    #             if w.in_degree() == 1 and w.out_degree() == 1:
    #                 for x in w.out_neighbors():
    #                     if x.in_degree() == 1:
    #                         data.append(get_movement_descriptor(p, v, w, x))
    #             elif w.in_degree() == 1 and w.out_degree() > 1:
    #                 data2.append([])
    #                 cases.append([])
    #                 for x in w.out_neighbors():
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
            c = (0, 255, 0, 70)
            if val <= 1e-10:
                c = (255, 0, 0, 70)
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
            # draw_points(im2, r1.pts(), color=(0, 255, 255, 40))

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
                v = t.start_vertex()
                if v.in_degree() > 0:
                    options = []

                    for v2 in v.in_neighbors():
                        val = hist_query(H, edges, get_movement_descriptor(p, v2, t[0], t[1]))
                        options.append((val + 1, v2))

                    options = sorted(options, key=lambda x: -x[0])

                    if len(options) > 1:
                        ratio = options[0][0] / options[1][0]
                    else:
                        ratio = options[0][0]

                    if ratio > THRESH:
                        v2 = options[0][1]
                        t.append_left(v2)
                    else:
                        break
                else:
                    break

            while True:
                v = t.end_vertex()
                if v.out_degree() > 0:
                    options = []

                    for v2 in v.out_neighbors():
                        val = hist_query(H, edges, get_movement_descriptor(p, t[-2], t[-1], v2))
                        options.append((val + 1, v2))

                    options = sorted(options, key=lambda x: -x[0])

                    if len(options) > 1:
                        ratio = options[0][0] / options[1][0]
                    else:
                        ratio = options[0][0]

                    if ratio > THRESH:
                        v2 = options[0][1]
                        t.append_right(v2)
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

    d = hickle.load('/Users/flipajs/Desktop/temp/prepare_region_cardinality_samples/labels.pkl')
    labels = d['labels']
    vertices_ids = np.array(d['arr'])

    from core.graph.chunk_manager import ChunkManager
    p.chm = ChunkManager()


    singles_ids = list(vertices_ids[labels==0])

    print ("BEFORE:")
    print ("#vertices: {} #edges: {}".format(p.gm.g.num_vertices(), p.gm.g.num_edges()))

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
                for v2 in v_.in_neighbors():
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
                for v2 in v_.out_neighbors():
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

    print ("BEFORE:")
    print ("#vertices: {} #edges: {}".format(p.gm.g.num_vertices(), p.gm.g.num_edges()))
    print ("#chunks: {}".format(len(p.chm)))

    for ch in p.chm.chunk_gen():
        if ch.length() == 1:
            print ch

    with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_tracklets.pkl', 'wb') as f:
        pic = pickle.Pickler(f)
        pic.dump(p.gm.g)
        pic.dump([])
        pic.dump(p.chm)


def display_classification(project, ids, labels):
    print ("display regions")
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
                print (i)

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

                print ("TEST")

        collage = create_collage_rows(data, COLS, IT_H, IT_W)
        cv2.imwrite('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/' + F_NAME + str(class_) + '_' + str(part) + '.jpg', collage)


def singles_classifier(p):
    d = hickle.load('/Users/flipajs/Desktop/temp/prepare_region_cardinality_samples/labels.pkl')
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

    print ("NUM #singles: {} #not singles: {}".format(np.sum(y), len(y) - np.sum(y)))

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
    print (len(labels), np.sum(labels))

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
    p.chm.reset_itree()

    for n in p.gm.g.vertices():
        r = p.gm.region(n)
        if r.frame() not in frames or p.gm.get_chunk(n) is not None:
            continue

        if not p.gm.g.vp['active'][n]:
            continue

        p.chm.new_chunk([int(n)], p.gm)

    p.chm.reset_itree()

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

    p.chm.reset_itree()

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

        for v2 in v.out_neighbors():
            for e in v.out_edges():
                v3 = e.target()
                val = hist_query(H, edges, get_movement_descriptor(p, v, v2, v3))
                old_val = p.gm.g.ep['score'][e]
                new_val = max(val + 1.001, old_val)
                p.gm.g.ep['score'][e] = new_val

    # with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/part0_modified.pkl', 'wb') as f:
    #     pickle.dump(p.gm.g, f)


def decide_one2one(p):
    solver = Solver(p)

    confirm_later = []

    for v in p.gm.g.vertices():
        if p.gm.one2one_check(v):
            e = p.gm.out_e(v)
            confirm_later.append((e.source(), e.target()))

    solver.confirm_edges(confirm_later)

    p.gm.update_nodes_in_t_refs()
    p.chm.reset_itree()


def print_tracklet_stats(p):
    lengths = np.array([t.length() for t in p.chm.chunk_gen()])

    logger.debug("#chunks: {}".format(len(p.chm)))
    logger.debug(
    "LENGTHS mean: {:.1f} median: {}, max: {}, sum: {} coverage: {:.2%}".format(np.mean(lengths), np.median(lengths),
                                                                                lengths.max(), np.sum(lengths),
                                                                                np.sum(lengths) / float(
                                                                                    (p.gm.end_t - p.gm.start_t) * len(
                                                                                        p.animals))))


def load_p_checkpoint(p, name=''):
    with open(p.working_directory+'/temp/'+name+'.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        p.gm.g = up.load()
        up.load()
        p.chm = up.load()
        try:
            p.gm.vertices_in_t = up.load()
        except:
            print "vertices_in_t not loaded..."


def save_p_checkpoint(p, name=''):
    with open(p.working_directory+'/temp/'+name+'.pkl', 'wb') as f:
        pic = pickle.Pickler(f)
        pic.dump(p.gm.g)
        pic.dump(None)
        pic.dump(p.chm)
        pic.dump(p.gm.vertices_in_t)


def get_pair_fetures_appearance(r1, r2):
    # f = [
    #     r1.area() - r2.area(),
    #     r1.area() / float(r2.area()),
    #     r1.ellipse_major_axis_length() - r2.ellipse_major_axis_length(),
    #     r1.ellipse_major_axis_length() / r2.ellipse_major_axis_length(),
    #     (r1.ellipse_major_axis_length()/r1.ellipse_minor_axis_length()) / (r2.ellipse_major_axis_length()/r2.ellipse_minor_axis_length()),
    #     r1.eccentricity() - r2.eccentricity(),
    #     r1.sxx_ - r2.sxx_,
    #     r1.syy_ - r2.syy_,
    #     r1.sxy_ - r2.sxy_,
    #     int(r1.min_intensity_) - int(r2.min_intensity_),
    # ]

    f = [
        abs(r1.area() - r2.area()),
        r1.area() / float(r2.area()),
        r1.ellipse_major_axis_length() - r2.ellipse_major_axis_length(),
        r1.ellipse_major_axis_length() / r2.ellipse_major_axis_length(),
        (r1.ellipse_major_axis_length() / r1.ellipse_minor_axis_length()) / (r2.ellipse_major_axis_length() / r2.ellipse_minor_axis_length()),
        # r1.eccentricity() - r2.eccentricity(),
        # r1.sxx_ - r2.sxx_,
        # r1.syy_ - r2.syy_,
        # r1.sxy_ - r2.sxy_,
        int(r1.min_intensity_) - int(r2.min_intensity_),
    ]

    return f


def get_pair_fetures_movement(r1, r2):
    theta_diff = abs(r1.theta_ - r2.theta_)
    f = [
        np.linalg.norm(r1.centroid() - r2.centroid()),
        # A,
        # to deal with head orientation uncertainty
        min(theta_diff, np.pi - theta_diff),
        r1.get_phi(r2),
        # TODO: dist to prediction if present?
    ]

    return f


def learn_assignments(p, max_examples=np.inf, display=False):
    """

    :param p: project
    :param max_examples:
    :param display:
    :return: movement, appearance - isolation forests
    """
    X_appearance = []
    X_movement = []

    # TODO: remove
    chgen = p.chm.chunk_gen()
    # chgen.next()

    i = 0
    j = 0

    pairs = []
    for t in chgen:
        # TODO: remove
        # if i > 500:
        #     break

        i += 1

        rch = RegionChunk(t, p.gm, p.rm)
        gen = rch.regions_gen()
        r1 = gen.next()

        for r2 in gen:
            j += 1

            X_appearance.append(get_pair_fetures_appearance(r1, r2))
            X_movement.append(get_pair_fetures_movement(r1, r2))
            pairs.append((r1, r2))
            r1 = r2

        if j > max_examples:
            break

    # TODO: I think contamination doesn't matter...
    IF_appearance = IsolationForest(contamination=0.005)
    IF_appearance.fit(X_appearance)

    IF_movement = IsolationForest(contamination=0.005)
    IF_movement.fit(X_movement)

    if display:
        y = IF_appearance.predict(X_appearance)
        print len(y), np.sum(y == -1)
        pairs = np.array(pairs)

        display_pairs(p, pairs[y == -1], 'anomaly_parts_appearance', cols=3, item_height=250, item_width=500, border=70)

        y = IF_movement.predict(X_movement)
        print len(y), np.sum(y == -1)
        pairs = np.array(pairs)

        display_pairs(p, pairs[y == -1], 'anomaly_parts_movement', cols=3, item_height=250, item_width=500, border=70)

    return IF_movement, IF_appearance


def add_score_to_edges(gm, IF_movement, IF_appearance):
    print "#edges: {}".format(gm.g.num_edges())
    i = 0

    use_for_learning = 0.1

    features_appearance = []
    features_movement = []
    edges = []

    for e in tqdm(gm.g.edges(), total=gm.g.num_edges(), desc='adding score to edges'):
        i += 1
        if gm.edge_is_chunk(e):
            continue

        f = get_pair_fetures_appearance(gm.region(e.source()), gm.region(e.target()))
        features_appearance.append(f)

        f = get_pair_fetures_movement(gm.region(e.source()), gm.region(e.target()))
        features_movement.append(f)
        
        edges.append(e)

    print "computing isolation score..."
    vals_appearance = IF_appearance.decision_function(features_appearance)
    vals_movement = IF_movement.decision_function(features_movement)

    from sklearn.linear_model import LogisticRegression

    for type in ['appearance', 'movement']:
        if type == 'appearance':
            vals = vals_appearance
        else:
            vals = vals_movement

        lr = LogisticRegression()
        part_len = int(len(vals)*use_for_learning)

        vals_sorted = sorted(vals)
        part1 = np.array(vals_sorted[:part_len])
        part2 = np.array(vals_sorted[-part_len:])

        X = np.hstack((part1, part2))
        X.shape = ((X.shape[0], 1))

        y = np.array([1 if i < len(part1) else 0 for i in range(X.shape[0])])
        lr.fit(X, y)
        probs = lr.predict_proba(np.array(vals).reshape((len(vals), 1)))

        print "assigning score to edges.."
        for val, e in izip(probs[:, 0], edges):
            if type == 'appearance':
                gm.g.ep['score'][e] = val
            else:
                gm.g.ep['movement_score'][e] = val

        print "saving..."

    # save_p_checkpoint(p, 'isolation_score')


def process_project(p):
    from core.graph.solver2 import Solver2
    solver2 = Solver2(p)

    # prepare_region_cardinality_samples(p, compute_data=False)
    # display_clustering_results(p)
    # display_cluster_representants(p)

    # prepare_pairs(p)
    # max_dist = get_max_dist2(p)

    # max_dist = 80.
    # with open(p.working_directory+'/temp/data.pkl', 'wb') as f:
    #     pickle.dump({'max_measured_distance': max_dist}, f)
    #
    # solver2.prune_distant_connections(max_dist)
    # save_p_checkpoint(p, 'g_pruned')
    #
    # load_p_checkpoint(p, 'g_pruned')
    # p.chm = ChunkManager()
    # p.gm.update_nodes_in_t_refs()
    # decide_one2one(p)
    # tracklet_stats(p)
    # save_p_checkpoint(p, 'first_tracklets')
    #
    # learn_assignments(p)
    #
    load_p_checkpoint(p, 'first_tracklets')
    #
    p.gm.g.ep['movement_score'] = p.gm.g.new_edge_property("float")

    add_score_to_edges(p)

    save_p_checkpoint(p, 'edge_cost_updated')
    load_p_checkpoint(p, 'edge_cost_updated')

    p.gm.update_nodes_in_t_refs()
    p.chm.reset_itree()
    #
    solver = Solver(p)

    print_tracklet_stats(p)

    if False:
        score_type = 'appearance_motion_mix'
        eps = 0.3

        strongly_better_e = p.gm.strongly_better_eps(eps=eps, score_type=score_type)
        print "strongly better: {}".format(len(strongly_better_e))
        for e in strongly_better_e:
            solver.confirm_edges([(e.source(), e.target())])

        print_tracklet_stats(p)

        strongly_better_e = p.gm.strongly_better_eps(eps=eps, score_type=score_type)
        print "strongly better: {}".format(len(strongly_better_e))
        for e in strongly_better_e:
            solver.confirm_edges([(e.source(), e.target())])

        print_tracklet_stats(p)
        decide_one2one(p)

        p.gm.update_nodes_in_t_refs()
        p.chm.reset_itree()

        save_p_checkpoint(p, 'eps_edge_filter')
        print_tracklet_stats(p)
    # else:
        # from utils.geometry import get_region_group_overlaps
        #
        # for i in range(0, 1000, 10):
        #     print i
        #     rt1 = p.gm.regions_in_t(i)
        #     rt2 = p.gm.regions_in_t(i+1)
        #
        #     get_region_group_overlaps(rt1, rt2)
        # # overlap test...
        # # boolean matrix regions_t x regions_t+1
        # pass


if __name__ == '__main__':
    p = Project()
    # p.load('/Users/flipajs/Documents/wd/FERDA/zebrafish_playground')
    # p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_playground')
    p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_rf')
    # p.load('/Users/flipajs/Documents/wd/FERDA/Sowbug3')
    p.load('/Users/flipajs/Documents/wd/FERDA/Camera3')
    from core.region.region_manager import RegionManager

    p.rm = RegionManager(p.working_directory + '/temp', db_name='part0_rm.sqlite3')
    with open(p.working_directory + '/temp/part0.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        g_ = up.load()

    p.gm.g = g_
    p.gm.rm = p.rm

    process_project(p)

    if False:
        FILTER_EDGES = False
        DO_DECIDEONE2ONE = False
        LEARN_ASSIGNMENTS = False

        FILTER_EDGES2 = False

        # p = Project()
        # p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_playground')

        # display_cluster_representants(p, label=-1, N=50)


        max_dist = 94.59
        # max_dist = get_max_dist(p)
        print "MAX DIST: {}".format(max_dist)

        if False:
            if FILTER_EDGES:
                filter_edges(p, max_dist)
            else:
                with open(p.working_directory+'/temp/part0_modified.pkl', 'rb') as f:
                    p.gm.g = pickle.load(f)

                p.gm.update_nodes_in_t_refs()

            if DO_DECIDEONE2ONE:
                p.chm = ChunkManager()
                decide_one2one(p)
                save_p_checkpoint(p, 'part0_1to1_tracklets')
                tracklet_stats(p)
            else:
                load_p_checkpoint(p, name='part0_1to1_tracklets')

            if LEARN_ASSIGNMENTS:
                learn_assignments(p)

            p.gm.g.ep['movement_score'] = p.gm.g.new_edge_property("float")

            add_score_to_edges(p)
        elif False:
            load_p_checkpoint(p, 'isolation_score')
            p.gm.update_nodes_in_t_refs()

            solver = Solver(p)

            edges = p.gm.edges_with_score_in_range(lower_bound=0.1)
            solver.confirm_edges([(e.source(), e.target()) for e in edges])
            tracklet_stats(p)
            decide_one2one(p)

            edges = p.gm.edges_with_score_in_range(upper_bound=-0.1)
            p.gm.remove_edges(edges)

            # save_p_checkpoint(p, 'removed_edges')

            tracklet_stats(p)
            decide_one2one(p)
            save_p_checkpoint(p, 'part0_1to1_tracklets2')
            tracklet_stats(p)
        else:
            load_p_checkpoint(p, 'isolation_score')
            p.gm.update_nodes_in_t_refs()
            p.chm.reset_itree()

            # TODO: deal with noise...
            d = hickle.load('/Users/flipajs/Desktop/temp/prepare_region_cardinality_samples/labels.pkl')
            labels = d['labels']
            vertices = d['arr']

            for l in [1, 2, 3]:
                for v in vertices[labels == l]:
                    p.gm.remove_vertex(v, disassembly=False)


            solver = Solver(p)

            tracklet_stats(p)

            score_type = 'appearance_motion_mix'

            min_prob = 0.5**2
            min_prob = 0.1
            better_n_times = 50

            strongly_better_e = p.gm.strongly_better(min_prob=min_prob, better_n_times=better_n_times, score_type=score_type)
            print("strongly better: {}".format(len(strongly_better_e)))
            for e in strongly_better_e:
                solver.confirm_edges([(e.source(), e.target())])

            tracklet_stats(p)

            strongly_better_e = p.gm.strongly_better(min_prob=min_prob, better_n_times=better_n_times, score_type=score_type)
            print("strongly better: {}".format(len(strongly_better_e)))
            for e in strongly_better_e:
                solver.confirm_edges([(e.source(), e.target())])

            tracklet_stats(p)
            strongly_better_e = p.gm.strongly_better(min_prob=min_prob, better_n_times=better_n_times, score_type=score_type)
            print("strongly better: {}".format(len(strongly_better_e)))
            for e in strongly_better_e:
                solver.confirm_edges([(e.source(), e.target())])

            tracklet_stats(p)
            decide_one2one(p)

            # p.gm.update_nodes_in_t_refs()
            p.chm.reset_itree()

            save_p_checkpoint(p, 'strongly_better_filter')
            tracklet_stats(p)


        if False:
            # prepare_pairs(p)
            # display_head_pairs(p)

            # head_detector_features(p)
            # head_detector_classify(p)

            # filter_edges(p, max_dist)

            # get_assignment_histogram(p)

            # get_movement_histogram(p)
            # observe_cases(p)
            # observe_cases(p, type='case_n')

            # singles_classifier(p)

            assign_costs(p, set(range(1000)))
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

            with open(p.working_directory+'/temp/part0_tracklets_expanded.pkl', 'wb') as f:
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

            p.chm.reset_itree()
            # TODO: hack... no check for 1 to 1 assignment
            # build_tracklets_from_others(p)


            tracklet_stats(p)

            # plt.hist(lengths, bins=500)
            # plt.show()

            if False:
                prepare_region_cardinality_samples()
            # display_clustering_results(p, arr, labels)
            # plt.show()