import cPickle as pickle

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from core.id_detection.features import get_hu_moments
from utils.misc import print_progress
from utils.drawing.points import draw_points
from utils.video_manager import get_auto_video_manager
from utils.drawing.collage import create_collage_rows
from scipy.spatial.distance import cdist
import cv2

def clustering(p, compute_data=True):
    print "___________________________________"
    print "Preparing data for clustering..."

    i = 0

    if not compute_data:
        try:
            with open(p.working_directory + '/temp/clustering.pkl') as f:
                up = pickle.Unpickler(f)
                data = up.load()
                vertices = up.load()
        except:
            compute_data = True

    if compute_data:
        r_data = []
        vertices = []

        num_v = p.gm.g.num_vertices()
        for v in p.gm.g.vertices():
            r = p.gm.region(v)

            from utils.drawing.points import draw_points_crop_binary
            bimg = draw_points_crop_binary(r.pts())
            hu_m = get_hu_moments(np.asarray(bimg, dtype=np.uint8))
            r_data.append([r.area(), r.a_, r.b_, hu_m[0], hu_m[1]])
            vertices.append(int(v))

            i += 1
            if i % 100 == 0:
                print_progress(i, num_v)

        data = np.array(r_data)
        vertices=np.array(vertices)

        print_progress(num_v, num_v)
        print

    # label_names = np.array(['area', 'major axis', 'minor axis', 'hu1', 'hu2'])

    min_samples = max(5, int(len(data) * 0.001))
    eps = 0.1

    print "Normalising data..."
    X = StandardScaler().fit_transform(data)
    print "Clustering using DBSCAN... min samples: {}, eps: {}".format(min_samples, eps)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    labels_set = set(labels)
    n_clusters_ = len(labels_set) - (1 if -1 in labels_set else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    for i in labels_set:
        print "\tLabel: {}, #{}".format(i, np.sum(labels == i))

    # plotNdto3d(data, labels, core_samples_mask, [0, 1, 2], label_names[[0, 1, 2]])
    # plotNdto3d(data, labels, core_samples_mask, [0, 2, 3], label_names[[0, 2, 3]])
    # plotNdto3d(data, labels, core_samples_mask, [0, 2, 4], label_names[[0, 2, 4]])

    print "saving results"
    with open(p.working_directory+'/temp/clustering.pkl', 'wb') as f:
        pic = pickle.Pickler(f)
        pic.dump(data)
        pic.dump(vertices)
        pic.dump(labels)

    print "clustering part finished"
    print "_________________________________"


def display_cluster_representants(p, N=30):
    with open(p.working_directory+'/temp/clustering.pkl') as f:
        up = pickle.Unpickler(f)
        data = up.load()
        vertices = up.load()
        labels = up.load()

    labels_set = set(labels)
    scaler = StandardScaler()

    X = scaler.fit_transform(data)

    for label in labels_set:
        X_ = X[labels==label,:]
        print "displaying cluster {} representants, cluster size: {}".format(label, len(X_))
        vertices_ = vertices[labels==label]

        if len(X_) == 0:
            print "ZERO SIZE CLUSTER: ", label
            continue

        n_clusters = min(N, len(X_))

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_)

        kmeans_labels = kmeans.labels_
        data = []

        vm = get_auto_video_manager(p)

        for k in range(n_clusters):
            class_member_mask = (kmeans_labels == k)
            a_ = vertices_[class_member_mask]

            v1 = a_[0]

            r1 = p.gm.region(v1)

            im1 = vm.get_frame(r1.frame()).copy()

            draw_points(im1, r1.pts())

            im = im1[r1.roi().slices()].copy()
            data.append(im)

        collage = create_collage_rows(data, 7, 100, 100)
        cv2.imshow('collage', collage)
        cv2.waitKey(0)
        # cv2.imwrite(p.working_directory+'/temp/cluster_representant_'+str(label)+'.jpg', collage)

def __draw_region(p, vm, v):
    r1 = p.gm.region(v)
    im1 = vm.get_frame(r1.frame()).copy()
    draw_points(im1, r1.pts())
    im = im1[r1.roi().slices()].copy()

    return im

def __controls():
    k = cv2.waitKey()

    if k == 115:
        print "single"
    elif k == 109:
        print "multi"
    elif k == 110:
        print "noise"
    elif k == 112:
        print "part"


def most_distant(p):
    with open(p.working_directory+'/temp/clustering.pkl') as f:
        up = pickle.Unpickler(f)
        data = up.load()
        vertices = up.load()
        labels = up.load()

    labels_set = set(labels)
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    vm = get_auto_video_manager(p)
    id_ = 0


    # data =
    
    data = [__draw_region(p, vm, vertices[id_])]

    d = None
    for i in range(1):
        new_d = cdist([X[id_]], X)
        if d is None:
            d = new_d
        else:
            d = np.minimum(d, new_d)

        id_ = np.argmax(d)
        print d[0, id_], id_
        data.append(__draw_region(p, vm, vertices[id_]))

    collage = create_collage_rows(data, 18, 100, 100)
    cv2.imshow('collage', collage)
    print "wait key"
    print cv2.waitKey(0)
    print cv2.waitKey(0)
    print cv2.waitKey(0)
    print cv2.waitKey(0)
    print cv2.waitKey(0)

    pass

    # for label in labels_set:
    #     X_ = X[labels==label,:]
    #     print "displaying cluster {} representants, cluster size: {}".format(label, len(X_))
    #     vertices_ = vertices[labels==label]
    #
    #     if len(X_) == 0:
    #         print "ZERO SIZE CLUSTER: ", label
    #         continue
    #
    #     n_clusters = min(N, len(X_))
    #
    #     from sklearn.cluster import KMeans
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_)
    #
    #     kmeans_labels = kmeans.labels_
    #     data = []
    #
    #     vm = get_auto_video_manager(p)
    #
    #     for k in range(n_clusters):
    #         class_member_mask = (kmeans_labels == k)
    #         a_ = vertices_[class_member_mask]
    #
    #         v1 = a_[0]
    #
    #         im = __draw_region(p, vm, v1)
    #         data.append(im)
    #
    #     collage = create_collage_rows(data, 7, 100, 100)
    #     cv2.imshow('collage', collage)
    #     cv2.waitKey(0)