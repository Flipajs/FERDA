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
from PyQt4 import QtGui
from tqdm import tqdm


def get_data(r, scaler=None):
    from utils.drawing.points import draw_points_crop_binary
    # bimg = draw_points_crop_binary(r.pts())
    # hu_m = get_hu_moments(np.asarray(bimg, dtype=np.uint8))
    d = [r.area(), r.ellipse_major_axis_length(), r.ellipse_minor_axis_length(), r.min_intensity_, r.max_intensity_, r.margin_, len(r.contour()), r.ellipse_area_ratio()]

    if scaler is None:
        return d
    else:
        return scaler.transform(np.array([d]))[0]


def prepare_region_cardinality_samples(p, compute_data=True, num_random=1000):
    print "___________________________________"
    print "Preparing data for prepare_region_cardinality_samples..."

    if not compute_data:
        try:
            with open(p.working_directory + '/temp/region_cardinality_samples.pkl') as f:
                up = pickle.Unpickler(f)
                data = up.load()
                vertices = up.load()
                scaler = up.load()
        except:
            compute_data = True

    if compute_data:
        r_data = []
        vertices = []

        import random

        tracklet_ids = p.chm.chunks_.keys()

        for _ in tqdm(range(num_random), leave=False):
            t = p.chm[random.choice(tracklet_ids)]
            b = random.randint(0, len(t)-1)

            v = t[b]
            r = p.gm.region(v)

            r_data.append(get_data(r))
            vertices.append(int(v))

        data = np.array(r_data)
        vertices=np.array(vertices)

        print "preparations for region prepare_region_cardinality_samples FINISHED\n"

    min_samples = max(5, int(len(data) * 0.001))
    eps = 0.1

    print "Normalising data..."
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

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
    with open(p.working_directory+'/temp/region_cardinality_samples.pkl', 'wb') as f:
        pic = pickle.Pickler(f)
        pic.dump(data)
        pic.dump(vertices)
        pic.dump(labels)
        pic.dump(scaler)

    print "prepare_region_cardinality_samples part finished"
    print "_________________________________"


def display_cluster_representants(p, N=30):
    with open(p.working_directory+'/temp/region_cardinality_samples.pkl') as f:
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

def draw_region(p, vm, v):
    r1 = p.gm.region(v)
    im1 = vm.get_frame(r1.frame()).copy()
    c1 = QtGui.QColor(255, 0, 0, 255)
    draw_points(im1, r1.contour(), color=c1)
    c2 = QtGui.QColor(255, 0, 0, 20)
    draw_points(im1, r1.pts(), color=c2)
    roi = r1.roi().safe_expand(30, im1)
    im = im1[roi.slices()].copy()

    return im

if __name__ == '__main__':
    from core.project.project import Project

    p = Project()
    p.load_semistate('/Users/flipajs/Documents/wd/FERDA/zebrafish_playground')

    prepare_region_cardinality_samples(p, compute_data=True)