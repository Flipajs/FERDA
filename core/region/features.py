__author__ = 'fnaiser'

import numpy as np
from math import cos, sin, ceil
from utils.roi import get_roi
import pickle
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
import cv2
import matplotlib.pyplot as plt
import pylab
from scipy.ndimage.filters import gaussian_filter1d
from utils.drawing.points import draw_points_crop, draw_points


def shape_description(region, n):
    """
    returns list of distances to region contour in n equally distributed angles, starting in direction of main axis
    :param region:
    :param n: number of measured distances equally distributed
    :return: np array of length n with distances
    """

    dists = np.zeros(n)
    # dists += ceil(region.a_)
    angles = np.linspace(0, 2*np.pi, n)

    e = ceil(region.a_ * 1.5)

    offset = (int(region.centroid()[0] - e - 1), int(region.centroid()[1] - e - 1))
    offset = (0, 0)

    # im = np.zeros((e * 2 + 1, e * 2 + 1), dtype=np.bool)
    im = np.zeros((1024, 1024), dtype=np.bool)
    im[region.pts()[:, 0] - offset[0], region.pts()[:, 1] - offset[1]] = True

    middle = region.centroid() - np.array(offset)
    j = 0
    for a in angles:
        th = (region.theta_ + np.pi/2 + a) % (2*np.pi)

        prev = im[middle[0], middle[1]]
        for i in range(1, int(ceil(region.a_))):
            y = cos(th) * i
            x = sin(th) * i

            p = middle + np.array([y, x])
            actual = im[p[0], p[1]]

            if prev != actual:
                dists[j] = (x**2 + y**2)**0.5
                break
                # prev = actual

        j += 1

    return dists


def id_correlation():
    working_dir = '/Users/fnaiser/Documents/chunks'

    with open(working_dir+'/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)

    my_id = 0
    my_ids = []
    rest_ids = []
    for i in range(200):
        my_ids.append(chunks[my_id][i])
        for j in range(8):
            if j == my_id:
                continue
            rest_ids.append(chunks[j][i])

    descs = []

    i = -1
    for r in my_ids:
        i += 1
        # print r.centroid(), r.area()
        try:
            d = shape_description(r, 18)
        except:
            print "ROI PROBLEM", i
            continue

        d = gaussian_filter1d(d, 1)
        descs.append(d)


    # for r in rest_ids:
    #     print r.centroid(), r.area()
    #     try:
    #         d = shape_description(r, 18)
    #     except:
    #         continue
    #
    #     d = gaussian_filter1d(d, 1)
    #     descs.append(d)


    data = np.array(descs)
    plt.figure(1)
    R = np.corrcoef(data)
    pylab.pcolor(R)
    pylab.colorbar()
    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
    plt.show()


if __name__ == '__main__':
    # id_correlation()



    with open('/Users/fnaiser/Documents/graphs/mser/1.pkl', 'rb') as f:
        msers = pickle.load(f)

    groups = get_region_groups(msers)
    ids = margin_filter(msers, groups)
    ids = children_filter(msers, ids)
    ids = area_filter(msers, ids, 50)

    regions = [msers[i] for i in ids]

    descs = []
    i = 0
    for r in regions:
        # e_ = ceil(r.a_ * 1.5)
        # offset = (int(r.centroid()[0] - e_ - 1), int(r.centroid()[1] - e_ - 1))
        # im = np.zeros((e_ * 2 + 1, e_ * 2 + 1), dtype=np.bool)
        # try:
        #     im[r.pts()[:, 0] - offset[0], r.pts()[:, 1] - offset[1]] = True
        # except:
        #     print "ROI problem"
        #     continue

        crop = draw_points(np.zeros((1024, 1024, 3), dtype=np.uint8), r.pts())
        crop = draw_points(crop, np.array([np.asarray(r.centroid(), dtype=np.int32)]), color=(0, 0, 255, 0.5))
        crop = draw_points(crop, np.array([np.asarray(r.centroid() + np.array([0, 1]), dtype=np.int32)]), color=(0, 0, 255, 0.5))
        crop = draw_points(crop, np.array([np.asarray(r.centroid() + np.array([1, 0]), dtype=np.int32)]), color=(0, 0, 255, 0.5))
        crop = draw_points(crop, np.array([np.asarray(r.centroid() + np.array([0, -1]), dtype=np.int32)]), color=(0, 0, 255, 0.5))
        crop = draw_points(crop, np.array([np.asarray(r.centroid() + np.array([-1, 0]), dtype=np.int32)]), color=(0, 0, 255, 0.5))

        cv2.imshow('region'+str(i), np.asarray(crop, dtype=np.uint8)*255)
        # cv2.moveWindow('region'+str(i), 80*(i/7), 200 + (i%7)*70)
        # cv2.waitKey(0)

        d = shape_description(r, 18)
        d = gaussian_filter1d(d, 1)
        print i, d
        descs.append(np.array(d))

        i+=1

    # ids = [0, 2, 4, 6, 7, 9, 11, 13]

    data = np.array(descs)
    # data_ = {}
    # i = 0
    # for d in descs:
    #     data_[i] = d
    #     i +=1


    plt.figure(1)
    i = 0
    for d in descs:
        if i < 8:
            plt.plot(d)
            plt.hold(True)

        i+=1

    plt.hold(False)
    plt.legend([str(i) for i in range(8)])
    plt.show()
    plt.cla()


    plt.figure(1)
    R = np.corrcoef(data)
    pylab.pcolor(R)
    pylab.colorbar()

    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
    plt.show()


    # plt.figure(2)
    # from pandas.tools.plotting import scatter_matrix
    # import pandas
    # df = pandas.DataFrame(data.T)
    # print "continuing"
    # a = scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
    #
    # plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
    # plt.show()
