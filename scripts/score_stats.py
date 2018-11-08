from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from builtins import range
from past.utils import old_div
__author__ = 'fnaiser'

import pickle
import numpy as np
from utils.video_manager import get_auto_video_manager
from utils.drawing.points import draw_points, draw_points_crop
import cv2
from math import sin, cos
from PyQt4 import QtGui, QtCore
import sys
from gui.img_grid.img_grid_widget import ImgGridWidget
from gui.gui_utils import get_image_label
from core.region.mser import get_msers_img
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
from scripts.similarity_test import similarity_loss
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats.stats import pearsonr
from core.region import region
from utils.drawing.points import get_contour

import matplotlib.pyplot as plt


WORKING_DIR = '/Users/fnaiser/Documents/chunks'

def load_chunks():
    with open(WORKING_DIR+'/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)

    return chunks


if __name__ == '__main__':
    chunks = load_chunks()

    num_ = 500
    ids = 8
    thetas = np.zeros((ids, num_))
    similarities = np.zeros((ids, num_))
    distances100 = np.zeros((ids, num_))
    minI = np.zeros((ids, num_))
    maxI = np.zeros((ids, num_))
    avgI = np.zeros((ids, num_))
    percentile10I = np.zeros((ids, num_))

    contour_len_diff = np.zeros((ids, num_))
    contour_len = np.zeros((ids, num_))

    vid = get_auto_video_manager(WORKING_DIR + '/eight.m4v')


    a_ = np.zeros((ids, num_))

    img1 = vid.next_frame()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    for i in range(num_):
        img2 = vid.next_frame()
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        for id in range(ids):
            r1 = chunks[id][i]
            r2 = chunks[id][i+1]
            # t1 = (region.get_orientation(r1.sxx_, r1.syy_, r1.sxy_) + np.pi/2) % np.pi
            # t2 = (region.get_orientation(r2.sxx_, r2.syy_, r2.sxy_) + np.pi/2) % np.pi
            t1 = region.get_orientation(r1.sxx_, r1.syy_, r1.sxy_)
            t2 = region.get_orientation(r2.sxx_, r2.syy_, r2.sxy_)

            c1 = len(get_contour(r1.pts()))
            c2 = len(get_contour(r2.pts()))

            contour_len_diff[id, i] = c2 - c1
            contour_len[id, i] = c1

            if t1 < 0:
                t1 += np.pi
            if t2 < 0:
                t2 += np.pi

            t_ = max(t1, t2) - min(t1, t2)

            if t_ > old_div(np.pi,2):
                t_ = np.pi - t_
            # t_ %= np.pi/2

            # t_ = abs((t1-t2)) % np.pi/2
            # thetas[id, i] = max(t1, t2) - min(t1,t2)
            thetas[id, i] = t_


            # thetas[id, i] = (t1 + np.pi) - (t2 % np.pi)

            # thetas[id, i] = r1.theta_ - r2.theta_

            similarities[id, i] = old_div(abs(r1.area() - r2.area()), float(min(r1.area(), r2.area())))
            # similarities[id, i] = similarity_loss(r1, r2)

            a_[id, i] = r1.ellipse_major_axis_length()

            if i == 0:
                pred = np.array([0, 0])
            else:
                pred = r1.centroid() - chunks[id][i-1].centroid()
                # pred = np.array([0, 0])

            distances100[id, i] = np.linalg.norm(r1.centroid() + pred - r2.centroid())

            intensities1 = img1[r1.pts()[:, 0], r1.pts()[:, 1]]
            intensities2 = img2[r2.pts()[:, 0], r2.pts()[:, 1]]

            avgI[id, i] = np.mean(intensities1) - np.mean(intensities2)
            percentile10I[id, i] = np.percentile(intensities1, 10) - np.percentile(intensities2, 10)

            minI[id, i] = r2.min_intensity_ - r1.min_intensity_
            maxI[id, i] = r2.max_intensity_ - r1.max_intensity_

        img1 = img2


    n_bins = 100

    orientation_data = np.array(np.abs(thetas.reshape(-1)))
    avg_main_axis_len = 2*np.mean(a_.reshape(-1))
    print(avg_main_axis_len)
    distance_data = old_div(np.array(distances100.reshape(-1)), avg_main_axis_len)
    similarity_data = np.array(similarities.reshape(-1))
    mini_data = np.array(minI.reshape(-1))

    print("O D", pearsonr(orientation_data, distance_data))
    print("O S", pearsonr(orientation_data, similarity_data))
    print("O M", pearsonr(orientation_data, mini_data))
    print("D S", pearsonr(distance_data, similarity_data))
    print("D M", pearsonr(distance_data, mini_data))
    print("D M", pearsonr(similarity_data, mini_data))

    ##### dist orientation 2d hist
    h, x, y, p = plt.hist2d(orientation_data, distance_data, bins = 20, normed=True)

    plt.imshow(h, origin = "lower")
    plt.title('2d histogram')
    plt.ylabel('distance from prediction')
    plt.xlabel('orientation change')
    plt.colorbar()
    plt.savefig(WORKING_DIR+'/2d_dist_orientation.png')
    plt.clf()


    ##### dist similarity 2d hist
    avg_main_axis_len = 2*np.mean(a_.reshape(-1))
    data = old_div(np.array(distances100.reshape(-1)), avg_main_axis_len)
    h, x, y, p = plt.hist2d(similarity_data, distance_data, bins = 20, normed=True)
    plt.imshow(h, origin = "lower")
    plt.title('2d histogram')
    plt.ylabel('distance from prediction')
    plt.xlabel('overlap score')
    plt.colorbar()
    plt.savefig(WORKING_DIR+'/2d_dist_overlap.png')
    plt.clf()



    ##### similarity orientation 2d hist
    avg_main_axis_len = 2*np.mean(a_.reshape(-1))
    data = old_div(np.array(distances100.reshape(-1)), avg_main_axis_len)
    h, x, y, p = plt.hist2d(orientation_data, similarity_data, bins = 20, normed=True)
    plt.imshow(h, origin = "lower")
    plt.title('2d histogram')
    plt.ylabel('overlap score')
    plt.xlabel('oreianteation')
    plt.colorbar()
    plt.savefig(WORKING_DIR+'/2d_overlap_orientation.png')
    plt.clf()


    ##### similarity minI 2d hist
    avg_main_axis_len = 2*np.mean(a_.reshape(-1))
    data = old_div(np.array(distances100.reshape(-1)), avg_main_axis_len)
    h, x, y, p = plt.hist2d(similarity_data, mini_data, bins = 20, normed=True)
    plt.imshow(h, origin = "lower")
    plt.title('2d histogram')
    plt.ylabel('minI')
    plt.xlabel('overlap score')
    plt.colorbar()
    plt.savefig(WORKING_DIR+'/2d_minI_overlap.png')
    plt.clf()


    log_hists = {}

    plt.subplot(4, 2, 1)
    plt.title('histogram of absolute values of orientation diff')
    data = np.array(np.abs(thetas.reshape(-1)))
    bins = np.linspace(np.min(data), np.max(data), n_bins)
    data = np.append(data, bins)
    h_ = plt.hist(data, bins, normed=True)

    plt.subplot(4, 2, 2)
    plt.title('log of hist of absolute values of orientation diff')
    l_ = np.log(h_[0] + 1)
    log_hists['thetas'] = {'bins': bins, 'data': l_}
    print(np.min(l_))
    plt.plot(bins[:-1], l_)


    plt.subplot(4, 2, 3)
    plt.title('histogram of distances from predicted position')
    avg_main_axis_len = 2*np.mean(a_.reshape(-1))
    data = old_div(np.array(distances100.reshape(-1)), avg_main_axis_len)
    bins = np.linspace(np.min(data), np.max(data), n_bins)
    data = np.append(data, bins)
    data = gaussian_filter1d(data, 3)
    h_ = plt.hist(data, bins, normed=True)

    plt.subplot(4, 2, 4)
    plt.title('log of hist of distances from predicted position')
    l_ = np.log(h_[0] + 1)
    log_hists['distances'] = {'bins': bins, 'data': l_}
    print(np.min(l_))
    plt.plot(bins[:-1], l_)


    plt.subplot(4, 2, 5)
    plt.title('histogram of overlap scores')
    data = np.array(similarities.reshape(-1))
    bins = np.linspace(np.min(data), np.max(data), n_bins)
    data = np.append(data, bins)
    h_ = plt.hist(data, bins, normed=True)

    plt.subplot(4, 2, 6)
    plt.title('log of hist of overlap scores')
    l_ = np.log(h_[0] + 1)
    log_hists['similarities'] = {'bins': bins, 'data': l_}
    print(np.min(l_))
    plt.plot(bins[:-1], l_)



    plt.subplot(4, 2, 7)
    plt.title('histogram of region min intensity diffs')
    data = np.array(contour_len_diff.reshape(-1))
    bins = np.linspace(np.min(data), np.max(data), np.max(data)-np.min(data))
    data = np.append(data, bins)
    h_ = plt.hist(data, bins, normed=True)

    plt.subplot(4, 2, 8)
    plt.title('histogram of region min intensity diffs')
    data = np.array(contour_len.reshape(-1))
    bins = np.linspace(np.min(data), np.max(data), np.max(data)-np.min(data))
    data = np.append(data, bins)
    h_ = plt.hist(data, bins, normed=True)


    # plt.title('log of hist of region min intensity diffs')
    # l_ = np.log(h_[0] + 1)
    # log_hists['minI'] = {'bins': bins, 'data': l_}
    # print np.min(l_)
    # plt.plot(bins[:-1], l_)

    with open(WORKING_DIR+'/log_hists.pkl', 'wb') as f:
        pickle.dump(log_hists, f)


    plt.show()
    # plt.savefig(WORKING_DIR+'/histograms.png')

    # plt.subplot(312)
    # avg_main_axis_len = 2*np.mean(a_.reshape(-1))
    # plt.hist(distances100.reshape(-1) / avg_main_axis_len, normed=True, bins=n_bins)
    #
    # plt.subplot(313)
    # plt.hist(similarities.reshape(-1), normed=True, bins=100)

    # plt.show()
    #
    # plt.plot(similarities[0, :])
    # plt.show()

    with open(WORKING_DIR+'/thetas.pkl', 'wb') as f:
        pickle.dump(thetas, f)

    with open(WORKING_DIR+'/similarities.pkl', 'wb') as f:
        pickle.dump(similarities, f)

    with open(WORKING_DIR+'/distances.pkl', 'wb') as f:
        pickle.dump(distances100, f)

    thetas = np.abs(thetas)

    for it in [thetas, similarities, distances100, minI, maxI]:
        print(np.mean(it), np.std(it), np.median(it), np.min(it), np.max(it))


    print("ALPHA: ", old_div(np.mean(distances100), np.mean(thetas)))
    print("BETA: ", old_div(np.mean(distances100), np.mean(similarities)))