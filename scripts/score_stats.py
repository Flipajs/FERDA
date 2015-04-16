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
from core.region.mser import get_msers_
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
from scripts.similarity_test import similarity_loss
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats.stats import pearsonr

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


    a_ = np.zeros((ids, num_))

    for i in range(num_):
        for id in range(ids):
            r1 = chunks[id][i]
            r2 = chunks[id][i+1]
            thetas[id, i] = r1.theta_ - r2.theta_
            similarities[id, i] = similarity_loss(r1, r2)

            a_[id, i] = r1.a_

            if i == 0:
                pred = np.array([0, 0])
            else:
                pred = r1.centroid() - chunks[id][i-1].centroid()

            distances100[id, i] = np.linalg.norm(r1.centroid() + pred - r2.centroid())
            minI[id, i] = r2.min_intensity_ - r1.min_intensity_
            maxI[id, i] = r2.max_intensity_ - r1.max_intensity_


    n_bins = 100

    orientation_data = np.array(np.abs(thetas.reshape(-1)))
    avg_main_axis_len = 2*np.mean(a_.reshape(-1))
    distance_data = np.array(distances100.reshape(-1)) / avg_main_axis_len
    similarity_data = np.array(similarities.reshape(-1))
    mini_data = np.array(minI.reshape(-1))

    print "O D", pearsonr(orientation_data, distance_data)
    print "O S", pearsonr(orientation_data, similarity_data)
    print "O M", pearsonr(orientation_data, mini_data)
    print "D S", pearsonr(distance_data, similarity_data)
    print "D M", pearsonr(distance_data, mini_data)
    print "D M", pearsonr(similarity_data, mini_data)

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
    data = np.array(distances100.reshape(-1)) / avg_main_axis_len
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
    data = np.array(distances100.reshape(-1)) / avg_main_axis_len
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
    data = np.array(distances100.reshape(-1)) / avg_main_axis_len
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
    print np.min(l_)
    plt.plot(bins[:-1], l_)


    plt.subplot(4, 2, 3)
    plt.title('histogram of distances from predicted position')
    avg_main_axis_len = 2*np.mean(a_.reshape(-1))
    data = np.array(distances100.reshape(-1)) / avg_main_axis_len
    bins = np.linspace(np.min(data), np.max(data), n_bins)
    data = np.append(data, bins)
    data = gaussian_filter1d(data, 3)
    h_ = plt.hist(data, bins, normed=True)

    plt.subplot(4, 2, 4)
    plt.title('log of hist of distances from predicted position')
    l_ = np.log(h_[0] + 1)
    log_hists['distances'] = {'bins': bins, 'data': l_}
    print np.min(l_)
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
    print np.min(l_)
    plt.plot(bins[:-1], l_)



    plt.subplot(4, 2, 7)
    plt.title('histogram of region min intensity diffs')
    data = np.array(minI.reshape(-1))
    bins = np.linspace(np.min(data), np.max(data), np.max(data)-np.min(data))
    data = np.append(data, bins)
    h_ = plt.hist(data, bins, normed=True)

    plt.subplot(4, 2, 8)
    plt.title('log of hist of region min intensity diffs')
    l_ = np.log(h_[0] + 1)
    log_hists['minI'] = {'bins': bins, 'data': l_}
    print np.min(l_)
    plt.plot(bins[:-1], l_)

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
        print np.mean(it), np.std(it), np.median(it), np.min(it), np.max(it)


    print "ALPHA: ", np.mean(distances100) / np.mean(thetas)
    print "BETA: ", np.mean(distances100) / np.mean(similarities)