import math
from scipy import ndimage

__author__ = 'filip@naiser.cz'

import cv2
import numpy as np
from utils import video_manager
import experiment_params
import mser_operations
import my_utils
import visualize

#BGR
def color_nearby(c, im, max_d):
    # id0 = np.abs(np.int32(im[:,:,0])-c[0]) < max_d
    # id1 = np.abs(np.int32(im[:,:,1])-c[1]) < max_d
    # id2 = np.abs(np.int32(im[:,:,2])-c[2]) < max_d
    #
    # print id0.shape
    # id = np.logical_and(np.logical_and(id0, id1), id2)
    # print id
    id = np.linalg.norm(im-c, axis=2) < max_d
    # print id
    c_new = np.asarray((np.asarray(c, dtype=np.float) * 1.3), dtype=np.uint8)
    # im[id] = c_new
    im[id] = [255, 0, 255]

    return im
    # cv2.imshow('test2', im)
    # cv2.waitKey(0)

def kmeans_step(im, labels, means):
    rows, cols, _ = im.shape

    best_dists = np.ones(labels.shape, dtype=np.float) * 10000

    labels = np.zeros(labels.shape, dtype=np.uint8)
    for m in range(len(means)):
        dists = np.linalg.norm(im - means[m], axis=2)
        ids = dists < best_dists
        best_dists[ids] = dists[ids]
        labels[ids] = m

    # print "before nearest"
    # for y in range(rows):
    #     for x in range(cols):
    #         best_dist = np.inf
    #         label = -1
    #         for m in range(len(means)):
    #             # ci = im[y, x, :]
    #             # cm = means[m]
    #             # d = math.sqrt((ci[0]-cm[0])**2 + (ci[1]-cm[1])**2 + (ci[2]-cm[2])**2)
    #             # d = math.sqrt((ci[0]-cm[0])**2 + (ci[1]-cm[1])**2 + (ci[2]-cm[2])**2)
    #             d = (np.linalg.norm(im[y, x, :] - means[m]))
    #             if d < best_dist:
    #                 best_dist = d
    #                 label = m
    #
    #         labels[y, x] = label

    # print "after nearest"

    mean_counts = [0]*len(means)
    for m in range(len(means)):
        means[m] = np.zeros(3,)

    # for y in range(rows):
    #     for x in range(cols):
    #         l = labels[y, x]
    #         mean_counts[l] += 1
    #         means[l] += im[y, x, :]

    for m in range(len(means)):
        means[m] = np.zeros(3,)
        ids = labels == m
        means[m] = np.sum(im[ids, ], axis=0)
        s = np.sum(ids)
        if s == 0:
            means[m] = np.random.rand(3,)
        else:
            means[m] /= np.sum(ids)

    for m in range(len(means)):
        if(mean_counts[m] > 0):
            means[m] /= mean_counts[m]

    return labels, means


def color_kmeans(im, color_num):
    means = []
    for i in range(color_num):
        means.append(np.random.rand(3,)*255)


    print "INIT: ", means
    print ""

    labels = np.zeros([im.shape[0], im.shape[1]], np.uint8)

    i = 0
    while True:
        print i
        old_labels = labels
        labels, means = kmeans_step(im, labels, means)

        s = np.sum(np.sum(old_labels-labels))
        if s == 0:
            break

        i += 1

    return means

def get_color_around(im, pos, radius):
    c = np.zeros((1, 3), dtype=np.double)
    num_px = 0
    for h in range(radius*2 + 1):
        for w in range(radius*2 + 1):
            d = ((w-radius)**2 + (h-radius)**2)**0.5
            if d <= radius:
                num_px += 1
                c += im[pos[1]-radius+w,pos[0]-radius+h,:]


    print num_px
    c /= num_px

    return [c[0,0], c[0,1], c[0,2]]

if __name__ == "__main__":
    # vid = video_manager.get_auto_video_manager(['/media/flipajs/Seagate Expansion Drive/IST - videos/compressed/bigLenses_colormarks1/compressed.avi',
    #                                             '/media/flipajs/Seagate Expansion Drive/IST - videos/compressed/bigLenses_colormarks1/lossless.avi'])

    # vid = video_manager.get_auto_video_manager('/media/flipajs/Seagate Expansion Drive/IST - videos/bigLenses_colormarks2.avi')
    # im = vid.move2_next()

    im = cv2.imread('/home/flipajs/Pictures/test/im_014.png')

    ibg = np.asarray(im.copy(), dtype=np.double)
    ibg[:, :, 0] = np.sum(ibg, axis=2)
    ibg[:,:,1] = ibg[:,:,1] / ibg[:,:,0]
    ibg[:,:,2] = ibg[:,:,2] / ibg[:,:,0]

    radius = 2
    dot1_p = (601, 142) #orange
    dot2_p = (720, 237) #grey
    dot3_p = (507, 648) #dark blue
    dot4_p = (108, 663) #pink


    dot = dot4_p


    c1 = get_color_around(im, dot, radius)
    cv2.circle(im, dot, radius, (255,0,0), -1)

    cv2.imshow('im', im)


    c2 = [79, 47, 127] # purple
    c3 = [17, 71, 104] # orange
    c4 = [36, 53, 23] # green
    c5 = [141, 165, 150] # gray

    c2 = c1

    s = np.double(c2[0]+c2[1]+c2[2])
    c2_ = [s, c2[1]/s, c2[2]/s]


    m_ = np.max(ibg[:,:,0])
    ibg[:,:,0] /= m_
    c2_[0] /= m_


    print c2_
    print np.max(ibg)

    distim = np.linalg.norm(ibg-c2_, axis=2)

    print distim.shape
    print np.max(distim)
    distim /= np.max(distim)
    print np.max(distim)
    print np.min(distim)
    distim = np.asarray(distim*255, dtype=np.uint8)



    cv2.imshow('img', im)
    cv2.imshow('distim', distim)
    cv2.waitKey(0)

    #
    #
    # m = np.mean(im)
    #
    # cbg = [
    #     np.median(im[:,:,0]),
    #     np.median(im[:,:,1]),
    #     np.median(im[:,:,2])
    # ]
    #
    # cbg2 = [m,m,m]
    #
    # print cbg, cbg2
    #
    # # im = color_nearby(cbg, im, 50)
    # #
    # # cv2.imshow("im", im)
    # # cv2.waitKey(0)
    #
    # # im = color_nearby(cbg2, im, 30)
    # #
    # # cv2.imshow("im", im)
    # # cv2.waitKey(0)
    #
    # print im.shape[0], im.shape[1]
    # gim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # bim = ndimage.gaussian_filter(gim, sigma=3)
    # bim = np.asarray(bim, dtype=np.double)
    #
    # gim = np.asarray(gim, dtype=np.double)
    #
    # print "MAX, ", np.max(bim)
    # test = (255-bim+1)/255.0
    # id = test > 0.5
    # test[id] = 1.0
    #
    # print "MAX, ", np.max(test)
    # # test = np.asarray(test, dtype=np.uint8)
    #
    # amount = 50
    # im_ = (gim - (amount * test)) / (255-amount)
    #
    # cv2.imshow("mask", test)
    # cv2.imshow("gb", im_)
    # p = np.zeros((im.shape[0], im.shape[1]), dtype=np.double)
    # p = np.asarray(np.linalg.norm(im-cbg, axis=2), dtype=np.double)
    # print np.max(p)
    # p /= np.max(p)
    #
    # p2 = (np.min(im, axis=2) + np.max(im, axis=2))/510.0
    #
    # # p_uint8 = np.asarray(p, dtype=np.uinndimage.gaussian_filter(lena, sigma=3)t8)
    # # cv2.imshow("p", p)
    # # cv2.imshow("p2", p2)
    # cv2.waitKey(0)
    #
    # # im = np.asarray(im, dtype=np.float)
    # # means = color_kmeans(im, 10)
    # # im = np.asarray(im, dtype=np.uint8)
    # #
    # # print means
    # # for m in range(len(means)):
    # #     imc = im.copy()
    # #     print means[m]
    # #     im2 = color_nearby(means[m], imc, 10)
    # #
    # #     cv2.imshow("test", im2)
    # #     cv2.waitKey(0)
    #
    #
    # if False:
    #     params = experiment_params.Params()
    #     mser_ops = mser_operations.MserOperations(params)
    #     for i in range(10):
    #         im = vid.move2_next()
    #
    #         regions, chosen_regions_indexes = mser_ops.process_image(im)
    #         print chosen_regions_indexes
    #
    #         groups, _ = mser_operations.get_region_groups2(regions, check_flags=False)
    #         im_msers = visualize.draw_region_group_collection(im, regions, groups, params)
    #         # im_msers = visualize.draw_region_best_margins_collection(im, regions, chosen_regions_indexes, [])
    #         # im_msers = visualize.draw_region_collection(im, regions, params)
    #         # cv2.imshow("before", im_msers)
    #
    #         cbg = [
    #             np.median(im[:,:,0]),
    #             np.median(im[:,:,1]),
    #             np.median(im[:,:,2])
    #         ]
    #
    #
    #         im = color_nearby(c2, im, 30)
    #         im = color_nearby(c3, im, 30)
    #         im = color_nearby(c4, im, 30)
    #         # im = color_nearby(c5, im, 15)
    #
    #         # im = color_nearby(cbg, im, 70)
    #
    #         regions, chosen_regions_indexes = mser_ops.process_image(im)
    #         groups, _ = mser_operations.get_region_groups2(regions, check_flags=False)
    #         # im_msers = visualize.draw_region_collection(im, regions, params)
    #         # im_msers = visualize.draw_region_best_margins_collection(im, regions, chosen_regions_indexes, [])
    #         im_msers = visualize.draw_region_group_collection(im, regions, groups, params)
    #         cv2.imshow("after", im_msers)
    #         cv2.imshow("img", im)
    #
    #
    #         cv2.waitKey(0)