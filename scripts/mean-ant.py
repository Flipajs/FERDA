__author__ = 'filip@naiser.cz'

import time
import pickle
import numpy as np
import sys
import cv2
import math
import head_estimator
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def region_size(rle):
    row_start = rle[0]['line']
    col_start = sys.maxint
    col_end = 0

    for l in rle:
        if l['col1'] < col_start:
            col_start = l['col1']
        if l['col2'] > col_end:
            col_end = l['col2']

    row_end = l['line']

    return row_start, col_start, row_end, col_end


def create_region_img(rle, margin, square=True):
    r0, c0, r1, c1 = region_size(rle)
    height = r1-r0
    width = c1-c0
    if square:
        m = max(height, width)
        height = width = m

    img = np.zeros((height + 2*margin - 1, width + 2*margin - 1), dtype=np.uint8)

    for l in rle:
        line = l['line'] - r0 + margin - 1
        col_start = l['col1'] - c0 + margin - 1
        col_end = l['col2'] - c0 + margin

        if square:
            line += (height - (r1 - r0)) / 2
            d = (width - (c1 - c0)) / 2
            col_start += d
            col_end += d

        img[line, col_start:col_end] = 255

    return img


def create_centered_region_img(region, width, height, ant_num=1):
    img = np.zeros((height, width), dtype=np.uint8)
    mx = width/2
    my = width/2
    cx = region['cx']
    cy = region['cy']

    val = 255
    if ant_num > 1:
        val = math.floor(255/float(ant_num)) - 1

    for l in region['rle']:
        line = l['line'] - cy + my
        col_start = l['col1'] - cx + mx -1
        col_end = l['col2'] - cx + mx

        img[line, col_start:col_end] = val

    return img

def create_centered_region_img_half(region, width, height, ant_num=1):
    img = np.zeros((height, width), dtype=np.uint8)
    mx = width/2
    my = width
    cx = region['cx']
    cy = region['cy']

    val = 255
    if ant_num > 1:
        val = math.floor(255/float(ant_num)) - 1

    for i in range(1, len(region['rle']), 2):
        l = region['rle'][i]
        line = (l['line'] - cy + my) / 2
        col_start = l['col1'] - cx + mx -1
        col_end = l['col2'] - cx + mx

        c_len = col_end - col_start

        col_start += math.floor(c_len/4)
        col_end -= math.ceil(c_len/4)

        img[line, col_start:col_end] = val

    return img


def main():
    half_mode = True

    file = open('../out/ants_2.pkl', 'rb')
    ants = pickle.load(file)
    file.close()

    file = open('../out/regions_2pkl', 'rb')
    regions = pickle.load(file)
    file.close()


    imgs = [None] * len(ants)

    width = height = 60
    if half_mode:
        width /= 2
        height /= 2

    accum_img = np.zeros((height, width), dtype=np.float32)
    ant_num = 8

    for i in range(len(ants)):
        region = regions[ants[i].mser_id]
        if half_mode:
            img = create_centered_region_img_half(region, width, height, ant_num)
        else:
            img = create_centered_region_img(region, width, height, ant_num)

        theta = head_estimator.head_estimation(region)
        img_rot = ndimage.rotate(img, -(theta * 180 / math.pi), reshape=False)
        imgs[i] = img_rot

        accum_img += img_rot

        #plt.ion()
        #plt.imshow(img2, cmap='gray')
        #plt.waitforbuttonpress()
        #
        #plt.ion()
        #plt.imshow(img, cmap='gray')
        #plt.waitforbuttonpress()

    mean_ant_inv = 1/(accum_img + 1)

    plt.ion()
    plt.imshow(accum_img, cmap='gray')
    plt.waitforbuttonpress()

    for i in range(len(ants)):
        start = time.time()
        region = regions[ants[i].mser_id]
        t = time.time() - start
        #print "region_t: ", t

        start = time.time()
        if half_mode:
            img = create_centered_region_img_half(region, width, height, 1)
        else:
            img = create_centered_region_img(region, width, height, 1)
        t = time.time() - start
        #print "create_centered_reg_t: ", t

        #theta = ants[i].theta
        theta = head_estimator.head_estimation(region)
        start = time.time()
        img_rot = ndimage.rotate(img, -(theta * 180 / math.pi), reshape=False, order=0)
        t = time.time() - start
        #print "rotate_t: ", t

        start = time.time()
        img_dif = mean_ant_inv*img_rot
        t = time.time() - start
        #print "weighting: ", t

        #print "theta: ", theta
        print img_dif.sum() / (width*height)

        plt.ion()
        plt.imshow(img, cmap='gray')
        plt.waitforbuttonpress()

        #cv2.imshow("img", img)
        #cv2.imshow("dif", img2)
        #cv2.imwrite("test.png", img2)
        #cv2.waitKey(0)

if __name__ == '__main__':
    main()
