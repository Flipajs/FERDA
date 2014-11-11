from reportlab.graphics.barcode.widgets import _BarcodeWidget

__author__ = 'filip@naiser.cz'

import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import time
import math


def crit(ant1, ant2, mser):
    print mser

    a1r0, a1c0, a1r1, a1c1 = region_size(ant1['region']['rle'])
    a2r0, a2c0, a2r1, a2c1 = region_size(ant2['region']['rle'])
    margin = math.ceil(count_margin(a1c1-a1c0, a1r1-a1r0, a2c1-a2c0, a2r1-a2r0) / 4) + 4

    reg = create_region_img(mser['rle'], margin)
    reg = scipy.misc.imresize(reg, 0.25)
    print reg.shape
    ant1_reg = create_region_img(ant1['region']['rle'], 1, square=True)
    ant1_reg = scipy.misc.imresize(ant1_reg, 0.25)
    print ant1_reg.shape
    ant2_reg = create_region_img(ant2['region']['rle'], 1, square=True)
    ant2_reg = scipy.misc.imresize(ant2_reg, 0.25)
    print ant2_reg.shape

    plt.imshow(ant1_reg, cmap='gray')
    plt.show()

    start = time.time()
    pos_step = 4
    theta_step = 12
    cx = reg.shape[1] / 2
    cy = reg.shape[0] / 2
    #x_step = (reg.shape[1] - 2*margin) / pos_step
    #y_step = (reg.shape[0] - 2*margin) / pos_step

    x_step = 1
    y_step = 1

    p_s = 2
    p_s1 = p_s + 1

    min_score = sys.maxint
    min_params = []

    for th1 in range(0, 180, 25):
        for th2 in range(0, 180, 25):
            for pos1_x in range(cx - p_s*x_step, cx + p_s1*x_step, x_step):
                for pos1_y in range(cy - p_s*y_step, cy+p_s1*y_step, y_step):
                    for pos2_x in range(cx - p_s*x_step, cx+p_s1*x_step, x_step):
                        for pos2_y in range(cy - p_s*y_step, cy+p_s1*y_step, y_step):
                            score = count_score((pos1_y, pos1_y), th1, (pos2_y, pos2_x), th2, reg, ant1_reg, ant2_reg)
                            print th1, th2, pos1_x, pos1_y, pos2_x, pos2_y
                            if score < min_score:
                                min_score = score
                                min_params = {'th1': th1, 'th2': th2, 'pos1_x': pos1_x, 'pos1_y': pos1_y, 'pos2_x': pos2_x, 'pos2_y': pos2_y, 'score': score}


    #score = count_score((cy, cx), 10, (cy, cx), 70, reg, ant1_reg, ant2_reg)
    end = time.time()
    print min_params
    print end - start


def count_score(ant1_pos, ant1_theta, ant2_pos, ant2_theta, region, ant1_reg, ant2_reg):
    ant1_rot = ndimage.rotate(ant1_reg, ant1_theta, reshape=False)
    ant2_rot = ndimage.rotate(ant2_reg, ant2_theta, reshape=False)

    a1_cy = ant1_pos[0] - ant1_rot.shape[0] / 2
    a1_cx = ant1_pos[1] - ant1_rot.shape[1] / 2

    a2_cy = ant2_pos[0] - ant2_rot.shape[0] / 2
    a2_cx = ant2_pos[1] - ant2_rot.shape[1] / 2

    ant1 = np.zeros(region.shape, dtype=np.uint8)
    ant2 = np.zeros(region.shape, dtype=np.uint8)

    ant1[a1_cy:a1_cy+ant1_rot.shape[0], a1_cx:a1_cx+ant1_rot.shape[1]] += ant1_rot
    ant2[a2_cy:a2_cy+ant2_rot.shape[0], a2_cx:a2_cx+ant2_rot.shape[1]] += ant2_rot

    ant12 = ant1+ant2

    img_xor = np.bitwise_xor(ant12, region)
    img_and = np.bitwise_and(ant1, ant2)
    xor_sum = my_sum(img_xor)
    and_sum = my_sum(img_and)

    weight = 0.25
    return xor_sum + weight*and_sum


def my_sum(img_xor):
    counter = 0
    for line in img_xor:
        for px in line:
            if px > 0:
                counter += 1

    return counter


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


def count_margin(a1_weight, a1_height, a2_weight, a2_height):
    margin = max(max(a1_weight, a1_height), max(a2_weight, a2_height)) / 2
    margin += 3

    return margin


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



file9 = open('../out/collisions/regions_209pkl', 'rb')
regions9 = pickle.load(file9)
file9.close()

file10 = open('../out/collisions/regions_210pkl', 'rb')
regions10 = pickle.load(file10)
file10.close()

mser = regions10[8]

ant1 = {'region': regions9[20], 'x': 10, 'y': 20, 'theta': 0.1}
ant2 = {'region': regions9[24], 'x': 20, 'y': 10, 'theta': 0.2}

crit(ant1, ant2, mser)

