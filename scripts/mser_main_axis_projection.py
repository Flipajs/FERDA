from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from builtins import range
from past.utils import old_div
__author__ = 'filip@naiser.cz'

import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import my_utils

#file = open('../out/regions_2pkl', 'rb')
file = open('../out/test.pkl', 'rb')

regions = pickle.load(file)
file.close()

ids = [2, 3, 5, 7,  8, 9, 10, 13, 14, 16, 17, 20, 21, 22, 24, 25, 26, 28, 29, 31, 32, 33, 34, 36, 37, 39, 40, 42, 43, 44, 46, 47]

for id in ids:
#for id in range(10):
    ant1 = regions[id]
    print('ID: ', id)

    num_px = ant1['area']

    a = np.zeros((2, num_px))

    i = 0
    for rle in ant1['rle']:
        for c in range(rle['col1'], rle['col2']+1):
            #we must invert Y axis
            a[:, i] = [c-ant1['cx'], -(rle['line']-ant1['cy'])]
            i += 1

    U, s, V = np.linalg.svd(a, full_matrices=False)

    s2 = np.array([[s[0], 0], [0, 0]])

    U = np.mat(U)
    s2 = np.mat(s2)
    V = np.mat(V)
    a2 = U*s2*V

    main_axis = U[:, 0]
    if abs(main_axis[0]) > abs(100*main_axis[1]):
    #if True:
        values = a2[0, :]
        print("Y")
    elif abs(main_axis[0]*100) < abs(main_axis[1]):
        values = a2[1, :]
        print("X")
    else:
        values = a2
        print("XY")

    min = float('inf')
    max = float('-inf')

    if values.shape[0] == 1:
        for i in range(num_px):
            v = values[0, i]
            if v < min:
                min = v
            elif v > max:
                max = v
    else:
        values2 = np.zeros((1, num_px))
        for i in range(num_px):
            val = old_div(values[:, i], main_axis)
            new_val = math.sqrt(val[0]*val[0] + val[1]*val[1])
            new_val = math.copysign(new_val, val[0])
            new_val = math.copysign(new_val, val[1])
            val = new_val*(-1)
            values2[0, i] = val

            if val < min:
                min = val
            elif val > max:
                max = val

        values = values2

    #M00 = num_px
    #cx = ant1['cx']
    #cy = ant1['cy']
    #M10 = cx*M00
    #M01 = cy*M00
    #M11 = ant1['sxy']
    #M20 = ant1['sxx']
    #M02 = ant1['syy']
    #
    #u11 = M11 - cx*M01
    #u20 = M20 - cx*M10
    #u02 = M02 - cy*M01
    #
    #u_20 = M20 / M00 - cx*cx
    #u_02 = M02 / M00 - cy*cy
    #u_11 = M11 / M00 - cx*cy
    #
    #theta = my_utils.mser_theta(M11, M20, M02)
    #
    #print theta
    #print theta * 180 / math.pi

    theta1 = 0.5*math.atan2(2*ant1['sxy'], (ant1['sxx'] - ant1['syy']))
    theta1 = -theta1
    theta2 = 0.5*math.atan(old_div(ant1['sxy'], (old_div((ant1['sxx'] - ant1['syy']), 2))))

    #print "__Theta1: ", theta1 * 180 / math.pi
    #print "__Theta2: ", (-theta2) * 180 / math.pi
    #theta3 = my_utils.mser_theta(ant1['sxy'], ant1['sxx'], ant1['syy'])
    #print "__Theta: ", theta3 * 180 / math.pi

    m_a_r = old_div(s[0],s[1])

    main_x = main_axis[0]
    main_y = main_axis[1]


    r, lambda1, lambda2 = my_utils.mser_main_axis_ratio(ant1['sxy'], ant1['sxx'], ant1['syy'])
    print('Main axis (svd): ', main_axis, m_a_r)
    print('Lambdas (moments): ', lambda1, lambda2, r)

    length = max - min
    print('Length: ', length)

    theta = my_utils.mser_theta(ant1['sxy'], ant1['sxx'], ant1['syy'])
    print('Theta1: ', theta * 180 / math.pi)

    num_bins = int(math.floor(length))
    hist = [0]*num_bins

    positions = [0]*num_px
    for i in range(num_px):
        v = values[0, i]
        position = int((old_div((v-min), float(length))) * (num_bins - 1))
        positions[i] = position
        hist[position] += 1


    skip = int(math.floor(old_div(length,10)))
    sum_len = int(math.floor(old_div(length,7)))
    print('skip: ', skip)

    l = r = 0
    for i in range(skip, skip+sum_len+1):
        l += hist[i]

    for i in range(num_bins-1-skip-sum_len, num_bins-skip):
        r += hist[i]

    print("in direction: ", l)
    print("opposite: ", r)

    if l > r:
        theta += math.pi

    print('Theta2: ', theta * 180 / math.pi)

    n, bins, patches = plt.hist(positions, num_bins, histtype='bar')
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    plt.grid()
    plt.show()
    plt.close()