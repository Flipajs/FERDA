__author__ = 'filip@naiser.cz'

import numpy as np
import math
import matplotlib.pyplot as plt
import my_utils


#create matrix a with points of reagions
def fill_a(rle, ant_area, cx, cy):
    a = np.zeros((2, ant_area))
    i = 0
    for r in rle:
        for c in range(r['col1'], r['col2']+1):
            #we must invert Y axis
            a[:, i] = [c-cx, -(r['line']-cy)]
            i += 1

    return a


def fill_values(main_axis, a2):
    if abs(main_axis[0]) > abs(100*main_axis[1]):
        values = a2[0, :]
    elif abs(main_axis[0]*100) < abs(main_axis[1]):
        values = a2[1, :]
    else:
        values = a2

    return values


def process_values(values, main_axis, num_px):
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
            val = values[:, i] / main_axis
            new_val = math.sqrt(val[0]*val[0] + val[1]*val[1])
            new_val = math.copysign(new_val, val[0])
            #new_val = math.copysign(new_val, val[1])
            val = new_val*(-1)
            values2[0, i] = val

            if val < min:
                min = val
            elif val > max:
                max = val

        values = values2

    return max, min, values


def hist_test(values, max, min, num_px):
    length = max-min
    num_bins = int(math.floor(length))
    hist = [0]*num_bins

    positions = [0]*num_px
    for i in range(num_px):
        v = values[0, i]
        position = int(((v-min) / float(length)) * (num_bins - 1))
        positions[i] = position
        hist[position] += 1


    skip = int(math.floor(length/10))
    sum_len = int(math.floor(length/7))

    #n, bins, patches = plt.hist(positions, num_bins, histtype='bar')
    #plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    #plt.grid()
    #plt.show()
    #plt.close()

    l = r = 0
    for i in range(skip, skip+sum_len+1):
        l += hist[i]

    for i in range(num_bins-1-skip-sum_len, num_bins-skip):
        r += hist[i]

    if l > r:
        return math.pi

    return 0


#returns angle between 0 and 2PI
def head_estimation(region):
    theta = my_utils.mser_theta(region['sxy'], region['sxx'], region['syy'])
    a = fill_a(region['rle'], region['area'], region['cx'], region['cy'])
    U, s, V = np.linalg.svd(a, full_matrices=False)

    s2 = np.array([[s[0], 0], [0, 0]])
    U = np.mat(U)
    s2 = np.mat(s2)
    V = np.mat(V)
    a2 = U*s2*V

    main_axis = U[:, 0]

    if main_axis[1] < 0:
        main_axis[0] = -main_axis[0]
        main_axis[1] = -main_axis[1]

    values = fill_values(main_axis, a2)
    max, min, values = process_values(values, main_axis, region['area'])

    theta += hist_test(values, max, min, region['area'])

    return theta