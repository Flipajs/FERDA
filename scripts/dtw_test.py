#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_DTW.PY
Date: Wednesday, March  7 2012
Description: Test DTW algorithms.
"""
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from dtw import *
import dtw.fast
# import mlpy.dtw
import numpy.random as npr
import pylab
import cv2, cv
import matplotlib.pyplot as plt


if __name__ == '__main__':

    import itertools

    # load my data
    im1a = plt.imread('/Users/flipajs/Desktop/dtw_test/1A.png')
    im2a = plt.imread('/Users/flipajs/Desktop/dtw_test/2A.png')
    im2b = plt.imread('/Users/flipajs/Desktop/dtw_test/2B.png')

    im1a = cv2.cvtColor(im1a, cv2.COLOR_RGB2GRAY)
    im1a_data = np.argwhere(im1a > 0)

    im2a = cv2.cvtColor(im2a, cv2.COLOR_RGB2GRAY)
    im2a_data = np.argwhere(im2a > 0)

    print(im1a_data)
    plt.imshow(im1a)
    plt.show()

    # create sequences of related sequences
    x = npr.normal(0,15,(10,2))
    e = npr.normal(0,1,(10,2))
    y = x + e

    xa = []
    t = np.array([0.0,0.0])
    for i in x:
        t += i
        xa.append(t.copy())

    ya = []
    t = np.array([0.0,0.0])
    for i in x + e:
        t += i
        ya.append(t.copy())


    xa = np.array(xa)
    ya = np.array(ya)

    pylab.plot(xa[:,0],xa[:,1])
    pylab.plot(ya[:,0],ya[:,1])
    print("Slow Version")
    print(dtw_distance(xa,ya)) #,[1.0,1.0,0.0])
    print("Fast Version")
    print(dtw.fast.dtw_fast(xa,ya))
    #print "MLPY"
    #print mlpy.dtw.dtw_std(xa,ya)
    print("Fast Version 2D")
    print(dtw.fast.dtw_fast_2d(xa,ya))

    pylab.show()



