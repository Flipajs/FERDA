from __future__ import print_function
__author__ = 'fnaiser'

import numpy as np
from numpy.linalg import norm
from matplotlib.mlab import normpdf
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm


MAX_SPEED = 100

def null(A, atol=1e-13, rtol=0):
    """
    Returns null space of matrix A
    :param a:
    :param rtol:
    :return:
    """
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    return ns


def get_H(pts, pts_):
    """
    estimate homography based on 4 points references pts -> pts_
    :param a:
    :param b:
    :return:
    """
    A = np.zeros((8, 9))
    for i in range(4):
        x = pts[i][0]
        y = pts[i][1]
        x_ = pts_[i][0]
        y_ = pts_[i][1]

        A[i*2, :] = np.array([-x, -y, -1, 0, 0, 0, x_*x, x_*y, x_])
        A[i*2 + 1, :] = np.array([0, 0, 0, -x, -y, -1, y_*x, y_*y, y_])

    H = null(A)
    H = H.reshape(3, 3)

    return H

if __name__ == '__main__':
    p1 = np.array([50, 40])
    v = np.array([50, 50])

    M = p1 + 0.5*v

    v_size = norm(v)

    if v_size < 1.4142:

        print("no deformation")
    else:
        w = v*(MAX_SPEED / v_size)
        a = M + w

        q = (MAX_SPEED - v_size) / v_size
        b = M + np.array([-q*v[1], q*v[0]])
        c = M - q*v
        d = M + np.array([q*v[1], -q*v[0]])

        std = MAX_SPEED / 3
        a_ = [std, 0]
        b_ = [0, std]
        c_ = [-std, 0]
        d_ = [0, -std]


    pts1 = [a, b, c, d]
    pts2 = [a_, b_, c_, d_]

    i = 0

    c = ['r', 'b', 'g', 'c']

    H = get_H(pts1, pts2)

    rs = np.linspace(0, MAX_SPEED*0.5, 10)
    qs = np.linspace(-0.5, 0.5, 30)

    theta = np.linspace(-np.pi, np.pi, 20)

    pts = []
    x = 5
    y = 5

    colors = cm.rainbow(np.linspace(0, 1, len(rs)))

    plt.figure(1)
    plt.subplot(121)

    i = 0

    pts = []
    for r in rs:
        x = r
        y = r
        pts.append([])
        for t in theta:
            x_ = math.cos(t) * x
            y_ = math.sin(t) * y

            p = M + np.array([x_, y_])

            plt.scatter(p[0], p[1], c=colors[i])
            plt.hold(True)

            pts[i].append(p)

        i += 1

    plt.grid()
    plt.axis('equal')

    plt.subplot(122)

    pts_ = []
    i = 0
    for r in pts:
        for p in r:
            v_ = np.array([p[0], p[1], 1])
            p_ = np.dot(H, v_)
            p_ /= p_[2]
            pts_.append(p_)

            plt.scatter(p_[0], p_[1], c=colors[i])
            plt.hold(True)

        i += 1

    plt.grid(True)
    plt.axis('equal')
    plt.hold(False)
    plt.show()