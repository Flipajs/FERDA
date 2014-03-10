__author__ = 'flipajs'

import pickle
import numpy as np
from numpy.random import rand
from pylab import figure, show
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import math
import time
import my_utils

def get_points(region):
    points = []
    for r in region['rle']:
        for c in range(r['col1'], r['col2'] + 1):
            points.append([c, r['line']])

    return points


def get_trans_matrix(ellipse):
    e = float(ellipse.width)/2
    f = float(ellipse.height)/2
    th = ellipse.angle * math.pi / 180

    scale = np.array([[1 / e, 0], [0, 1 / f]])
    rot = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])

    t_matrix = np.dot(scale, rot)

    return t_matrix


def point_transformation(trans_matrix, ellipse_middle, pt):
    pt2 = np.array(pt - ellipse_middle)
    pt2 = np.dot(trans_matrix, pt2.reshape(2, 1))
    pt2 = pt2.reshape(2) + ellipse_middle

    return pt2


#def load_data():
#    file10 = open('../out/collisions/regions_139pkl', 'rb')
#    regions10 = pickle.load(file10)
#    file10.close()
#
#    mser = regions10[3]
#
#    data = get_points(mser)
#    data = np.array(data)
#
#    return data
#
#
#def ellipse_1():
#    e = 25.
#    f = 12.
#    theta = 70 * math.pi / 180
#    middle = np.array([435, 20])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell
#
#
#def ellipse_2():
#    e = 25.
#    f = 12.
#    theta = 45 * math.pi / 180
#    middle = np.array([450, 23])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell


#def load_data():
#    file10 = open('../out/collisions/regions_332pkl', 'rb')
#    regions10 = pickle.load(file10)
#    file10.close()
#
#    mser = regions10[0]
#
#    data = get_points(mser)
#    data = np.array(data)
#
#    return data
#
#
#def ellipse_1():
#    e = 25.
#    f = 12.
#    theta = 30 * math.pi / 180
#    middle = np.array([100, 590])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell
#
#
#def ellipse_2():
#    e = 25.
#    f = 12.
#    theta = 45 * math.pi / 180
#    middle = np.array([106, 609])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell


#def load_data():
#    file10 = open('../out/collisions/regions_29pkl', 'rb')
#    regions10 = pickle.load(file10)
#    file10.close()
#
#    mser = regions10[0]
#
#    data = get_points(mser)
#    data = np.array(data)
#
#    return data
#
#
#def ellipse_1():
#    e = 25.
#    f = 12.
#    theta = 30 * math.pi / 180
#    middle = np.array([725, 210])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell
#
#
#def ellipse_2():
#    e = 25.
#    f = 12.
#    theta = 0 * math.pi / 180
#    middle = np.array([730, 225])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell


#def load_data():
#    file10 = open('../out/collisions/regions_102pkl', 'rb')
#    regions10 = pickle.load(file10)
#    file10.close()
#
#    mser = regions10[14]
#
#    data = get_points(mser)
#    data = np.array(data)
#
#    return data
#
#
#def ellipse_1():
#    e = 25.
#    f = 12.
#    theta = 100 * math.pi / 180
#    middle = np.array([362, 18])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell
#
#
#def ellipse_2():
#    e = 25.
#    f = 12.
#    theta = 80 * math.pi / 180
#    middle = np.array([340, 29])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell


#def load_data():
#    file10 = open('../out/collisions/regions_146pkl', 'rb')
#    regions10 = pickle.load(file10)
#    file10.close()
#
#    mser = regions10[20]
#
#    data = get_points(mser)
#    data = np.array(data)
#
#    return data
#
#
#def ellipse_1():
#    e = 25.
#    f = 12.
#    theta = 70 * math.pi / 180
#    middle = np.array([434, 12])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell
#
#
#def ellipse_2():
#    e = 25.
#    f = 12.
#    theta = 140 * math.pi / 180
#    middle = np.array([450, 13])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell


#def load_data():
#    file10 = open('../out/collisions/regions_148pkl', 'rb')
#    regions10 = pickle.load(file10)
#    file10.close()
#
#    mser = regions10[5]
#
#    data = get_points(mser)
#    data = np.array(data)
#
#    return data
#
#
#def ellipse_1():
#    e = 25.
#    f = 12.
#    theta = 70 * math.pi / 180
#    middle = np.array([434, 12])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell
#
#
#def ellipse_2():
#    e = 25.
#    f = 12.
#    theta = 140 * math.pi / 180
#    middle = np.array([450, 13])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell


#def load_data():
#    file10 = open('../out/collisions/regions_159pkl', 'rb')
#    regions10 = pickle.load(file10)
#    file10.close()
#
#    mser = regions10[1]
#
#    data = get_points(mser)
#    data = np.array(data)
#
#    return data
#
#
#def ellipse_1():
#    e = 10.
#    f = 4.
#    theta = 70 * math.pi / 180
#    middle = np.array([456, 14])
#    ell = Ellipse(xy=middle, width=e*2, height=f*2, angle=(theta * 180 / math.pi))
#
#    return ell
#
#
#def ellipse_2():
#    e = 10.
#    f = 4.
#    theta = 140 * math.pi / 180
#    middle = np.array([465, 16])
#    ell = Ellipse(xy=middle, width=e*2, height=f*2, angle=(theta * 180 / math.pi))
#
#    return ell

def load_data():
    file10 = open('../out/collisions/regions_210pkl', 'rb')
    regions10 = pickle.load(file10)
    file10.close()

    mser = regions10[8]

    data = get_points(mser)
    data = np.array(data)

    return data


def ellipse_1():
    e = 25.
    f = 12.
    theta = 20 * math.pi / 180
    middle = np.array([800, 427])
    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))

    return ell


def ellipse_2():
    e = 25.
    f = 12.
    theta = 90 * math.pi / 180
    middle = np.array([798, 415])
    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))

    return ell

#def load_data():
#    file10 = open('../out/collisions/regions_161pkl', 'rb')
#    regions10 = pickle.load(file10)
#    file10.close()
#
#    mser = regions10[0]
#
#    data = get_points(mser)
#    data = np.array(data)
#
#    return data
#
#
#def ellipse_1():
#    e = 25.
#    f = 12.
#    theta = 90 * math.pi / 180
#    middle = np.array([456, 14])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell
#
#
#def ellipse_2():
#    e = 25.
#    f = 12.
#    theta = 0 * math.pi / 180
#    middle = np.array([464, 16])
#    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))
#
#    return ell


def draw_ellipse(ell, pl):
    pl.add_artist(ell)
    ell.set_clip_box(pl.bbox)
    ell.set_alpha(0.6)
    ell.set_facecolor(rand(3))
    plt.plot(ell.center[0], ell.center[1], 'bx', 2)


def point_score(ell, t_mat, pt):
    pt2 = point_transformation(t_mat, ell.center, pt)
    return np.linalg.norm(pt2 - ell.center)


def label(data, ell1, ell2):
    labels = np.array([0]*len(data[:, 0]))

    t_mat1 = get_trans_matrix(ell1)
    t_mat2 = get_trans_matrix(ell2)

    i = 0
    for pt in data:
        pt1 = point_transformation(t_mat1, ell1.center, pt)
        pt2 = point_transformation(t_mat2, ell2.center, pt)

        s1 = point_score(ell1, t_mat1, pt1)
        s2 = point_score(ell2, t_mat2, pt2)

        if s1 < s2:
            labels[i] = 1
        else:
            labels[i] = 2

        i += 1

    return labels


def update_centre(data, ell):
    center = [0, 0]
    i = 0
    for p in data:
        center += p
        i += 1

    center /= i

    ell.center = center


def update_theta(data, ell):
    theta_sum = 0
    i = 0
    theta = ell.angle * math.pi / 180
    for p in data:
        pt = p - ell.center
        if pt[0] != 0:
            t = math.atan(pt[1]/float(pt[0])) - theta

            if t < -math.pi / 2:
                t = -(t + math.pi / 2)
            if t > math.pi / 2:
                t = -(t - math.pi / 2)

            theta_sum += t

        i += 1

    theta_sum /= float(i)
    theta_new = theta + theta_sum
    #theta_new = theta_sum
    ell.angle = theta_new * 180 / math.pi


def update_theta_med(data, ell):
    theta_difs = [0] * len(data[:, 1])
    theta = ell.angle * math.pi / 180

    i = 0
    for p in data:
        pt = p - ell.center
        if pt[0] != 0:
            t = math.atan(pt[1]/float(pt[0])) - theta

            if t < -math.pi / 2:
                t = -(t + math.pi / 2)
            if t > math.pi / 2:
                t = -(t - math.pi / 2)

            theta_difs[i] = t

        i += 1

    if len(theta_difs) % 2 == 0:
        mid = len(theta_difs) / 2
        theta_new = sorted(theta_difs)[mid]
    else:
        mid = len(theta_difs) / 2
        s = sorted(theta_difs)
        theta_new = s[mid]
        theta_new += s[mid+1]
        theta_new /= 2

    theta_new += theta
    ell.angle = theta_new * 180 / math.pi


def update_theta_moments(data, ell):
    u00 = len(data[:, 0])

    m11 = 0
    m10 = 0
    m01 = 0
    m20 = 0
    m02 = 0

    for pt in data:
        m11 += pt[0] * pt[1]
        m10 += pt[0]
        m01 += pt[1]
        m20 += pt[0]*pt[0]
        m02 += pt[1]*pt[1]


    cx = m10/float(u00)
    cy = m01/float(u00)

    u11 = m11 - cx*m01
    u20 = m20 - cx*m10
    u02 = m02 - cy*m01

    u11 /= float(u00)
    u20 /= float(u00)
    u02 /= float(u00)

    #theta = my_utils.mser_theta(u11, u20, u02)
    theta = 0.5*math.atan2(2*u11, (u20 - u02))

    ell.angle = theta * 180 / math.pi
    ell.center = [cx, cy]


def test_end(ell1, ell2, old_c1, old_c2, old_t1, old_t2):
    center_eps = 1
    theta_eps = 5
    dx1 = abs(ell1.center[0] - old_c1[0])
    dy1 = abs(ell1.center[1] - old_c1[1])
    dx2 = abs(ell2.center[0] - old_c2[0])
    dy2 = abs(ell2.center[1] - old_c2[1])

    if dx1 < center_eps and dy1 < center_eps and dx2 < center_eps and dy2 < center_eps:
        if abs(ell1.angle - old_t1) < theta_eps and abs(ell2.angle - old_t2) < theta_eps:
            return True

    return False


def k_ellipse(data, ell1, ell2):
    old_c1 = ell1.center
    old_c2 = ell2.center
    old_t1 = ell1.angle
    old_t2 = ell2.angle

    visualize_init(data, ell1, ell2)

    while True:
        print old_c1, old_c2, old_t1, old_t2

        labels = label(data, ell1, ell2)
        l1 = np.where(labels == 1)
        l2 = np.where(labels == 2)

        print "BEFORE reweighting"
        visualize(data, l1, l2, ell1, ell2)

        update_theta_moments(data[l1], ell1)
        update_theta_moments(data[l2], ell2)

        if test_end(ell1, ell2, old_c1, old_c2, old_t1, old_t2):
            print "FINISHED: "

        visualize(data, l1, l2, ell1, ell2)

        if test_end(ell1, ell2, old_c1, old_c2, old_t1, old_t2):
            break

        old_c1 = ell1.center
        old_c2 = ell2.center
        old_t1 = ell1.angle
        old_t2 = ell2.angle


def visualize_init(data, ell1, ell2):
    plt.close()
    plt.ion()
    fig = figure()
    ax = fig.add_subplot(111, aspect='equal')
    plt.plot(data[:, 0], data[:, 1], 'ko')

    plt.axis('equal')
    a, b, c, d = plt.axis()
    border = 5
    plt.axis((a-border, b+border, c-border, d+border))

    #draw_ellipse(ell1, ax)
    #draw_ellipse(ell2, ax)

    epts1 = get_ellipse_coords(a=ell1.width/2, b=ell1.height/2, x=ell1.center[0], y=ell1.center[1], angle=-ell1.angle, k=1./8)
    epts2 = get_ellipse_coords(a=ell2.width/2, b=ell2.height/2, x=ell2.center[0], y=ell2.center[1], angle=-ell2.angle, k=1./8)
    plt.plot(epts1[:, 0], epts1[:, 1], 'y', linewidth=3)
    plt.plot(ell1.center[0], ell1.center[1], 'rx', mew=2)

    plt.plot(epts2[:, 0], epts2[:, 1], 'g', linewidth=3)
    plt.plot(ell2.center[0], ell2.center[1], 'rx', mew=2)

    plt.gca().invert_yaxis()
    show()
    plt.waitforbuttonpress()


def visualize(data, l1, l2, ell1, ell2, noellipse=False):
    plt.close()
    fig = figure()
    ax = fig.add_subplot(111, aspect='equal')
    plt.plot(data[l1, 0], data[l1, 1], 'yo')
    plt.plot(data[l2, 0], data[l2, 1], 'go')

    plt.axis('equal')
    a, b, c, d = plt.axis()
    border = 5
    plt.axis((a-border, b+border, c-border, d+border))

    epts1 = get_ellipse_coords(a=ell1.width/2, b=ell1.height/2, x=ell1.center[0], y=ell1.center[1], angle=-ell1.angle, k=1./8)
    epts2 = get_ellipse_coords(a=ell2.width/2, b=ell2.height/2, x=ell2.center[0], y=ell2.center[1], angle=-ell2.angle, k=1./8)
    if not noellipse:
        plt.plot(epts1[:, 0], epts1[:, 1], 'y', linewidth=3)
        plt.plot(ell1.center[0], ell1.center[1], 'rx', mew=2)
        plt.plot(epts2[:, 0], epts2[:, 1], 'g', linewidth=3)
        plt.plot(ell2.center[0], ell2.center[1], 'rx', mew=2)
        #draw_ellipse(ell1, ax)
        #draw_ellipse(ell2, ax)

    plt.gca().invert_yaxis()
    show()
    plt.waitforbuttonpress()


def get_ellipse_coords(a=0.0, b=0.0, x=0.0, y=0.0, angle=0.0, k=2):
    """ Draws an ellipse using (360*k + 1) discrete points; based on pseudo code
    given at http://en.wikipedia.org/wiki/Ellipse
    k = 1 means 361 points (degree by degree)
    a = major axis distance,
    b = minor axis distance,
    x = offset along the x-axis
    y = offset along the y-axis
    angle = clockwise rotation [in degrees] of the ellipse;
        * angle=0  : the ellipse is aligned with the positive x-axis
        * angle=30 : rotated 30 degrees clockwise from positive x-axis
    """
    pts = np.zeros((360*k+1, 2))

    beta = -angle * np.pi/180.0
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    alpha = np.radians(np.r_[0.:360.:1j*(360*k+1)])

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)

    pts[:, 0] = x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta)
    pts[:, 1] = y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta)

    return pts


def main():
    data = load_data()
    ell1 = ellipse_1()
    ell2 = ellipse_2()

    #fig = figure()
    #update_theta_moments(data, ell1)
    #ax = fig.add_subplot(111, aspect='equal')
    #plt.plot(data[:, 0], data[:, 1], 'ko')
    #
    #epts1 = get_ellipse_coords(a=ell1.width/2, b=ell1.height/2, x=ell1.center[0], y=ell1.center[1], angle=ell1.angle, k=1./8)
    #plt.gca().invert_yaxis()
    #plt.plot(epts1[:, 0], epts1[:, 1], 'y', linewidth=3)
    #
    #
    #print "angle: ", ell1.angle
    #plt.plot(data[:, 0], data[:, 1], 'o')
    #plt.axis('equal')
    #plt.show()


    start = time.time()
    k_ellipse(data, ell1, ell2)
    t = time.time() - start

    print "TIME: ", t



if __name__ == '__main__':
    main()