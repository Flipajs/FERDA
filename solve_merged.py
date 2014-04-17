__author__ = 'flipajs'

import pickle
import numpy as np
from numpy.random import rand
from pylab import figure, show
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import math
import time

def get_points(region):
    points = []
    for r in region['rle']:
        for c in range(r['col1'], r['col2'] + 1):
            points.append([c, r['line']])

    return points


def get_trans_matrix(ellipse):
    e = float(ellipse.width)
    f = float(ellipse.height)
    th = ellipse.angle * math.pi / 180

    scale = np.array([[1 / e, 0], [0, 1 / f]])
    #clokwise...
    rot = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])

    t_matrix = np.dot(scale, rot)

    return t_matrix


def point_transformation(trans_matrix, ellipse_middle, pt):
    pt2 = np.array(pt - ellipse_middle)
    pt2 = np.dot(trans_matrix, pt2.reshape(2, 1))
    pt2 = pt2.reshape(2)

    return pt2

def draw_ellipse(ell, pl):
    pl.add_artist(ell)
    ell.set_clip_box(pl.bbox)
    ell.set_alpha(0.6)
    ell.set_facecolor(rand(3))
    plt.plot(ell.center[0], ell.center[1], 'bx', 2)


def point_score(ell, t_mat, pt):
    pt2 = point_transformation(t_mat, ell.center, pt)

    return np.linalg.norm(pt2)
    #return np.linalg.norm(pt2 - ell.center)


def label(data, ellipses):
    labels = np.array([0]*len(data[:, 0]))

    t_mat = []
    for e in ellipses:
        t_mat.append(get_trans_matrix(e))

    i = 0
    for pt in data:
        #pts = [0]*len(ellipses)
        #for j in range(len(ellipses)):
        #    pts[j] = point_transformation(t_mat[j], ellipses[j].center, pt)

        scores = [0]*len(ellipses)
        for j in range(len(ellipses)):
            scores[j] = point_score(ellipses[j], t_mat[j], pt)

        min_s = np.argmin(scores)
        labels[i] = min_s+1

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

    if u00 == 0:
        print "U00 is zero"
        return 0, 0, 0

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

    return u11, u20, u02

def test_end(ellipses, old_c, old_t):
    center_eps = 1
    theta_eps = 5

    for i in range(len(ellipses)):
        dx = abs(ellipses[i].center[0] - old_c[i][0])
        dy = abs(ellipses[i].center[1] - old_c[i][1])
        if dx > center_eps or dy > center_eps:
            return False
        if abs(ellipses[i].angle - old_t[i]) > theta_eps:
            return False

    return True


def k_ellipse(data, ellipses):
    old_c = [0]*len(ellipses)
    old_t = [0]*len(ellipses)

    for i in range(len(ellipses)):
        old_c[i] = ellipses[i].center
        old_t[i] = ellipses[i].angle

    #visualize_init(data, ellipses)

    labels = []
    moments = [[0, 0, 0] for i in range(len(ellipses))]
    iter = 0
    while True:
        print "old centers: ", old_c
        print "old_thetas: ", old_t

        if iter > 15:
            print "too much iterations in solve_merged... KILLED"
            return [], [], []

        labels = label(data, ellipses)

        #visualize(data, ellipses, labels)

        for i in range(len(ellipses)):
            l = np.where(labels == i+1)

            moments[i] = list(update_theta_moments(data[l], ellipses[i]))

        #visualize(data, ellipses, labels)
        if test_end(ellipses, old_c, old_t):
            print "FINISHED: "
            break

        for i in range(len(ellipses)):
            old_c[i] = ellipses[i].center
            old_t[i] = ellipses[i].angle

        iter += 1

    return labels, ellipses, moments


def visualize_init(data, ellipses):

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

    for e in ellipses:
        epts = get_ellipse_coords(a=e.width/2, b=e.height/2, x=e.center[0], y=e.center[1], angle=-e.angle, k=1./8)
        plt.plot(epts[:, 0], epts[:, 1], 'y', linewidth=3)
        plt.plot(e.center[0], e.center[1], 'rx', mew=2)

    plt.gca().invert_yaxis()
    show()
    plt.waitforbuttonpress()


def visualize(data, ellipses, labels, noellipse=False):

    plt.close()
    fig = figure()
    ax = fig.add_subplot(111, aspect='equal')

    colors = ['y', 'g', 'r', 'c', 'k']
    colorsm = ['yo', 'go', 'ro', 'bo', 'co', 'ko']
    for i in range(len(ellipses)):
        l = np.where(labels == i+1)
        plt.plot(data[l, 0], data[l, 1], colorsm[i])

    plt.axis('equal')
    a, b, c, d = plt.axis()
    border = 5
    plt.axis((a-border, b+border, c-border, d+border))

    if not noellipse:
        for i in range(len(ellipses)):
            e = ellipses[i]
            epts = get_ellipse_coords(a=e.width/2, b=e.height/2, x=e.center[0], y=e.center[1], angle=-e.angle, k=1./8)
            plt.plot(epts[:, 0], epts[:, 1], colors[i], linewidth=3)
            plt.plot(e.center[0], e.center[1], 'rx', mew=2)

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


def solve_merged(region, ants, ants_idx):
    ellipses = []
    data = get_points(region)
    data = np.array(data)

    for id in ants_idx:
        a = ants[id]
        #pred = a.predicted_position(1)
        #middle = np.array([pred.x, pred.y])

        middle = np.array([a.state.position.x, a.state.position.y])
        theta = a.state.theta
        theta = -(theta - math.pi)
        theta = theta * 180 / math.pi
        theta = theta % 180

        ell = Ellipse(xy=middle, width=40, height=10, angle=theta)
        ellipses.append(ell)

    if len(ellipses) > 2:
        print "MORE ANTS THEN 2!!!"

    labels, ellipses, moments = k_ellipse(data, ellipses)

    i = 1
    regions = []
    for e in ellipses:
        r = {}
        r['splitted'] = True
        l = np.where(labels == i)
        r['points'] = data[l]
        r['area'] = len(r['points'])

        if r['area'] == 0:
            print "zero area"
            continue

        r['cx'] = e.center[0]
        r['cy'] = e.center[1]
        r['maxI'] = region['maxI']
        theta = -(e.angle - 180)
        theta = (theta * math.pi) / 180
        theta = theta % math.pi

        r['theta'] = theta
        r['flags'] = ''
        r['sxy'] = moments[i-1][0]
        r['sxx'] = moments[i-1][1]
        r['syy'] = moments[i-1][2]
        r['margin'] = 0
        regions.append(r)

        i += 1

    return regions


def load_data():
    file10 = open('out/regions_670.pkl', 'rb')
    regions10 = pickle.load(file10)
    file10.close()

    mser = regions10[3]

    data = get_points(mser)
    data = np.array(data)

    return data


def ellipse_1():
    e = 40.
    f = 10.
    theta = 178 * math.pi / 180
    middle = np.array([796, 382])
    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))

    return ell


def ellipse_2():
    e = 40.
    f = 10.
    theta = 0 * math.pi / 180
    middle = np.array([787, 374])
    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))

    return ell

def ellipse_3():
    e = 40.
    f = 10.
    theta = 58 * math.pi / 180
    middle = np.array([813, 363])
    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))

    return ell

def ellipse_4():
    e = 40.
    f = 10.
    theta = 5 * math.pi / 180
    middle = np.array([817, 406])
    ell = Ellipse(xy=middle, width=e, height=f, angle=(theta * 180 / math.pi))

    return ell

def main():
    data = load_data()
    ell1 = ellipse_1()
    ell2 = ellipse_2()
    ell3 = ellipse_3()
    ell4 = ellipse_4()
    ellipses = [ell1, ell2, ell3, ell4]

    k_ellipse(data, ellipses)


if __name__ == '__main__':
    main()