__author__ = 'flipajs'

import pickle
import numpy as np
import cv2
import sys
import math
import my_utils
import time

def get_points(region):
    points = []
    for r in region['rle']:
        for c in range(r['col1'], r['col2'] + 1):
            points.append([c, r['line']])

    return points


def get_contour(region):
    min_c = 100000
    max_c = 0
    min_r = region['rle'][0]['line']
    max_r = region['rle'][-1]['line']

    for r in region['rle']:
        if min_c > r['col1']:
            min_c = r['col1']
        if max_c < r['col2']:
            max_c = r['col2']

    rows = max_r - min_r
    cols = max_c - min_c

    img = np.zeros((rows+1, cols+1, 1), dtype=np.uint8)

    for r in region['rle']:
        row = r['line'] - min_r
        col1 = r['col1'] - min_c
        col2 = r['col2'] - min_c
        img[row][col1:col2+1] = 255


    #cv2.imshow("img", img)

    ret,thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt = contours[0]

    img_cont = np.zeros((rows+1, cols+1, 1), dtype=np.uint8)
    pts = []
    for p in cnt:
        img_cont[p[0][1]][p[0][0]] = 255
        pts.append([p[0][0] + min_c, p[0][1] + min_r])

    #cv2.imshow("test", img2)
    #cv2.waitKey(0)

    return pts, img, img_cont, min_r, min_c


def load_region():
    file = open('../out/collisions/regions_312pkl', 'rb')
    regions = pickle.load(file)
    file.close()

    r = regions[0]
    cont, img, img_cont, min_r, min_c = get_contour(r)

    region = {'cont': cont, 'y_min': min_r, 'x_min': min_c, 'img': img, 'img_cont': img_cont}
    return region


def count_theta(region):
    theta = 0.5*math.atan2(2*region['sxy'], (region['sxx'] - region['syy']))

    return theta


def load_ants():
    file = open('../out/noplast_errors/regions/136.pkl', 'rb')
    regions = pickle.load(file)
    file.close()

    r1 = regions[20]
    r2 = regions[32]
    cont1, img1, img_c1, min_r1, min_c1 = get_contour(r1)
    cont2, img2, img_c2, min_r2, min_c2 = get_contour(r2)

    th1 = count_theta(r1)
    #print "TH1: ", th1,  th1*57
    th2 = count_theta(r2)
    #print "TH2: ", th2, th2*57


    ant1 = {'id': 0, 'x': r1['cx'], 'y': r1['cy'], 'theta': th1, 'cont': cont1, 'min_y': min_r1, 'min_x': min_c1}
    ant2 = {'id': 1, 'x': r2['cx'], 'y': r2['cy'], 'theta': th2, 'cont': cont2, 'min_y': min_r2, 'min_x': min_c2}

    ants = [ant1, ant2]

    return ants


def e_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def nearest_p_on_ant_bound(pt, ants):
    best_dist = sys.maxint
    best_ant = -1
    best = [-1, -1]

    for a in ants:
        for apt in a['cont']:
            d = e_dist(apt, pt)
            if d < best_dist:
                best = apt
                best_dist = d
                best_ant = a['id']

    return best, best_dist, best_ant


def nearest_p_on_region_bound(apt, region):
    best_dist = sys.maxint
    best = [-1, -1]
    for pt in region['cont']:
        d = e_dist(pt, apt)
        if d < best_dist:
            best = pt
            best_dist = d

    return best, best_dist


def is_inside_region(apt, r):
    x = apt[0] - r['x_min']
    y = apt[1] - r['y_min']
    if x < 0:
        return False
    if x >= r['img'].shape[1]:
        return False

    if y < 0:
        return False
    if y >= r['img'].shape[0]:
        return False

    if r['img'][y][x] == 0:
        return False

    return True


def region_boundary_cover(region, ants):
    s = 0
    pairs = []
    for pt in region['cont']:
        apt, dist, best_ant = nearest_p_on_ant_bound(pt, ants)
        pairs.append([apt, pt, best_ant])
        s += dist

    return pairs, s


def ant_boundary_cover(region, ant):
    s = 0
    pairs = []
    for apt in ant['cont']:
        if not is_inside_region(apt, region):
            pt, dist = nearest_p_on_region_bound(apt, region)
            pairs.append([apt, pt])
            s += dist

    return pairs, s


def draw_situation(region, ants):
    img = np.zeros((1280, 1024, 3), dtype=np.uint8)

    for a in ants:
        for pt in a['cont']:
            img[pt[1]][pt[0]][2] = 255

    for pt in region['cont']:
        img[pt[1]][pt[0]][1] = 255

    return img


def move_ant(ant, tx, ty, th):
    #if ant['min_x'] + tx < 0:
    #    return False, ant
    #if ant['min_y'] + ty < 0:
    #    return False, ant
    #
    ##TODO upper image bound
    #
    #ant['min_x'] += tx
    #ant['min_y'] += ty

    rot = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
    for i in range(len(ant['cont'])):
        pt = ant['cont'][i]
        pt2 = np.array([pt[0] - ant['x'], pt[1] - ant['y']])
        pt2 = np.dot(rot, pt2.reshape(2, 1))

        ant['cont'][i][0] = pt2[0][0] + ant['x'] + tx
        ant['cont'][i][1] = pt2[1][0] + ant['y'] + ty

    ant['x'] += tx
    ant['y'] += ty

    return True


def trans_ant(ant, t, rot):
    #TODO upper image bound
    #ant['min_x'] += t[0]
    #ant['min_y'] += t[1]

    for i in range(len(ant['cont'])):
        pt = ant['cont'][i]
        pt2 = np.array([pt[0], pt[1]])
        pt2 = np.dot(rot, pt2.reshape(2, 1))

        ant['cont'][i][0] = pt2[0][0] + t[0]
        ant['cont'][i][1] = pt2[1][0] + t[1]

    ant['x'] += t[0]
    ant['y'] += t[1]

    return True


def opt_ant(rpts, apts, a):
    x = []
    y = []

    p = np.array([0, 0])
    q = np.array([0, 0])
    for pts in apts:
        x.append(pts[0])
        y.append(pts[1])
        p += pts[0]
        q += pts[1]

    for pts in rpts:
        if pts[2] == a['id']:
            x.append(pts[0])
            y.append(pts[1])
            p += pts[0]
            q += pts[1]


    x = np.array(x)
    y = np.array(y)

    p /= float(x.shape[0])
    q /= float(y.shape[0])

    #centering
    for i in range(x.shape[0]):
        x[i] -= p
        y[i] -= q


    s = np.dot(x.transpose(), y)

    U, _, V = np.linalg.svd(s)

    middle = np.array([[1, 0], [0, np.linalg.det(np.dot(V.T, U.T))]])
    R = np.dot(V.T, middle)
    R = np.dot(R, U.T)


    #t = np.dot(R, q - p)
    t = q - np.dot(R, p)

    #t = q-p
    #th = 0

    return t, R


def iteration(region, ants):
    #rpts, rscore = region_boundary_cover(region, ants)
    rpts, rscore = region_boundary_cover(region, ants)
    ascores = 0
    for a in ants:
        apts, s = ant_boundary_cover(region, a)
        ascores += s

        trans, rot = opt_ant(rpts, apts, a)
        trans_ant(a, trans, rot)

        #print "trans: ", trans

    return rscore, ascores


def prepare_ants(ants):
    out = []
    for a in ants:
        r = a.region

        cont, img, img_c, min_r, min_c = get_contour(r)

        th = count_theta(r)


        ant = {'id': 0, 'x': r['cx'], 'y': r['cy'], 'theta': th, 'cont': cont, 'min_y': min_r, 'min_x': min_c}
        out.append(ant)

    return out


def test_convergence(history):
    if len(history) < 2:
        return False

    if abs(history[-2][0] - history[-1][0]) < 0.01 and abs(history[-2][1] - history[-1][1]) < 0.01:
        return True

    return False


def solve(region, exp_ants, max_iterations=30, debug=False):
    ants = prepare_ants(exp_ants)

    history = []
    for i in range(max_iterations):
        rscore, ascore = iteration(region, ants)
        history.append([rscore, ascore])
        print i, rscore, ascore
        if test_convergence(history):
            im = draw_situation(region, ants)
            cv2.imshow("s", im)
            cv2.waitKey(0)
            break

    return ants


def main():
    region = load_region()
    ants = load_ants()

    move_ant(ants[0], 20, 5, 1)
    move_ant(ants[1], 100, -11, -0.35)

    #move_ant(ants[0], 0, 0, 0)
    #move_ant(ants[1], 1, -1, -0.35)

    #move_ant(ants[0], -200, 490, 1)
    #move_ant(ants[1], -210, 490, -0.35)

    solve(region, ants)

    #start = time.time()
    #for i in range(50):
    #    im = draw_situation(region, ants)
    #    cv2.imshow("s", im)
    #    cv2.waitKey(0)
    #    rscore, ascore = iteration(region, ants)
    #    print i, rscore, ascore
    #
    #
    #end = time.time()
    #print end - start

    return

if __name__ == '__main__':
    main()