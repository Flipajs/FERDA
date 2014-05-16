__author__ = 'flipajs'

import pickle
import numpy as np
import cv2
import sys
import math
import my_utils
import os
import time
import matplotlib.pyplot as plt

dist_thresh = 4
in_debug = True

def get_points(region):
    points = []
    for r in region['rle']:
        for c in range(r['col1'], r['col2'] + 1):
            points.append([c, r['line']])

    return points


def get_contour(region, data=None):
    min_c = 100000
    max_c = 0
    min_r = 100000
    max_r = 0

    if data == None:
        min_r = region['rle'][0]['line']
        max_r = region['rle'][-1]['line']
        for r in region['rle']:
            if min_c > r['col1']:
                min_c = r['col1']
            if max_c < r['col2']:
                max_c = r['col2']
    else:
        for pt in data:
            if min_c > pt[0]:
                min_c = pt[0]
            if max_c < pt[0]:
                max_c = pt[0]
            if min_r > pt[1]:
                min_r = pt[1]
            if max_r < pt[1]:
                max_r = pt[1]

    rows = max_r - min_r
    cols = max_c - min_c

    img = np.zeros((rows+1, cols+1, 1), dtype=np.uint8)

    if data == None:
        for r in region['rle']:
            row = r['line'] - min_r
            col1 = r['col1'] - min_c
            col2 = r['col2'] - min_c
            img[row][col1:col2+1] = 255
    else:
        for pt in data:
            row = pt[1] - min_r
            col1 = pt[0] - min_c
            #col2 = r['col2'] - min_c
            img[row][col1] = 255


    #cv2.imshow("img", img)

    ret,thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt = []

    for c in contours:
        for pt in c:
            cnt.append(pt)

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


def e_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def nearest_p_on_ant_bound(pt, ants):
    best_dist = sys.maxint
    best_ant = -1
    best = [-1, -1]

    for a_id in range(len(ants)):
        a = ants[a_id]
        for apt in a['cont']:
            d = e_dist(apt, pt)
            if d < best_dist:
                best = apt
                best_dist = d
                best_ant = a_id

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

def ant_boundary_cover2(region, ant):
    s = 0
    pairs = []
    for apt in ant['cont']:
        #if not is_inside_region(apt, region):
        pt, dist = nearest_p_on_region_bound(apt, region)
        pairs.append([apt, pt])
        s += dist

    return pairs, s


def draw_situation(region, ants, img_shape, fill=False):
    img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)

    if fill:
        for a_id in range(len(ants)):
            a = ants[a_id]
            c = 50*a_id
            for pt in a['points']:
                img[pt[1]][pt[0]][2] = 255-c
                img[pt[1]][pt[0]][1] = c
    else:
        for a_id in range(len(ants)):
            a = ants[a_id]
            c = 50*a_id
            for pt in a['cont']:
                img[pt[1]][pt[0]][2] = 255-c
                img[pt[1]][pt[0]][1] = c

    for pt in region['cont']:
        img[pt[1]][pt[0]][0] = 255
        img[pt[1]][pt[0]][1] = 255
        img[pt[1]][pt[0]][2] = 255

    return img


def plot_situation(region, ants):
    if in_debug:
        plt.close()
    #plt.ion()
    plt.ioff()

    x = np.zeros(len(region['cont']))
    y = np.zeros(len(region['cont']))

    i = 0
    for pt in region['cont']:
        x[i] = pt[0]
        y[i] = pt[1]
        i += 1

    plt.scatter(x, y, color='black', s=25)

    colors = ['red', 'green', 'blue', 'yellow', 'cyan']
    for a_id in range(len(ants)):
        a = ants[a_id]
        x = np.zeros(len(a['cont']))
        y = np.zeros(len(a['cont']))

        i = 0
        for pt in a['cont']:
            x[i] = pt[0]
            y[i] = pt[1]
            i += 1

        plt.scatter(x, y, color=colors[a_id], s=15, edgecolor='black')




    plt.axis('equal')
    if in_debug:
        plt.ion()
        plt.show()
        plt.waitforbuttonpress(0)


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

    pt2 = np.array([ant['x'], ant['y']])
    pt2 = np.dot(rot, pt2.reshape(2, 1))
    ant['x'] = pt2[0][0] + t[0]
    ant['y'] = pt2[1][0] + t[1]

    pt2 = np.array([ant['head_end'].x, ant['head_end'].y])
    pt2 = np.dot(rot, pt2.reshape(2, 1))
    ant['head_end'].x = pt2[0][0] + t[0]
    ant['head_end'].y = pt2[1][0] + t[1]
    
    pt2 = np.array([ant['back_end'].x, ant['back_end'].y])
    pt2 = np.dot(rot, pt2.reshape(2, 1))
    ant['back_end'].x = pt2[0][0] + t[0]
    ant['back_end'].y = pt2[1][0] + t[1]


    help = np.dot(rot, np.array([1, 0]).reshape(2, 1))
    ant['theta'] = math.atan2(help[1], help[0])


    return True


def opt_ant(rpts, apts, a_id, i, weights=None):
    x = []
    y = []

    dist_weight = True
    if i < 3:
        dist_weight = False

    w_sum = 0
    w = []

    p = np.array([0., 0.])
    q = np.array([0., 0.])
    for pts in apts:
        x.append([float(i) for i in pts[0]])
        y.append([float(i) for i in pts[1]])
        weight = np.linalg.norm(np.array(pts[0]) - np.array(pts[1])) * 2
        w_sum += weight
        w.append(weight)
        p += np.multiply(pts[0], weight)
        q += np.multiply(pts[1], weight)

    j = 0
    for pts in rpts:
        if pts[2] == a_id:
            x.append([float(i) for i in pts[0]])
            y.append([float(i) for i in pts[1]])
            #if weights is not None:
            #    weight = weights[j]
            #else:
            #    weight = 1
            if dist_weight:
                weight = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
            else:
                weight = 1

            w_sum += weight
            w.append(weight)
            p += np.multiply(pts[0], weight)
            q += np.multiply(pts[1], weight)
        j += 1

    if len(x) + len(y) == 0:
        print "In OPT_ANT returned zero t and I rot"
        return np.array([0, 0]), np.array([[1, 0], [0, 1]])

    x = np.array(x)
    y = np.array(y)


    #p /= float(len(apts) * weight + len(rpts))
    #q /= float(len(apts) * weight + len(rpts))
    p /= w_sum
    q /= w_sum

    #centering
    for i in range(x.shape[0]):
        x[i] -= p
        y[i] -= q

    #w = np.array([1.] * x.shape[0])

    #w[0:len(apts)] = weight

    W = np.diag(np.array(w))

    s = np.dot(x.transpose(), W)
    s = np.dot(s, y)

    U, _, V = np.linalg.svd(s)

    middle = np.array([[1, 0], [0, np.linalg.det(np.dot(V.T, U.T))]])
    R = np.dot(V.T, middle)
    R = np.dot(R, U.T)


    #t = np.dot(R, q - p)
    t = q - np.dot(R, p)

    #t = q-p
    #th = 0

    return t, R


def region_2_ant_plus_weights(region, ants):
    s = 0
    thresh = 3
    val = 1 / float(len(ants))
    pairs = []
    weights = []
    for pt in region['cont']:
        dists = np.ones(len(ants))*1000
        best_pts = np.zeros((len(ants), 2))

        for a_id in range(len(ants)):
            a = ants[a_id]
            for apt in a['cont']:
                d = e_dist(apt, pt)
                if d < dists[a_id]:
                    dists[a_id] = d
                    best_pts[a_id][0] = apt[0]
                    best_pts[a_id][1] = apt[1]

        min_id = np.argmin(dists)
        s += dists[min_id]
        if dists[min_id] < thresh:
            apt = [best_pts[min_id][0], best_pts[min_id][1]]
            pairs.append([apt, pt, min_id])
            weights.append(1)
        else:
            for a_id in range(len(ants)):
                apt = [best_pts[a_id][0], best_pts[a_id][1]]
                pairs.append([apt, pt, a_id])
                weights.append(val)

    return pairs, weights, s


def iteration(region, ants, i):
    #rpts, rscore = region_boundary_cover(region, ants)
    ascores = 0
    rscore = 0

    apts_list = []
    ant_scores = np.zeros(len(ants))

    i = 0
    for a in ants:
        apts, ant_scores[i] = ant_boundary_cover(region, a)
        apts_list.append(apts)
        i += 1

    #reverse order
    indexes = np.argsort(ant_scores[::-1])

    for id in indexes:
        a = ants[id]
        ascores += ant_scores[id]

        rpts, rscore = region_boundary_cover(region, ants)
        #rpts, weights, rscore = region_2_ant_plus_weights(region, ants)

        trans, rot = opt_ant(rpts, apts_list[id], id, i)
        trans_ant(a, trans, rot)

    #
    #chosen_ant = None
    #worst_apts = None
    #worts_s = -1
    #for a in ants:
    #    apts, s = ant_boundary_cover(region, a)
    #    if s > worts_s:
    #        worts_s = s
    #        worst_apts = apts
    #        chosen_ant = a

    #for a in ants:
    #a = ants[1]
    #rpts, rscore = region_boundary_cover(region, ants)
    #apts, s = ant_boundary_cover(region, a)
    #ascores += s
    #
    #trans, rot = opt_ant(rpts, apts, a, i)
    #trans_ant(a, trans, rot)

    #print "trans: ", trans

    return rscore, ascores


def prepare_ants(ant_ids, ants):
    out = []
    for a_id in ant_ids:
        a = ants[a_id]
        r = a.state.region

        #if "rle" not in r:
        #    continue

        if a.state.contour == None:
            cont, _, _, min_r, min_c = get_contour(r)
        else:
            cont = a.state.contour

        th = a.state.theta

        ant = {'id': a_id, 'x': a.state.position.x, 'y': a.state.position.y, 'theta': th, 'cont': cont}
        ant['head_start'] = my_utils.Point(a.state.head.x, a.state.head.y)
        ant['back_start'] = my_utils.Point(a.state.back.x, a.state.back.y)

        ant['head_end'] = my_utils.Point(a.state.head.x, a.state.head.y)
        ant['back_end'] = my_utils.Point(a.state.back.x, a.state.back.y)

        ant['x_old'] = a.state.position.x
        ant['y_old'] = a.state.position.y

        ant['a'] = r['a']
        ant['b'] = r['b']

        ant['area'] = r['area']
        ant['maxI'] = r['maxI']
        ant['sxy'] = r['sxy']
        ant['sxx'] = r['sxx']
        ant['syy'] = r['syy']
        if "points" not in r:
            ant['points'] = get_points(r)
        else:
            ant['points'] = r['points']

        out.append(ant)

    return out


def prepare_region(exp_region, points):
    cont, img, img_cont, min_r, min_c = get_contour(exp_region, points)

    region = {'cont': cont, 'y_min': min_r, 'x_min': min_c, 'img': img, 'img_cont': img_cont}
    return region


def test_convergence(history):
    thresh = 1
    if len(history) < 2:
        return False

    if abs(history[-2][0] - history[-1][0]) < thresh and abs(history[-2][1] - history[-1][1]) < thresh:
        return True

    #if history[-2][0] + history[-2][1] < history[-1][0] + history[-1][1]:
    #    return True

    return False


def trans_region_points(ants):
    for a in ants:
        x = a['back_start'].x
        y = a['back_start'].y

        xx = a['head_start'].x - x
        yy = a['head_start'].y - y

        th1 = math.atan2(yy, xx)
        
        x = a['back_end'].x
        y = a['back_end'].y

        xx = a['head_end'].x - x
        yy = a['head_end'].y - y

        th2 = math.atan2(yy, xx)

        th = th2-th1

        th2 = -th2
        if th2 < 0:
            th2 += math.pi

        a['theta'] = th2
        t = [a['x'] - a['x_old'], a['y'] - a['y_old']]
        print th * 57.3, th2 * 57.3
        print t

        rot = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]
        for i in range(len(a['points'])):
            pt = a['points'][i]
            pt2 = np.array([pt[0] - a['x_old'], pt[1] - a['y_old']])
            pt2 = np.dot(rot, pt2.reshape(2, 1))

            a['points'][i][0] = pt2[0][0] + a['x_old'] + t[0]
            a['points'][i][1] = pt2[1][0] + a['y_old'] + t[1]


def count_overlaps(ants, img_shape):
    img = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.uint8)

    for a in ants:
        for pt in a['points']:
            img[pt[1]][pt[0]] += 1

    for a in ants:
        counter = 0
        for pt in a['points']:
            if img[pt[1]][pt[0]] > 1:
                counter += 1

        a['overlap'] = counter / float(a['area'])


def count_crossovers(ants, region):
    for a_id in range(len(ants)):
        counter = 0
        for pt in ants[a_id]['points']:
            if not is_inside_region([int(pt[0]), int(pt[1])], region):
                counter += 1

        ants[a_id]['crossover'] = counter / float(ants[a_id]['area'])


def get_ant_stability(ant, region):
    score = 0
    for apt in ant['cont']:
        pt, d = nearest_p_on_region_bound(apt, region)
        if d < dist_thresh:
            score += 1
        elif is_inside_region(apt, region):
            score += 0
        else:
            score -= 1

    return score / float(len(ant['cont']))


def new_alg(ants, region):
    stable_thresh = 0.75

    stable = np.ones(len(ants)) * [False]
    scores = np.zeros(len(ants))

    stable_num = 0
    for a_id in range(len(ants)):
        a = ants[a_id]
        s = get_ant_stability(a, region)
        scores[a_id] = s
        print a_id, "stability: ", s
        if s > stable_thresh:
            stable[a_id] = True
            stable_num += 1

    if stable_num == 0:
        return

    #kdyz jsou oba moc dobri, ale region to nevyvetluje, pak je mozne, ze maji velky prekryv a jeden se proste vybere a pusti se na nej nevysvetlena oblast.

    indexes = np.argsort(scores)
    ascores = 0

    converged = np.ones(len(ants)) * [False]
    hist = np.zeros((len(ants), 3))
    for a_id in range(len(ants)):
        hist[a_id][0] = ants[a_id]['x']
        hist[a_id][0] = ants[a_id]['y']
        hist[a_id][0] = 1000 #... ants['theta'] asi jeste neexistuje

    for i in range(7):
        changed = False
        print i
        for id in indexes:
            if converged[id]:
                continue
            if stable[id]:
                continue

            changed = True

            a = ants[id]
            apts, s = ant_boundary_cover(region, a)
            ascores += s

            rpts, rscore = region_boundary_cover2(region, ants, stable, 20)

            trans, rot = opt_ant(rpts, apts, id, i)
            trans_ant(a, trans, rot)

            if test_position_convergence(a, hist[a_id]):
                converged[a_id] = True

            hist[a_id][0] = a['x']
            hist[a_id][1] = a['y']
            hist[a_id][2] = a['theta']
            print a['x'], a['y'], a['theta']

            if in_debug:
                plot_situation(region, ants)

        if not changed:
            break

def region_contour_nearest(region, ants, max_dist=10):
    s = 0
    pairs = []
    for pt in region['cont']:
        apt, dist, best_ant = region_2_ant_pt_assignment2(pt, ants, max_dist)
        if best_ant != -1:
            pairs.append([apt, pt, best_ant])
            s += dist


    #x1 = []
    #y1 = []
    #x2 = []
    #y2 = []
    #
    #i = 0
    #for pt_pair in pairs:
    #    if pt_pair[2] != 1:
    #        continue
    #
    #    x1.append(pt_pair[0][0])
    #    y1.append(pt_pair[0][1])
    #
    #    x2.append(pt_pair[1][0])
    #    y2.append(pt_pair[1][1])
    #    i += 1
    #
    #plt.close()
    #plt.scatter(x1, y1, color='green', s=15, edgecolor='black')
    #plt.scatter(x2, y2, color='yellow', s=15, edgecolor='black')
    #
    #
    #plt.axis('equal')
    #plt.show()
    #plt.waitforbuttonpress(0)

    return pairs, s


def init(ants, region):
    ascores = 0
    rscore = 0

    apts_list = []
    ant_scores = np.zeros(len(ants))

    i = 0
    for a in ants:
        apts, ant_scores[i] = ant_boundary_cover(region, a)
        apts_list.append(apts)
        i += 1

    #reverse order
    indexes = np.argsort(ant_scores[::-1])

    for id in indexes:
        a = ants[id]
        ascores += ant_scores[id]
        #rpts, rscore = region_contour_nearest(region, ants, max_dist=7)
        rpts, rscore = region_boundary_cover(region, ants)

        trans, rot = opt_ant(rpts, apts_list[id], id, i)
        trans_ant(a, trans, rot)

def test_position_convergence(a, history):
    thresh = 0.5
    thresh_th = 1
    x = history[0] - a['x']
    y = history[1] - a['y']
    th = abs(history[2] - a['theta'])
    if th > math.pi:
        th = th - math.pi

    if abs(x) < thresh and abs(y) < thresh and th*57 < thresh_th:
        return True

    return False


def region_boundary_cover2(region, ants, stable, max_dist=4):
    s = 0
    pairs = []
    for pt in region['cont']:
        apt, dist, best_ant = region_2_ant_pt_assignment(pt, ants, stable, max_dist)
        if best_ant != -1:
            pairs.append([apt, pt, best_ant])
            s += dist


    #x1 = []
    #y1 = []
    #x2 = []
    #y2 = []
    #
    #i = 0
    #for pt_pair in pairs:
    #    if pt_pair[2] != 1:
    #        continue
    #
    #    x1.append(pt_pair[0][0])
    #    y1.append(pt_pair[0][1])
    #
    #    x2.append(pt_pair[1][0])
    #    y2.append(pt_pair[1][1])
    #    i += 1
    #
    #plt.close()
    #plt.scatter(x1, y1, color='green', s=15, edgecolor='black')
    #plt.scatter(x2, y2, color='yellow', s=15, edgecolor='black')
    #
    #
    #plt.axis('equal')
    #plt.show()
    #plt.waitforbuttonpress(0)

    return pairs, s


def region_2_ant_pt_assignment(pt, ants, stable, max_dist):
    best_dist = max_dist
    best_ant = -1
    best = [-1, -1]

    for a_id in range(len(ants)):
        a = ants[a_id]
        for apt in a['cont']:
            d = e_dist(apt, pt)
            if d < best_dist and (not stable[a_id] or d < dist_thresh):
                best = apt
                best_dist = d
                best_ant = a_id



    return best, best_dist, best_ant


def region_2_ant_pt_assignment2(pt, ants, max_dist):
    best_dist = sys.maxint
    best_ant = -1
    best = [-1, -1]

    for a_id in range(len(ants)):
        a = ants[a_id]
        for apt in a['cont']:
            d = e_dist(apt, pt)
            if d < max_dist:
                best = apt
                best_dist = d
                best_ant = a_id



    return best, best_dist, best_ant

def get_contour_weights(region):
    step = 2
    c = np.array(region['cont'])
    l = len(c)
    max_dist = math.sqrt(32)

    x1 = []
    y1 = []

    weights = np.zeros(l)
    for i in range(l):
        pt1 = c[(i-2) % l]
        pt2 = c[(i+2) % l]

        d = np.linalg.norm(pt1 - pt2)
        weights[i] = d/max_dist

        x1.append(c[i][0])
        y1.append(c[i][1])

    plt.close()
    plt.scatter(x1, y1, color=weights, s=35, edgecolor='black', cmap=plt.cm.jet)
    plt.show()
    plt.waitforbuttonpress(0)

    return weights

def solve(exp_region, points, ants_ids, exp_ants, params, img_shape, max_iterations=30, debug=False):
    global in_debug
    run = False
    if params is not None:
        run = True
        in_debug = False

    if run:
        frame = params.frame
        ants = prepare_ants(ants_ids, exp_ants)
        region = prepare_region(exp_region, points)

        #if len(region['cont']) > 300:
        #    print "too big region"
        #    return []


        pack = [ants, region]

        afile = open(params.dumpdir+"/split_by_cont/"+str(frame)+"_"+str(ants_ids[0])+".pkl", "wb")
        pickle.dump(pack, afile)
        afile.close()
    else:
        dir = os.path.expanduser('~/dump/eight')
        #afile = open(dir+"/split_by_cont/695_0.pkl", "rb")
        #afile = open(dir+"/split_by_cont/1143_1.pkl", "rb")
        #afile = open(dir+"/split_by_cont/209_3.pkl", "rb")
        #afile = open(dir+"/split_by_cont/703_0.pkl", "rb")
        #afile = open(dir+"/split_by_cont/704_0.pkl", "rb")
        #afile = open(dir+"/split_by_cont/672_5.pkl", "rb")
        #afile = open(dir+"/split_by_cont/690_0.pkl", "rb")
        afile = open(dir+"/split_by_cont/691_0.pkl", "rb")
        pack = pickle.load(afile)
        afile.close()
        ants = pack[0]
        region = pack[1]

        #afile = open(dir+"/regions/695.pkl", "rb")
        ##afile = open(dir+"/regions/1143.pkl", "rb")
        #afile = open(dir+"/regions/209.pkl", "rb")
        ##afile = open(dir+"/regions/703.pkl", "rb")
        #
        #regions = pickle.load(afile)
        #afile.close()
        #
        #region = regions[1]
        ##region = regions[12]
        ##region = regions[11]
        ##region = regions[2]
        #points = get_points(region)
        #
        #region = prepare_region(exp_region, points)

    history = []

    if debug and run:
        fig = plt.figure()
        plot_situation(region, ants)
        plt.savefig(params.dumpdir+'/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_a.png')
        plt.close(fig)
        im = draw_situation(region, ants, img_shape)
        cv2.imshow('test', im)
        cv2.imwrite(params.dumpdir+'/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_a.jpg', im)

    #if not run:
    #    plot_situation(region, ants)
    #    #new_alg(ants, region)
    #    #im = draw_situation(region, ants, img_shape)
    #    #cv2.imshow('test', im)
    #    #cv2.waitKey(0)

    #get_contour_weights(region)

    #init(ants, region)
    #iteration(region, ants, 0)
    plot_situation(region, ants)
    new_alg(ants, region)

    done = False
    for i in range(max_iterations):
        if not run:
            plot_situation(region, ants)

            #im = draw_situation(region, ants, img_shape)
            #cv2.imshow('test', im)
            #cv2.waitKey(0)

        rscore, ascore = iteration(region, ants, i)
        history.append([rscore, ascore])
        print i, rscore, ascore
        if test_convergence(history):
            done = True
            break


    trans_region_points(ants)
    count_overlaps(ants, img_shape)
    count_crossovers(ants, region)

    im = draw_situation(region, ants, img_shape, fill=True)
    if run and debug:
        fig = plt.figure()
        plot_situation(region, ants)
        if not done:
            cv2.imwrite(params.dumpdir+'/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_e.jpg', im)
            plt.savefig(params.dumpdir+'/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_e.png')
        else:
            cv2.imwrite(params.dumpdir+'/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_en.jpg', im)
            plt.savefig(params.dumpdir+'/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_en.png')

        plt.close(fig)

    if not run:
        im = draw_situation(region, ants, img_shape, fill=True)
        cv2.imshow('test', im)
        cv2.waitKey(0)


    #im = draw_situation(region, ants)
    #cv2.imshow("test", im)

    return ants


def main():
    solve(None, None, None, None, None, (1024, 1280), debug=True)

    return

if __name__ == '__main__':
    main()