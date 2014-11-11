__author__ = 'filip@naiser.cz'

import pickle
import numpy as np
import sys
import math
import os
from scipy import ndimage

import cv2
import matplotlib.pyplot as plt

import my_utils


dist_thresh = 4
in_debug = True

cont_weight = None
bounds = None

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

    img = np.zeros((rows+1, cols+1), dtype=np.uint8)

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
    try:
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    except:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt = []

    for c in contours:
        for pt in c:
            cnt.append(pt)

    img_cont = np.zeros((rows+1, cols+1), dtype=np.uint8)
    pts = []
    for p in cnt:
        img_cont[p[0][1]][p[0][0]] = 255
        pts.append([p[0][0] + min_c, p[0][1] + min_r])

    #cv2.imshow("test", img2)
    #cv2.waitKey(0)

    return pts, img, img_cont, min_r, min_c


def e_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def nearest_p_on_ant_bound_new(pt, ants):
    best_dist = 1000000-1 #value lower than returned from nearest_pt
    best_ant = -1
    best = [-1, -1]

    for a_id in range(len(ants)):
        apt, d = nearest_pt_transform(ants[a_id], pt[0], pt[1])

        if d < best_dist:
            best = apt
            best_dist = d
            best_ant = a_id

    return best, best_dist, best_ant

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
    x = apt[0] - r['min_c']
    y = apt[1] - r['min_r']
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
        #apt, dist, best_ant = nearest_p_on_ant_bound(pt, ants)
        #print pt
        #print apt, dist, best_ant
        apt, dist, best_ant = nearest_p_on_ant_bound_new(pt, ants)
        #print apt, dist, best_ant


        if dist < 1000000-1:
            pairs.append([apt, pt, best_ant])
            s += dist

    return pairs, s


def ant_boundary_cover(region, ant):
    s = 0
    pairs = []
    for apt in ant['cont']:
        if not is_inside_region(apt, region):
            pt, dist = nearest_pt(region, apt[0], apt[1])
            #print apt
            #print "nearest: ", pt, dist
            #pt, dist = nearest_p_on_region_bound(apt, region)
            #print "orig: ", pt, dist
            if dist < 100000:
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

    rx = np.zeros(len(region['cont']))
    ry = np.zeros(len(region['cont']))
    i = 0
    for pt in region['cont']:
        x[i] = pt[0]
        y[i] = pt[1]

        i += 1



    fig = plt.figure(figsize=(4, 6.5))
    #fig.set_figheight(3)
    #plt.xlim(715, 795)
    #plt.ylim(280, 370)
    reg = plt.scatter(x, y, color='#777777', s=35, edgecolor='k')
    #plt.scatter(rx, ry, color='purple', s=15)

    colors = ['magenta', 'cyan', 'yellow', 'blue', 'cyan']
    legends = []
    for a_id in range(len(ants)):
        a = ants[a_id]
        x = np.zeros(len(a['cont']))
        y = np.zeros(len(a['cont']))

        i = 0
        for pt in a['cont']:
            x[i] = pt[0]
            y[i] = pt[1]
            i += 1

        leg = plt.scatter(x, y, color=colors[a_id], s=35, edgecolor='black')
        legends.append(leg)




    if in_debug:
        #plt.legend((reg, legends[0], legends[1], legends[2]), ('Region points', 'Ant1 points', 'Ant2 points', 'Ant3 points'), scatterpoints=1,
        #   loc='top right',
        #   ncol=4,
        #   fontsize=13)

        #fig.set_figheight(3)
        #plt.grid()
        plt.ion()
        plt.show()
        #plt.axis([720, 790, 280, 365])
        #plt.axis('equal')
        #plt.xlim()
        #plt.ylim()
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

    img_cont = np.zeros((ant['img_cont'].shape[0], ant['img_cont'].shape[1]), dtype=np.uint8)
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


    #for transformation to original ant distance_map
    t, th = get_t_for_points2(ant)
    rot = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]

    ant['global_t'][0] = t[0]
    ant['global_t'][1] = t[1]

    ant['global_rot'] = rot

    return True


def opt_ant(rpts, apts, a_id, i):
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
            #if len(pts) > 3:
            #    weight = pts[3]
            #else:

            weight = 1

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
    k = 0.1

    pairs = []
    weights = np.zeros(len(region['cont'])*len(ants))

    i = 0
    pt_id = 0
    for pt in region['cont']:
        #dists = np.ones(len(ants))*1000
        #best_pts = np.zeros((len(ants), 2))

        best_dist = 100000

        for a_id in range(len(ants)):
            a = ants[a_id]
            best_d = 10000
            best_pt = [-1, -1]
            for apt in a['cont']:
                d = e_dist(apt, pt)

                if d < best_d:
                    best_d = d
                    best_pt = [apt[0], apt[1]]

                if d < best_dist:
                    best_dist = d

                #best_pts[a_id][0] = apt[0]
                #best_pts[a_id][1] = apt[1]

            if best_d < 10:
                pairs.append([best_pt, pt, a_id, math.exp(-k*best_d)*cont_weight[pt_id]])
                #weights[i] = math.exp(-k*best_d)
                i += 1

        #min_id = np.argmin(dists)
        s += best_dist

        pt_id += 1

        #if dists[min_id] < thresh:
        #    apt = [best_pts[min_id][0], best_pts[min_id][1]]
        #    pairs.append([apt, pt, min_id])
        #    weights.append(1)
        #else:
        #    for a_id in range(len(ants)):
        #        apt = [best_pts[a_id][0], best_pts[a_id][1]]
        #        pairs.append([apt, pt, a_id])
        #        weights.append(val)

    return pairs, s


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
        #rpts, rscore = region_2_ant_plus_weights(region, ants)

        #print rpts
        #plt.waitforbuttonpress(0)

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
            cont, img, img_cont, min_r, min_c = get_contour(r)
        else:
            cont = a.state.contour
            img = a.state.region['img']
            min_c = sys.maxint
            max_c = 0
            min_r = sys.maxint
            max_r = 0

            for pt in cont:
                p = [round(pt[0]), round(pt[1])]
                if min_c > p[0]:
                    min_c = p[0]
                if max_c < p[0]:
                    max_c = p[0]
                if min_r > p[1]:
                    min_r = p[1]
                if max_r < p[1]:
                    max_r = p[1]

            img_cont = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=np.uint8)
            for pt in cont:
                img_cont[int(round(pt[1])) - min_r][int(round(pt[0])) - min_c] = 255

        th = a.state.theta

        ant = {'id': a_id, 'x': a.state.position.x, 'y': a.state.position.y, 'theta': th, 'cont': cont,
               'min_r': min_r, 'min_c': min_c,
               'img': img, 'img_cont': img_cont,
               'head_start': my_utils.Point(a.state.head.x, a.state.head.y),
               'back_start': my_utils.Point(a.state.back.x, a.state.back.y),
               'head_end': my_utils.Point(a.state.head.x, a.state.head.y),
               'back_end': my_utils.Point(a.state.back.x, a.state.back.y), 'x_old': a.state.position.x,
               'y_old': a.state.position.y, 'a': r['a'], 'b': r['b'], 'area': r['area'], 'maxI': r['maxI'],
               'sxy': r['sxy'], 'sxx': r['sxx'], 'syy': r['syy'],
               'global_t': [0, 0],
               'global_rot': np.array([[1, 0], [0, 1]])}

        if "points" not in r:
            ant['points'] = get_points(r)
        else:
            ant['points'] = r['points']

        out.append(ant)

    return out


def prepare_region(exp_region, points):
    cont, img, img_cont, min_r, min_c = get_contour(exp_region, points)

    region = {'cont': cont, 'min_r': min_r, 'min_c': min_c, 'img': img, 'img_cont': img_cont}
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


def get_t_for_points(a):
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
    #print th * 57.3, th2 * 57.3
    #print t

    rot = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]

    return t, rot


def get_t_for_points2(a):
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

    return t, th



def trans_region_points(ants):
    for a in ants:
        t, rot = get_t_for_points(a)

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
        pt, d = nearest_pt(region, apt[0], apt[1])
        if d < dist_thresh:
            score += 1
        elif is_inside_region(apt, region):
            score += 0
        else:
            score -= 1

    return score / float(len(ant['cont']))


def move_unsettled(ants, region, params):
    stable_thresh = 0.75

    stable = np.ones(len(ants)) * [False]
    scores = np.zeros(len(ants))

    stable_num = 0
    for a_id in range(len(ants)):
        a = ants[a_id]
        s = get_ant_stability(a, region)
        scores[a_id] = s
        #print a_id, "stability: ", s
        if s > stable_thresh:
            stable[a_id] = True
            stable_num += 1

    if stable_num == 0:
        return

    #kdyz jsou oba moc dobri, ale region to nevyvetluje, pak je mozne, ze maji velky prekryv a jeden se proste vybere a pusti se na nej nevysvetlena oblast.
    #if stable_num == len(ants):
    #    stable[0] = False
    #    scores[0] = 0.74

    indexes = np.argsort(scores)
    ascores = 0

    converged = np.ones(len(ants)) * [False]
    hist = np.zeros((len(ants), 3))
    for a_id in range(len(ants)):
        hist[a_id][0] = ants[a_id]['x']
        hist[a_id][0] = ants[a_id]['y']
        hist[a_id][0] = 1000 #... ants['theta'] asi jeste neexistuje

    #for i in range(7):
    for i in range(5):
        changed = False
        #print i
        for id in indexes:
            if converged[id]:
                continue
            if stable[id]:
                continue

            changed = True

            a = ants[id]
            apts, s = ant_boundary_cover(region, a)
            ascores += s

            #TODO
            #max_dist = params.avg_ant_axis_a
            rpts, rscore = region_boundary_cover2(region, ants, stable, 20)

            trans, rot = opt_ant(rpts, apts, id, i)
            trans_ant(a, trans, rot)

            if test_position_convergence(a, hist[id]):
                converged[id] = True

            hist[id][0] = a['x']
            hist[id][1] = a['y']
            hist[id][2] = a['theta']
            #print a['x'], a['y'], a['theta']

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
    thresh = 1
    thresh_th = 3
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

        apt, d = nearest_pt_transform(a, pt[0], pt[1])
        #d = e_dist(apt, pt)
        if d < best_dist and (not stable[a_id] or d < dist_thresh):
            best = apt
            best_dist = d
            best_ant = a_id

    #for a_id in range(len(ants)):
    #    a = ants[a_id]
    #    for apt in a['cont']:
    #        d = e_dist(apt, pt)
    #        if d < best_dist and (not stable[a_id] or d < dist_thresh):
    #            best = apt
    #            best_dist = d
    #            best_ant = a_id
    #
    #return best, best_dist, best_ant

    #best_dist = max_dist
    #best_ant = -1
    #best = [-1, -1]
    #
    #for a_id in range(len(ants)):
    #    a = ants[a_id]
    #    for apt in a['cont']:
    #        d = e_dist(apt, pt)
    #        if d < best_dist and (not stable[a_id] or d < dist_thresh):
    #            best = apt
    #            best_dist = d
    #            best_ant = a_id
    #
    #
    #
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
    plt.scatter(x1, y1, c=weights, s=35, edgecolor='black', cmap=plt.cm.hot)
    cbar = plt.colorbar(ticks = [min(weights), max(weights)])
    cbar.ax.set_yticklabels(['Low', 'High'])
    cbar.set_label(r'weights')
    plt.show()
    plt.waitforbuttonpress(0)

    return weights


def nearest_pt(o, x, y):
    global bounds;

    x = round(x)
    y = round(y)
    if x < bounds['min_c_minus_margin'] or x >= bounds['max_c_plus_margin'] or y < bounds['min_r_minus_margin'] or y >= bounds['max_r_plus_margin']:
        new_x = x
        new_y = y
        if new_x < bounds['min_c_minus_margin']:
            new_x = bounds['min_c_minus_margin']
        if new_x > bounds['max_c_plus_margin']:
            new_x = bounds['max_c_plus_margin']

        if new_y < bounds['min_r_minus_margin']:
            new_y = bounds['min_r_minus_margin']
        if new_y > bounds['max_r_plus_margin']:
            new_y = bounds['max_r_plus_margin']

        d = math.sqrt((x-new_x)**2 + (y-new_y)**2)
        #print "OUT OF BOUNDS", x, y
        return [new_x, new_y], d

    r = y - bounds['min_r_minus_margin']
    c = x - bounds['min_c_minus_margin']

    d = o['dist_map'][r, c]
    pt = o['dist_map_labels'][:, r, c]

    nearest = [pt[1] + bounds['min_c_minus_margin'], pt[0] + bounds['min_r_minus_margin']]
    return nearest, d


def nearest_pt_transform(o, x, y):
    t = np.array(o['global_t'])
    rot = np.array(o['global_rot'])

    ox = np.array([float(o['x']), float(o['y'])])
    ox_old = np.array([float(o['x_old']), float(o['y_old'])])

    pt = np.array([float(x), float(y)])
    pt -= ox
    pt = np.dot(rot.T, pt.reshape(2, 1))
    pt = pt.reshape(2)
    pt += ox
    pt -= t

    nearest, d = nearest_pt(o, pt[0], pt[1])

    pt = np.array(nearest)
    pt = np.array([float(pt[0]), float(pt[1])])
    pt -= ox_old
    pt = np.dot(rot, pt.reshape(2, 1))
    pt = pt.reshape(2)
    pt += ox_old
    pt += t

    #pt = np.array([x, y])
    #pt -= ox
    #pt = np.dot(rot.T, pt.reshape(2, 1))
    #pt = pt.reshape(2)
    #pt += ox
    #pt -= t
    #
    #nearest, d = nearest_pt(o, pt[0], pt[1])
    #
    #pt = np.array(nearest)
    #pt -= ox
    #pt = np.dot(rot, pt.reshape(2, 1))
    #pt = pt.reshape(2)
    #pt += ox
    #pt += t

    nearest = [pt[0], pt[1]]

    #pt2 = np.array([x - o['x'], y - o['y']])
    #pt2 = np.dot(o['global_rot'], pt2.reshape(2, 1))
    #
    #new_x = pt2[0][0] + o['x_old'] + o['global_t'][0]
    #new_y = pt2[1][0] + o['y_old'] + o['global_t'][1]
    #
    #nearest, d = nearest_pt(o, new_x, new_y)
    #
    #nearest[0] -= o['x_old']
    #nearest[1] -= o['y_old']
    #
    #nearest = np.dot(np.array(o['global_rot']).T, np.array(nearest).reshape(2, 1))
    #nearest = [nearest[0][0] + o['x_old'] + t[0], nearest[1][0] + o['y_old'] + t[1]]

    return nearest, d


def count_distance_maps(o):
    global bounds;
    img = np.zeros((bounds['max_r'] - bounds['min_r'] + 2 * bounds['margin'],
                    bounds['max_c'] - bounds['min_c'] + 2 * bounds['margin']), dtype='uint8')

    r = o['min_r'] - bounds['min_r'] + bounds['margin']
    c = o['min_c'] - bounds['min_c'] + bounds['margin']
    img[r:r+o['img_cont'].shape[0], c:c+o['img_cont'].shape[1]] = o['img_cont']
    img = np.invert(img)

    o['dist_map'], o['dist_map_labels'] = ndimage.distance_transform_edt(img, return_indices=True)


def prepare_distance_maps(r, ants):
    min_rs = [r['min_r']]
    min_cs = [r['min_c']]
    max_rs = [r['min_r'] + r['img_cont'].shape[0]]
    max_cs = [r['min_c'] + r['img_cont'].shape[1]]

    for a in ants:
        min_rs.append(a['min_r'])
        min_cs.append(a['min_c'])
        max_rs.append(a['min_r'] + a['img_cont'].shape[0])
        max_cs.append(a['min_c'] + a['img_cont'].shape[1])

    global bounds
    margin = 5
    bounds = {'min_r': np.min(min_rs), 'min_c': np.min(min_cs),
              'max_r': np.max(max_rs), 'max_c': np.max(max_cs),
              'min_r_minus_margin': np.min(min_rs) - margin,
              'min_c_minus_margin': np.min(min_cs) - margin,
              'max_r_plus_margin': np.max(max_rs) + margin,
              'max_c_plus_margin': np.max(max_cs) + margin,
              'margin': margin}

    count_distance_maps(r)
    for a in ants:
        count_distance_maps(a)


#def distance_map_test(r):
#    img = r['img_cont']
#    img = img.reshape(img.shape[0], img.shape[1])
#    img = np.invert(img)
#
#    start = time.time()
#    edt, inds = ndimage.distance_transform_edt(img, return_indices=True)
#    #edt, inds = cv2.distanceTransformWithLabels(img, cv.CV_DIST_L2, 3, labels=True)
#    print "TIME: ", time.time() - start
#    print edt.shape
#
#    pt, d = nearest_pt(edt, inds, [0, 0])
#    print pt, d


def save_input(frame, exp_region, points, ants, params):
    path = params.dumpdir
    try:
        os.makedirs(path+str(frame))
    except:
        pass


    cont, img, img_cont, min_r, min_c = get_contour(exp_region, points)
    cv2.imwrite(path+str(frame)+"/region.png", img)
    cv2.imwrite(path+str(frame)+"/region_cont.png", img_cont)
    cv2.imwrite(path+str(frame)+"/frame.png", params._img)
    cv2.imwrite(path+str(frame)+"/frame_crop.png", params._img[min_r:min_r+img.shape[0], min_c:min_c+img.shape[1], :])

    for a in ants:
        cv2.imwrite(path+str(frame)+"/ant_"+str(a['id'])+".png", a['img'])
        cv2.imwrite(path+str(frame)+"/ant_"+str(a['id'])+"_cont.png", a['img_cont'])


#def solve(exp_region, points, ants_ids, exp_ants, params, img_shape, max_iterations=30, debug=False):
def solve(exp_region, points, ants_ids, exp_ants, params, img_shape, max_iterations=10, debug=False):
    global in_debug
    global bounds


    run = False
    if params is not None:
        run = True
        in_debug = False

    debug = False
    in_debug = False

    if run:
        frame = params.frame
        ants = prepare_ants(ants_ids, exp_ants)
        # save_input(params.frame, exp_region, points, ants, params)
        region = prepare_region(exp_region, points)

        prepare_distance_maps(region, ants)

        pack = [ants, region, bounds]

        if debug:
            afile = open(params.dumpdir+"/split_by_cont/"+str(frame)+"_"+str(ants_ids[0])+".pkl", "wb")
            pickle.dump(pack, afile)
            afile.close()
    else:
        dir = os.path.expanduser('~/dump/eight')
        afile = open(dir+"/split_by_cont/209_1.pkl", "rb")
        #afile = open(dir+"/split_by_cont/60_4.pkl", "rb")
        #afile = open(dir+"/split_by_cont/452_0.pkl", "rb")
        #afile = open(dir+"/split_by_cont/727_0.pkl", "rb")
        #afile = open(dir+"/split_by_cont/695_0.pkl", "rb")
        #afile = open(dir+"/split_by_cont/1143_1.pkl", "rb")
        #afile = open(dir+"/split_by_cont/209_3.pkl", "rb")
        #afile = open(dir+"/split_by_cont/703_0.pkl", "rb")
        #afile = open(dir+"/split_by_cont/704_0.pkl", "rb")
        #afile = open(dir+"/split_by_cont/673_1.pkl", "rb")
        #afile = open(dir+"/split_by_cont/690_0.pkl", "rb")
        #afile = open(dir+"/split_by_cont/691_0.pkl", "rb")
        pack = pickle.load(afile)
        afile.close()
        ants = pack[0]
        region = pack[1]

        bounds = pack[2]

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
        #region = regions[11]
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
        #im = draw_situation(region, ants, img_shape)
        #cv2.imshow('test', im)
        #cv2.imwrite(params.dumpdir+'/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_a.jpg', im)

    #if not run:
    #    plot_situation(region, ants)
    #    #move_unsettled(ants, region)
    #    #im = draw_situation(region, ants, img_shape)
    #    #cv2.imshow('test', im)
    #    #cv2.waitKey(0)

    #global cont_weight
    #cont_weight = get_contour_weights(region)

    #distance_map_test(region)
    #return

    #init(ants, region)
    #iteration(region, ants, 0)
    #plot_situation(region, ants)

    #plot_situation(region, ants)
    #plot_situation(region, ants)
    move_unsettled(ants, region, params)

    done = False
    for i in range(max_iterations):
        if not run:
            plot_situation(region, ants)

            #im = draw_situation(region, ants, img_shape)
            #cv2.imshow('test', im)
            #cv2.waitKey(0)

        rscore, ascore = iteration(region, ants, i)
        history.append([rscore, ascore])
        #print i, rscore, ascore, rscore+ascore
        if test_convergence(history):
            done = True
            break


    trans_region_points(ants)
    count_overlaps(ants, img_shape)
    count_crossovers(ants, region)

    #im = draw_situation(region, ants, img_shape, fill=True)
    if run and debug:
        fig = plt.figure()
        plot_situation(region, ants)
        if not done:
            #cv2.imwrite(params.dumpdir+'/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_e.jpg', im)
            plt.savefig(params.dumpdir+'/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_e.png')
        else:
            #cv2.imwrite(params.dumpdir+'/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_en.jpg', im)
            plt.savefig(params.dumpdir+'/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_en.png')

        plt.close(fig)

    #if not run:
    #    im = draw_situation(region, ants, img_shape, fill=True)
        #cv2.imshow('test', im)
        #cv2.waitKey(0)


    #im = draw_situation(region, ants)
    #cv2.imshow("test", im)

    return ants


def main():
    solve(None, None, None, None, None, (1024, 1280), debug=True)

    return

if __name__ == '__main__':
    main()