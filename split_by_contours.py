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


def draw_situation(region, ants, img_shape, fill=False):
    img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)

    if fill:
        for a in ants:
            for pt in a['points']:
                img[pt[1]][pt[0]][2] = 255
    else:
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

    return True


def opt_ant(rpts, apts, a, i):
    x = []
    y = []

    if i < 5:
        weight = 50.
    else:
        weight = 1.

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

    for pts in rpts:
        if pts[2] == a['id']:
            x.append([float(i) for i in pts[0]])
            y.append([float(i) for i in pts[1]])
            weight = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
            w_sum += weight
            w.append(weight)
            p += np.multiply(pts[0], weight)
            q += np.multiply(pts[1], weight)


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


def iteration(region, ants, i):
    #rpts, rscore = region_boundary_cover(region, ants)
    ascores = 0
    rscore = 0

    chosen_ant = None
    worst_apts = None
    worts_s = -1
    for a in ants:
        apts, s = ant_boundary_cover(region, a)
        if s > worts_s:
            worts_s = s
            worst_apts = apts
            chosen_ant = a

    #for a in ants:
    rpts, rscore = region_boundary_cover(region, ants)
    apts, s = ant_boundary_cover(region, chosen_ant)
    ascores += s

    trans, rot = opt_ant(rpts, apts, chosen_ant, i)
    trans_ant(chosen_ant, trans, rot)

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
    thresh = 0.5
    if len(history) < 2:
        return False

    if abs(history[-2][0] - history[-1][0]) < thresh and abs(history[-2][1] - history[-1][1]) < thresh:
        return True

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


def solve(exp_region, points, ants_ids, exp_ants, frame, img_shape, max_iterations=30, debug=False):
    run = False

    if run:
        ants = prepare_ants(ants_ids, exp_ants)
        region = prepare_region(exp_region, points)

        #if len(region['cont']) > 300:
        #    print "too big region"
        #    return []


        pack = [ants, region]

        afile = open("out/split_by_cont/"+str(frame)+"_"+str(ants_ids[0])+".pkl", "wb")
        pickle.dump(pack, afile)
        afile.close()
    else:
        afile = open("../out/split_by_cont/1144_2.pkl", "rb")
        pack = pickle.load(afile)
        afile.close()
        ants = pack[0]
        region = pack[1]

    history = []

    if debug and run:
        im = draw_situation(region, ants, img_shape)
        cv2.imshow('test', im)
        cv2.imwrite('out/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_a.jpg', im)

    if not run:
        im = draw_situation(region, ants, img_shape)
        cv2.imshow('test', im)
        cv2.waitKey(0)

    done = False
    for i in range(max_iterations):
        if not run:
            im = draw_situation(region, ants, img_shape)
            cv2.imshow('test', im)
            cv2.waitKey(0)

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
        if not done:
            cv2.imwrite('out/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_e.jpg', im)
        else:
            cv2.imwrite('out/split_by_cont/'+str(frame)+'_'+str(ants_ids[0])+'_en.jpg', im)

    if not run:
        im = draw_situation(region, ants, img_shape, fill=True)
        cv2.imshow('test', im)
        cv2.waitKey(0)


    #im = draw_situation(region, ants)
    #cv2.imshow("test", im)

    return ants


def main():
    solve(None, None, None, None, 659, (1024, 1280), debug=True)

    return

if __name__ == '__main__':
    main()