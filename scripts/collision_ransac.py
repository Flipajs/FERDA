from core.project.project import Project
from utils.video_manager import get_auto_video_manager
from matplotlib import pyplot as plt
import cv2
import numpy as np
from utils.geometry import rotate
from scipy.spatial.distance import cdist
import networkx as nx
import itertools
from core.graph.region_chunk import RegionChunk
from heapq import *
import cPickle as pickle


def data_cam1():
    collisions = {
        331: {'s': [96, 97], 'm': 88, 'e': [84, 85]},
        501: {'s': [115, 118], 'm': 127, 'e': [131, 130]},
        680: {'s': [128, 131], 'm': 141, 'e': [142, 140]},
        1526: {'s': [300, 301], 'm': 291, 'e': [293, 296]},
        1653: {'s': [283, 216], 'm': 305, 'e': [311, 312]},
        1681: {'s': [296, 309], 'm': 307, 'e': [308, 313]},
        1757: {'s': [322, 313], 'm': 320, 'e': [324, 325]},
        2025: {'s': [337, 384], 'm': 377, 'e': [378, 391]},
        3396: {'s': [583, 582], 'm': 577, 'e': [606, 607]},
        3621: {'s': [688, 687], 'm': 679, 'e': [677, 678]},
    }

    return collisions

def data_cam2():
    #Cam2
    collisions = {
        1: {'s': [6, 7], 'm': 1, 'e': [18, 23]},
        2: {'s': [64, 65], 'm': 50, 'e': [48, 62]},
        3: {'s': [111, 112, 120], 'm': 132, 'e': [123, 124, 116]},
        332: {'s': [64, 65], 'm': 50, 'e': [48, 62]},
        111: {'s': [23, 18], 'm': 24, 'e': [25, 26]},
        394: {'s': [60, 61], 'm': 49, 'e': [77, 78]},
        1985: {'s': [306, 307], 'm': 302, 'e': [297, 305]},
        3130: {'s': [425, 411], 'm': 420, 'e': [421, 422]},
        3350: {'s': [464, 460], 'm': 457, 'e': [452, 467]}
    }

    return collisions


def __get_rts(a1, a2, b1, b2, rot_center):
    a_ = a2 - a1
    a_n = np.linalg.norm(a_)
    b_ = b2 - b1
    b_n = np.linalg.norm(b_)

    if a_n == 0 or b_n == 0:
        return None, None, None, None

    s = b_n / a_n

    x_ = np.dot(a_.T, b_) / (a_n * b_n)
    # numerical errors fix
    x_ = min(1, max(-1, x_))

    from math import acos
    theta = acos(x_)

    # compute the orientation
    # http://math.stackexchange.com/questions/317874/calculate-the-angle-between-two-vectors
    U = np.array([[a_[0], b_[0]], [a_[1], b_[1]]])
    if np.linalg.det(U) < 0:
        theta = -theta % 2*np.pi

    t = (b1+b2) / 2 - rotate(np.array([(a1+a2)/2]), theta, rot_center)[0]

    return t, theta, s, (a1+a2) / 2

def __transform_pts(pts, r, t, rot_center):
    pts = np.array(rotate(pts, r, rot_center))

    pts[:, 0] = pts[:, 0] + t[0]
    pts[:, 1] = pts[:, 1] + t[1]

    return pts

def get_geom_s(start_, coef, num):
    r = []
    for i in range(num):
        start_ = start_ * coef
        r.append(start_)

    return r

def __get_cost(pts1, p_type_starts1, pts2, p_type_starts2, type_weights, r, t, rot_center, dmap=None, intensities=None, thresh=1):
    pts1 =__transform_pts(pts1, r, t, rot_center)

    # threshs = get_geom_s(2, 2**0.5, 5)
    cost = 0
    # -1 because, there is last number describing the end of interval in p_type_starts...
    for c in range(len(p_type_starts1)-1):
        ids1_ = slice(p_type_starts1[c], p_type_starts1[c+1])
        ids2_ = slice(p_type_starts2[c], p_type_starts2[c+1])

        # if dmap is None:
        d = cdist(pts1[ids1_, :], pts2[ids2_, :])
        # else:
        #     d = []
        #     pts_ = pts1[ids1_, :]
        #     for pt in pts_:
        #         if dmap.is_inside_object(pt):
        #             _, d_ = dmap.get_nearest_point(pt)
        #
        #             d.append(d_)
        #
        #     d = np.array(d)

        mins_ = np.min(d, axis=1)
        amins_ = np.argmin(d, axis=1)

        # mins_ = mins_**1.1
        mins_[mins_ > thresh] = thresh
        # mins_[mins_ <= thresh] = 0

        cost += np.sum(mins_) * type_weights[c]
        if intensities is not None:
            int2 = intensities[1][ids2_]
            int_diff = intensities[0][ids1_] - int2[amins_]
            int_diff = abs(np.asarray(int_diff[mins_ < thresh], dtype=np.float) / 3)

            cost += sum(int_diff) * type_weights[c]

    return cost


def __solution_distance(t1, t2, r1, r2):
    d = np.linalg.norm(t1 - t2)

    a = 1.7
    c = 50

    rd = abs((max(r1, r2) - min(r1, r2)) % (2*np.pi))
    rd = 2*np.pi - rd if rd > np.pi else rd

    return d**a + c * rd

def __compare_solution(best_t, best_r, best_rot_center, best_cost, t, r, rot_center, cost, best_n, min_diff_t = 50):
    j = 0
    while j < len(best_cost) and cost < best_cost[j]:
        j += 1

    remove = -1
    for i in xrange(1, len(best_t)):
        if __solution_distance(best_t[i], t, best_r[i], r) < min_diff_t:
            if best_cost[i] > cost:
                remove = i
            else:
                return

    if j > 0 or len(best_t) == 0:
        best_r.insert(j, r)
        best_t.insert(j, t)
        best_rot_center.insert(j, rot_center)
        best_cost.insert(j, cost)

    remove = max(remove, 0)
    # + 1 because there is a fake solution on first position with inf. cost
    if len(best_r) > best_n + 1 or remove:
        best_r.pop(remove)
        best_t.pop(remove)
        best_rot_center.pop(remove)
        best_cost.pop(remove)


def estimate_rt(kps1, rot_center, kps2, best_n=1):
    p_type1 = []
    type_starts1 = []
    pts1 = []
    intensities1 = []
    si = 0

    for a in sorted([int(x) for x in kps1]):
        type_starts1.append(len(pts1))
        for b in kps1[a]:
            pts1.append(b['point'])
            intensities1.append(b['intensity'])
            p_type1.append(si)

        si += 1

    type_starts1.append(len(pts1))
    intensities1 = np.array(intensities1, dtype=np.int32)

    pts1 = np.array(pts1)

    type_starts2 = []
    pts2 = []
    intensities2 = []
    for a in sorted([int(x) for x in kps2]):
        type_starts2.append(len(pts2))
        for b in kps2[a]:
            pts2.append(b['point'])
            intensities2.append(b['intensity'])

    intensities2 = np.array(intensities2, dtype=np.int32)
    type_starts2.append(len(pts2))

    pts2 = np.array(pts2)


    max_steps = 20000
    num_trials = 500

    best_t = [None]
    best_r = [None]
    best_rot_center = [None]
    best_cost = [np.inf]

    trials = 0
    for i in range(max_steps):
        ai = random.randint(len(pts1), size=2)
        s_ = p_type1[ai[0]]
        bi1 = random.randint(type_starts2[s_], type_starts2[s_+1])
        s_ = p_type1[ai[1]]
        bi2 = random.randint(type_starts2[s_], type_starts2[s_+1])

        pa1 = pts1[ai[0], :]
        pa2 = pts1[ai[1], :]
        pb1 = pts2[bi1, :]
        pb2 = pts2[bi2, :]

        # test if they are reasonable pairs
        if np.linalg.norm(pa1-pa2) < 15:
            continue

        if abs(np.linalg.norm(pa1-pa2) - np.linalg.norm(pb1-pb2)) > 2:
            continue

        t, r, s, _ = __get_rts(pa1, pa2, pb1, pb2, rot_center)

        if t is None:
            continue

        trials += 1

        type_weights = [1, 1]
        cost = __get_cost(pts1, type_starts1, pts2, type_starts2, type_weights, r, t, rot_center, intensities=(intensities1, intensities2), thresh=20)

        __compare_solution(best_t, best_r, best_rot_center, best_cost, t, r, rot_center, cost, best_n)

        if trials >= num_trials:
            break

    print "SKIPPED: ", i - num_trials

    # remove fake data
    best_t.pop(0)
    best_r.pop(0)
    best_rot_center.pop(0)
    best_cost.pop(0)

    return list(reversed(best_t)), list(reversed(best_r)), list(reversed(best_rot_center)), list(reversed(best_cost))


def __prepare_pts_and_cont(r, step, gray):
    result = {0: [], 1: []}
    pts = r.pts()
    pts = pts[np.random.randint(len(pts), size=len(pts)/step), :]

    for i in range(len(pts)):
        p = pts[i, :]
        result[0].append({'point': p, 'intensity': gray[p[0], p[1]]})

    ptsc = r.contour_without_holes()
    for i in range(len(ptsc)):
        if i % step == 0:
            p = ptsc[i, :]
            result[1].append({'point': p, 'intensity': gray[p[0], p[1]]})

    ptsc = ptsc[range(0, len(ptsc), step), :]

    return result, pts, ptsc


def __add_node(G, ni, nodes_groups, ng, data):
    G.add_node(ni, data=data)
    if len(nodes_groups) <= ng:
        nodes_groups.append([])

    nodes_groups[ng].append(ni)

    return ni + 1


def __reconstruct_path(predecesors, id_):
    path = []

    while predecesors[id_] is not None:
        path.append(id_)
        id_ = predecesors[id_]

    return list(reversed(path))


def __optimize(G):
    q = []

    heappush(q, (0, 0))
    predecesors = [None] * G.number_of_nodes()
    costs = [np.inf] * G.number_of_nodes()
    costs[0] = 0

    while len(q):
        cost, id_ = heappop(q)

        if costs[id_] != cost:
            continue # outdated...

        if G.out_degree(id_) == 0:
            path = __reconstruct_path(predecesors, id_)
            return path, cost

        for _, id2, c in G.out_edges(id_, data=True):
            c = c['cost']
            # for d in G.node[id2]['data']:
            #     c += d['cost']

            new_cost = c+cost
            if predecesors[id2] is None or costs[id2] > new_cost:
                heappush(q, (new_cost, id2))
                predecesors[id2] = id_
                costs[id2] = new_cost


if __name__ == '__main__':
    from numpy import random
    random.seed(19)

    p = Project()
    data = data_cam1()
    name = 'Cam1/cam1.fproj'
    wd = '/Users/flipajs/Documents/wd/gt/'
    p.load(wd+name)
    vm = get_auto_video_manager(p)
    #
    # 1: {'s': [6, 7], 'm': 1, 'e': [18, 23]},
    #     2: {'s': [64, 65], 'm': 50, 'e': [48, 62]},
    #     3: {'s': [111, 112, 120], 'm': 132, 'e': [123, 124, 116]},
    #     111: {'s': [23, 18], 'm': 24, 'e': [25, 26]},
    #     394: {'s': [60, 61], 'm': 49, 'e': [77, 78]},
    #     1985: {'s': [306, 307], 'm': 302, 'e': [297, 305]},
    #     3130: {'s': [425, 411], 'm': 420, 'e': [421, 422]}, HARD
    #     3350: {'s': [464, 460], 'm': 457, 'e': [452, 467]}


    #     331: {'s': [96, 97], 'm': 88, 'e': [84, 85]},
    #     501: {'s': [115, 118], 'm': 127, 'e': [131, 130]},
    #     680: {'s': [128, 131], 'm': 141, 'e': [142, 140]},
    #     1653: {'s': [283, 216], 'm': 305, 'e': [311, 312]},
    #     1681: {'s': [296, 309], 'm': 307, 'e': [308, 313]},
    #     1757: {'s': [322, 313], 'm': 320, 'e': [324, 325]},
    #     2025: {'s': [337, 384], 'm': 377, 'e': [378, 391]}, # chyba - resitelna podminkou na overlap
    # }

    seq_n = 3621
    d = data[seq_n]


    G = nx.DiGraph()
    ni = 0  # node id
    nodes_groups = []
    ng = 0  # node group id

    STEP = 5
    STEP2 = 3
    BEST_N = 10

    plt.ion()

    start_rs = []
    transformations = []
    start_data = []
    start_pts = []
    start_ptsc = []

    start_im = vm.get_frame(p.chm[d['s'][0]].end_frame(p.gm))
    start_gray = cv2.cvtColor(start_im, cv2.COLOR_BGR2GRAY)

    for ch_id in d['s']:
        r = p.gm.region(p.chm[ch_id].end_vertex_id())
        print r.theta_
        start_rs.append(r)
        transformations.append({'t': np.array([0, 0]), 'r': 0, 'rot_center': r.centroid(), 'cost': 0})
        x, pts, ptsc = __prepare_pts_and_cont(r, STEP, start_gray)
        start_data.append(x)
        start_pts.append(pts)
        start_ptsc.append(ptsc)

    end_rs = []
    end_pts = []
    end_ptsc = []
    for ch_id in d['e']:
        r = p.gm.region(p.chm[ch_id].start_vertex_id())
        print r.theta_
        end_rs.append(r)
        x, pts, ptsc = __prepare_pts_and_cont(r, STEP2, start_gray)
        end_pts.append(pts)
        end_ptsc.append(ptsc)

    # add first node
    ni = __add_node(G, ni, nodes_groups, ng, transformations)

    # for each middle region...
    print p.chm[d['m']].length()
    rch = RegionChunk(p.chm[d['m']], p.gm, p.rm)
    ii = -1
    if True:
        for r in rch.regions_gen():
            ii += 1
            # if ii not in [17, 18, 19]:
            #     continue

            print "FRAME: ", r.frame()
            ng += 1
            im = vm.get_frame(r.frame())
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            # step-1 = little hack to have higher densiti where to fit
            x, x_pts, ptsc = __prepare_pts_and_cont(r, STEP2, gray)

            results = []
            for i, sd in enumerate(start_data):
                best_t, best_r, best_rot_center, cost = estimate_rt(sd, start_rs[i].centroid(), x, best_n=BEST_N)

                # for j in range(len(best_t)):
                #     plt.cla()
                #     plt.scatter(x_pts[:, 1], x_pts[:, 0], c='k', marker='x', s=20, alpha=0.5)
                #     plt.hold(True)
                #     plt.scatter(ptsc[:, 1], ptsc[:, 0], c='k', alpha=.7, s=25)
                #     plt.scatter(start_pts[i][:, 1], start_pts[i][:, 0], c='r', s=30, alpha=.20)
                #
                #     print j
                #     print best_t[j], best_r[j], cost[j]
                #
                #     plt.title(str(j) + ' ' + str(cost[j]))
                #     pts_ = __transform_pts(start_pts[i], best_r[j], best_t[j], best_rot_center[j])
                #     ptsc_ = __transform_pts(start_ptsc[i], best_r[j], best_t[j], best_rot_center[j])
                #     plt.hold(True)
                #     plt.scatter(pts_[:, 1], pts_[:, 0], c='r', s=20, alpha=0.4, marker='x')
                #     plt.scatter(ptsc_[:, 1], ptsc_[:, 0], c='r', s=25, alpha=0.7)
                #     plt.scatter(pts_[0, 1], pts_[0, 0], c='g', s=100, alpha=0.9)
                #     plt.hold(False)
                #
                #     plt.axis('equal')
                #
                #     plt.show()
                #     plt.waitforbuttonpress()



                # for i in range(len(best_r)):
                #     best_r.append((best_r[i] + np.pi) % 2*np.pi)
                #     best_t.append(best_t[i])
                #     best_rot_center.append(best_rot_center[i])
                #     cost.append(cost[i])

                results.append((best_t, best_r, best_rot_center, cost))


            for ids in itertools.product(*[range(len(x[0])) for x in results]):
                transformations = []
                for i, id_ in enumerate(ids):
                    x = results[i]
                    # 0 - best_T, 1 - best_r, 2 - best_rot_center, 3 - cost
                    transformations.append({'t': x[0][id_],
                                            'r': x[1][id_],
                                            'rot_center': x[2][id_],
                                            'cost': x[3][id_]})

                ni = __add_node(G, ni, nodes_groups, ng, transformations)

        ng += 1
        # add all permutations for the last frame of collision
        for ids in itertools.permutations(range(len(start_data))):
            transformations = []

            for i, id_ in enumerate(ids):
                rs = p.gm.region(p.chm[d['s'][i]].end_vertex_id())
                re = p.gm.region(p.chm[d['e'][id_]].start_vertex_id())

                t = re.centroid() - rs.centroid()
                r = re.theta_ - rs.theta_

                transformations.append({'t': t,
                                        'r': r,
                                        'permutation': ids,
                                        'cost': 0})

            ni = __add_node(G, ni, nodes_groups, ng, transformations)

        # add end node
        ng += 1
        ni = __add_node(G, ni, nodes_groups, ng, [{'cost': 0}])

        print "adding edges"

        # add edges...
        for i in range(len(nodes_groups)-1):
            g1 = nodes_groups[i]
            g2 = nodes_groups[i+1]

            for id1, id2 in itertools.product(g1, g2):
                cost = 0
                if 't' in G.node[id2]['data'][0]:
                    imgs = []

                    j = 0
                    for d1, d2 in zip(G.node[id1]['data'], G.node[id2]['data']):
                        t1 = d1['t']
                        t2 = d2['t']
                        r1 = d1['r']
                        r2 = d2['r']

                        dist = 3 * __solution_distance(t1, t2, r1, r2)

                        cost = cost + dist + d2['cost']

                G.add_edge(id1, id2, cost=cost)

        with open('/Users/flipajs/Desktop/temp/'+str(seq_n)+'.pkl', 'wb') as f:
            pickle.dump(G, f)

    else:
        with open('/Users/flipajs/Desktop/temp/'+str(seq_n)+'.pkl', 'rb') as f:
            G = pickle.load(f)

    print "OPTIMIZING"
    path, cost = __optimize(G)

    # edge_labels=dict([((u,v,), int(d['cost']))
    #              for u,v,d in G.edges(data=True)])
    #
    # pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'))
    # nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='r', arrows=True)
    # nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    # # nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
    # plt.show()

    print "COST: ", cost
    cs = ['r', 'c', 'y', 'b', 'm']

    from utils.roi import get_roi

    margin = 10
    y = np.inf
    yy = 0
    x = np.inf
    xx = 0

    pts = list(start_pts)
    pts.extend([r.pts() for r in rch.regions_gen()])
    pts.extend(end_pts)

    for p in pts:
        roi = get_roi(p)
        y = min(y, roi.y())
        yy = max(yy, roi.y() + roi.height())
        x = min(x, roi.x())
        xx = max(xx, roi.x() + roi.width())

    print y, yy, x, xx

    s = 25
    s2 = 25
    m = 'o'
    m2 = 'x'
    alpha = .5
    alpha2 = .35

    ## INIT
    plt.figure()
    plt.imshow(cv2.cvtColor(start_im, cv2.COLOR_BGR2RGB))
    plt.hold(True)
    for i, sp in enumerate(start_pts):
        plt.scatter(sp[:, 1], sp[:, 0], c=cs[i], marker=m2, s=s2, alpha=alpha2)
        plt.scatter(start_ptsc[i][:, 1], start_ptsc[i][:, 0], c=cs[i], marker=m, s=s, alpha=alpha)
    plt.hold(False)

    # plt.axis('equal')
    axes = plt.gca()
    axes.set_xlim([x-margin, xx+margin])
    axes.set_ylim([y-margin, yy+margin])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()

    import os
    try:
        os.mkdir('/Users/flipajs/Desktop/temp/'+str(seq_n))
    except:
        pass

    plt.savefig('/Users/flipajs/Desktop/temp/'+str(seq_n)+'/0_start.jpg')
    plt.show()
    # plt.waitforbuttonpress()

    # MIDDLE
    prev_id = -1
    frame = 0
    j = 1
    for n, r in zip(path, rch.regions_gen()):
        frame = r.frame()
        im = vm.get_frame(frame)

        # gray is not used... in this case so it is not problem if it is feeded by wrong img - start_gray
        _, x_pts, ptsc = __prepare_pts_and_cont(r, STEP2, start_gray)

        plt.cla()
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.hold(True)
        plt.scatter(x_pts[:, 1], x_pts[:, 0], c='k', marker=m2, s=s2, alpha=alpha2)
        plt.scatter(ptsc[:, 1], ptsc[:, 0], c='k', alpha=alpha, s=s, marker=m)

        for i, d in enumerate(G.node[n]['data']):
            edge_c = 0
            if prev_id > -1:
                edge_c = G.get_edge_data(prev_id, n)
                edge_c = edge_c['cost']
            print d['t'], d['r'], d['rot_center'], d['cost'] if 'cost' in d else "", edge_c

            pts_ = __transform_pts(start_pts[i], d['r'], d['t'], d['rot_center'])
            ptsc_ = __transform_pts(start_ptsc[i], d['r'], d['t'], d['rot_center'])

            plt.scatter(pts_[:, 1], pts_[:, 0], c=cs[i], s=s2, alpha=alpha2, marker=m2,)
            plt.scatter(ptsc_[:, 1], ptsc_[:, 0], c=cs[i], s=s, alpha=alpha)
            plt.scatter(pts_[0, 1], pts_[0, 0], c='g', s=s, alpha=alpha, marker=m)

        prev_id = n

        plt.hold(False)
        # plt.axis('equal')
        axes = plt.gca()
        axes.set_xlim([x-margin, xx+margin])
        axes.set_ylim([y-margin, yy+margin])

        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw()
        plt.savefig('/Users/flipajs/Desktop/temp/'+str(seq_n)+'/'+str(j)+'.jpg')

        # plt.show()
        # plt.waitforbuttonpress()
        j += 1


    plt.cla()
    im = vm.get_frame(frame+1)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.hold(True)
    for i, id_ in enumerate(G.node[path[-2]]['data'][0]['permutation']):
        ep = end_pts[id_]
        epc = end_ptsc[id_]
        plt.scatter(ep[:, 1], ep[:, 0], c=cs[i], s=s2, alpha=alpha2, marker=m2)
        plt.scatter(epc[:, 1], epc[:, 0], c=cs[i], s=s, alpha=alpha, marker=m)

    plt.hold(False)
    # plt.axis('equal')
    axes = plt.gca()
    axes.set_xlim([x-margin, xx+margin])
    axes.set_ylim([y-margin, yy+margin])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()

    plt.savefig('/Users/flipajs/Desktop/temp/'+str(seq_n)+'/'+str(j)+'_end.jpg')
    # plt.show()
    # plt.waitforbuttonpress()