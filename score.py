__author__ = 'flip'

import math
import networkx as nx
import numpy as np
import matplotlib.mlab as mlab
import utils as my_utils


def count_scores(ants, regions, indexes):
    scores = []

    for ant_id in range(len(ants)):
        for i in range(len(indexes)):
            scores.append(pred_distance_score(ants[ant_id], regions[indexes[i]]))

    return scores


def pred_distance_score(a, region):
    x = a.predicted_position(1).x - region["cx"]
    y = a.predicted_position(1).y - region["cy"]

    return math.sqrt(x*x + y*y)


def max_weight_matching(ants, regions, params):
    G = nx.Graph()
    for aidx in range(len(ants)):
        G.add_node('a'+str(aidx))
        G.add_node('u'+str(aidx))

    prev = -1
    groups = []
    i = -1
    for ridx in range(len(regions)):
        r = regions[ridx]
        if r["label"] > prev:
            prev = r["label"]
            G.add_node('g'+str(r["label"]))
            #groups.append('g'+str(r["label"]))
            groups.append([ridx])
            i += 1
        else:
            groups[i].append(ridx)

    ants_groups_preferences = [[-1]*len(groups)]*len(ants)

    #thresh = 0.5
    thresh = params.undefined_threshold
    for a in ants:
        G.add_edge('a'+str(a.id), 'u'+str(a.id), weight=thresh)
        for g in range(len(groups)):
            max = thresh
            max_id = -1
            for ridx in groups[g]:
                w = count_node_weight(a, regions[ridx], params)
                if w > max:
                    max = w
                    max_id = ridx

            if max_id > -1:
                ants_groups_preferences[a.id][g] = max_id
                G.add_edge('a'+str(a.id), 'g'+str(g), weight=max)

    result = nx.max_weight_matching(G, True)
    region_idxs = [None]*len(ants)

    for a in ants:
        node = result['a'+str(a.id)]
        if node[0] == 'u':
            region_idxs[a.id] = -1
        else:
            idx = int(node[1:])
            region_idxs[a.id] = ants_groups_preferences[a.id][idx]

    return region_idxs


def distance_prob(ant, region):
    weight = 0.01
    d = pred_distance_score(ant, region)

    return math.exp(-weight*d)


def area_prob(ant, region):
    weight = 0.5
    a = abs(ant.area_weighted - region["area"])/float(ant.state.area)

    return math.exp(-weight*a)


def axis_change_prob(ant, region):
    _, region_a, region_b = my_utils.mser_main_axis_rate(region["sxy"], region["sxx"], region["syy"])
    a = axis_a_change_prob(ant.state.a, region_a)
    b = axis_b_change_prob(ant.state.b, region_b)

    return a*b


def axis_a_change_prob(a_a, r_a):
    u = 0
    s = 0.4765*3
    max_val = mlab.normpdf(u, u, s)

    x = r_a - a_a
    val = mlab.normpdf(x, u, s) / max_val

    return val


def axis_b_change_prob(a_b, r_b):
    u = 0
    s = 0.1676*3
    max_val = mlab.normpdf(u, u, s)

    x = r_b - a_b
    val = mlab.normpdf(x, u, s) / max_val

    return val


def theta_change_prob(ant, region):
    u = 0
    s = 0.1619*3
    max_val = mlab.normpdf(u, u, s)

    theta = my_utils.mser_theta(region["sxy"], region["sxx"], region["syy"])
    x = theta - ant.state.theta

    #this solve the jump between phi and -phi
    if x > math.pi:
        x = -(x - 2*math.pi)
    elif x < -math.pi:
        x = -(x + 2*math.pi)

    if ant.state.lost:
        x /= ((math.log(ant.state.lost_time) + 2) / 2)

    val = mlab.normpdf(x, u, s) / max_val

    return val


def position_prob(ant, region):
    u = 0
    s = 4.5669*2
    max_val = mlab.normpdf(u, u, s)

    #vel = ant.velocity(1)
    #pred_dist = math.sqrt(vel.x*vel.x + vel.y*vel.y)
    #
    #xx = ant.state.position.x - region["cx"]
    #yy = ant.state.position.y - region["cy"]
    #dist = math.sqrt(xx*xx + yy*yy)

    #x = dist - pred_dist

    x = pred_distance_score(ant, region)

    if ant.state.lost:
        x /= (math.log(ant.state.lost_time) + 1)

    val = mlab.normpdf(x, u, s) / max_val

    #print "@@@ ", val, " ", x, " ", vel.x, " ", vel.y, " ", region["cx"], " ", region["cy"]
    return val


def count_node_weight(ant, region, params):
    axis_p = axis_change_prob(ant, region)
    theta_p = theta_change_prob(ant, region)
    position_p = position_prob(ant, region)

    prob = axis_p * theta_p * position_p
    #prob = axis_p + theta_p + position_p

    #if prob > 0.0001:
    #    print ">>>>>"
    ##
    #print axis_p, " ", theta_p, " ", position_p, " ", axis_p * theta_p * position_p

    if ant.state.lost:
        if prob > params.undefined_threshold:
            #this ensure that this valu will be higher than weight of undefined state
            #prob = np.nextafter(params.undefined_threshold, 1)
            #prob = params.undefined_threshold*2
            prob *= 0.01
            print "LOST", prob


    return prob
    #d = distance_prob(ant, region)
    #a = area_prob(ant, region)
    #val = d * a
    #return val



#def reset_losts(ant, ants, losts, regions, result):
#    for i in losts:



#def match(ants, regions, indexes):
#    region_ants = [None] * len(regions)
#    scores = count_scores(ants, regions, indexes)
#
#    for i in sorted(range(len(scores)), key = lambda k: scores[k]):
#        #TODO> allow settings for this value
#        if scores[i] > 50:
#            break
#
#        ant_id = i / len(indexes)
#        region_id = indexes[i % len(indexes)]
#
#        if region_ants[region_id] is None:
#            region_ants[region_id] = [ant_id, scores[i]]
#        else:
#            defender = ants[region_ants[region_id][0]]
#            attacker = ants[ant_id]
#            if ant_fight(regions[region_id], defender, attacker) == ant_id:
#                region_ants[region_id] = [ant_id, scores[i]]
#
#    ant_regions = [None] * len(ants)
#    for i in range(len(regions)):
#        r = region_ants[i]
#        if r is None:
#            continue
#        if ant_regions[r[0]] is None:
#            ant_regions[r[0]] = [i, r[1]]
#        else:
#            if ant_regions[r[0]][1] > r[1]:
#                ant_regions[r[0]] = [i, r[1]]
#
#    assigned = []
#    for a in ant_regions:
#        if a is None:
#            assigned.append(-1)
#        else:
#            assigned.append(a[0])
#
#    print assigned
#    return assigned

#returns winners id, -1 if tie
#def ant_fight(region, defender, attacker):
#    #TODO> availiable depth settings
#    depth = 5
#    d_stability = defender.stability(depth)
#    a_stability = attacker.stability(depth)
#
#    if d_stability > a_stability:
#        return defender.id
#    elif a_stability < d_stability:
#        return attacker.id
#    else:
#        return -1
