__author__ = 'flip'

import math
import networkx as nx
import matplotlib.mlab as mlab
import my_utils as my_utils
import numpy as np
import time

def pred_distance_score(a, region, params):
    pred = a.predicted_position(1)
    if my_utils.e_distance(pred, params.arena.center) > (params.arena.size.width / 2) - params.avg_ant_axis_b:
        p = my_utils.get_circle_line_intersection(params, a.state.position, pred)
        return math.sqrt((p.x - region["cx"])**2 + (p.y - region["cy"])**2)
    else:
        x = pred.x - region["cx"]
        y = pred.y - region["cy"]

        return math.sqrt(x*x + y*y)

def max_weight_matching(ants, regions, unassigned_a, unassigned_r, assignment, costs, params):
    graph = nx.Graph()

    start = time.time()
    graph_add_ants(graph, unassigned_a)
    print "ants: ", time.time() - start
    start = time.time()
    graph_add_edges(graph, ants, regions, unassigned_a, unassigned_r, params)
    print "edges: ", time.time() - start

    print "nodes: ", len(graph.nodes())
    print "edges: ", len(graph.edges())
    start = time.time()
    result = nx.max_weight_matching(graph, True)
    print "result: ", time.time() - start
    assignment, costs = interpret_results_of_max_weighted_matching(result, unassigned_a, assignment, costs, graph)

    return assignment, costs


def max_weight_matching_lost(ants, lost_ants, regions, free_regions, params):
    graph = nx.Graph()

    for a_id in lost_ants:
        graph.add_node('a'+str(a_id))
        graph.add_node('u'+str(a_id))

        graph.add_edge('a'+str(a_id), 'u'+str(a_id), weight=params.weighted_matching_lost_edge_cost)

        for r_id in free_regions:
            if ab_area_prob(regions[r_id], params) < 0.5:
                continue

            w = count_lost_node_weight(ants[a_id], regions[r_id], params)
            graph.add_edge('a'+str(a_id), 'g'+str(r_id), weight=w)

    result = nx.max_weight_matching(graph, True)


    region_ids = [None]*len(lost_ants)
    costs = [None]*len(lost_ants)

    for id in range(len(lost_ants)):
        node = result['a'+str(lost_ants[id])]
        if node[0] == 'u':
            region_ids[id] = -1
            costs[id] = -1
        else:
            region_ids[id] = int(node[1:])
            costs[id] = graph.get_edge_data('a'+str(lost_ants[id]), node)['weight']

    return region_ids, costs


def count_lost_node_weight(a, r, params):
    r_p = my_utils.Point(r['cx'], r['cy'])
    return my_utils.e_distance(a.state.position, r_p)


def interpret_results_of_max_weighted_matching(result, unassigned_a, assignment, costs, graph):
    for a_id in unassigned_a:
        node = result['a'+str(a_id)]
        if node[0] == 'u':
            assignment[a_id] = -1
            costs[a_id] = -1
        else:
            idx = int(node[1:])
            assignment[a_id] = idx
            costs[a_id] = graph.get_edge_data('a'+str(a_id), node)['weight']

    return assignment, costs


def graph_add_ants(G, unassigned_ants):
    for a_id in unassigned_ants:
        G.add_node('a'+str(a_id))
        G.add_node('u'+str(a_id))


def prepare_and_add_region_groups(graph, regions, regions_idx):
    prev = -1
    groups = []
    i = -1
    for ridx in regions_idx:
        r = regions[ridx]
        if r["label"] > prev:
            prev = r["label"]
            graph.add_node('g'+str(r["label"]))
            groups.append([ridx])
            i += 1
        else:
            groups[i].append(ridx)

    return groups


def graph_add_edges(graph, ants, regions, unassigned_a, unassigned_r, params):
    thresh = params.undefined_threshold
    for a_id in unassigned_a:
        graph.add_edge('a'+str(a_id), 'u'+str(a_id), weight=thresh)
        for r_id in unassigned_r:
            w = count_node_weight(ants[a_id], regions[r_id], params)

            if w > thresh:
                graph.add_edge('a'+str(a_id), 'r'+str(r_id), weight=w)


def log_normpdf(x, u, s):
    a = math.log(1/(math.sqrt(2*math.pi) * s))
    b = ((x - u)*(x - u))/(2*s*s)

    return a - b


def axis_change_prob(ant, region):
    _, region_a, region_b = my_utils.mser_main_axis_ratio(region["sxy"], region["sxx"], region["syy"])
    a = axis_a_change_prob(ant.state.a, region_a)
    b = axis_b_change_prob(ant.state.b, region_b)

    return a*b


def axis_a_change_prob(a_a, r_a):
    u = 0
    s = 0.4765*3
    max_val = mlab.normpdf(u, u, s)
    #max_val = log_normpdf(u, u, s)

    x = r_a - a_a
    val = mlab.normpdf(x, u, s) / max_val
    #val = log_normpdf(x, u, s) / abs(max_val)

    return val


def axis_b_change_prob(a_b, r_b):
    u = 0
    s = 0.1676*3
    max_val = mlab.normpdf(u, u, s)
    #max_val = log_normpdf(u, u, s)

    x = r_b - a_b
    val = mlab.normpdf(x, u, s) / max_val
    #val = log_normpdf(x, u, s) / abs(max_val)

    return val

def theta_change_prob(ant, region):
    u = 0
    s = 0.1619*5
    max_val = mlab.normpdf(u, u, s)
    #max_val = log_normpdf(u, u, s)

    #theta = my_utils.mser_theta(region["sxy"], region["sxx"], region["syy"])
    theta = region["theta"]

    x = theta - ant.state.theta

    #this solve the jump between phi and -phi
    if x > math.pi/2:
        x = -(x - math.pi)
    elif x < -math.pi/2:
        x = -(x + math.pi)

    if ant.state.lost:
        x /= ((math.log(ant.state.lost_time) + 2) / 2)

    val = mlab.normpdf(x, u, s) / max_val
    #val = log_normpdf(x, u, s) / abs(max_val)

    return val


def position_prob(ant, region, params):
    u = 0
    #s = 4.5669*2
    s = params.avg_ant_axis_a
    max_val = mlab.normpdf(u, u, s)
    #max_val = log_normpdf(u, u, s)
    x = pred_distance_score(ant, region, params)

    #if ant.state.lost:
    #    x /= (math.log(ant.state.lost_time) + 1)

    val = mlab.normpdf(x, u, s) / max_val
    #val = log_normpdf(x, u, s) / abs(max_val)
    return val


def position_prob_collision(ant, region, params):
    u = 0
    #s = 4.5669*2
    s = params.avg_ant_axis_a
    max_val = mlab.normpdf(u, u, s)
    p_x = ant.state.position.x - region['cx']
    p_y = ant.state.position.y - region['cy']
    x = math.sqrt(p_x*p_x + p_y*p_y)

    if ant.state.lost:
        x /= (math.log(ant.state.lost_time) + 1)

    val = mlab.normpdf(x, u, s) / max_val
    return val


def position_prob_without_prediction(ant, region, params):
    u = 0
    s = params.avg_ant_axis_a
    max_val = mlab.normpdf(u, u, s)
    p_x = ant.state.position.x - region['cx']
    p_y = ant.state.position.y - region['cy']
    x = math.sqrt(p_x*p_x + p_y*p_y)

    if ant.state.lost:
        x /= (math.log(ant.state.lost_time) + 1)

    val = mlab.normpdf(x, u, s) / max_val
    return val


def area_prob(area, avg_area):
    a_1_3 = avg_area / 3.
    a_3 = 3 * avg_area
    if area < a_1_3:
        return 0
    elif area < avg_area:
        return math.log(area - a_1_3 + 1) / math.log((2/3.) * avg_area + 1)
    elif area < a_3:
        return math.log(a_3 + 1 - area) / math.log(2 * avg_area + 1)
    else:
        return 0


def axis_ratio_prob(ratio, avg_ratio):
    upper = 8.
    if ratio < 1:
        return 0
    elif ratio < avg_ratio:
        return math.log(ratio + 1) / math.log(avg_ratio + 1)
    elif ratio < upper:
        return math.log(upper + 1 - ratio) / math.log(upper - avg_ratio + 1)
    else:
        return 0

def ab_area_prob(region, params):
    ratio, _, _ = my_utils.mser_main_axis_ratio(region['sxy'], region['sxx'], region['syy'])
    a = (region['area'] / float(params.avg_ant_area)) - params.ab_area_xstart
    ab = (ratio / params.avg_ant_axis_ratio) - params.ab_area_ystart

    x_id = int(math.floor(a / params.ab_area_step))
    y_id = int(math.floor(ab / params.ab_area_step))

    val = 0
    if x_id > 0 and y_id > 0:
        if x_id < params.ab_area_xmax and y_id < params.ab_area_ymax:
            val = params.ab_area_hist[y_id][x_id] / params.ab_area_max

    return val

def a_area_prob(region, params):
    _, a, b = my_utils.mser_main_axis_ratio(region['sxy'], region['sxx'], region['syy'])
    a, b = my_utils.count_head_tail(region['area'], a, b)

    area = (region['area'] / float(params.avg_ant_area)) - params.a_area_xstart
    a = (a / params.avg_ant_axis_a) - params.a_area_ystart

    x_id = int(math.floor(area / params.a_area_step))
    y_id = int(math.floor(a / params.a_area_step))

    val = 0
    if x_id > 0 and y_id > 0:
        if x_id < params.a_area_xmax and y_id < params.a_area_ymax:
            val = params.a_area_hist[y_id][x_id]

    return val

def count_node_weight(ant, region, params):
    #region byl rozdelen... vznikaly by duplikaty
    if "used_for_splitting" in region:
        return 0

    if len(ant.state.collisions) > 0:
        position_p = position_prob_collision(ant, region, params)
    else:
        position_p = position_prob(ant, region, params)

    if params.undefined_threshold > position_p:
        return 0

    theta_p = theta_change_prob(ant, region)

    a_area_p = a_area_prob(region, params)

    if "splitted" in region:
        if a_area_p < 0.5 and theta_p * position_p > 0.8:
            a_area_p = 0.5

    #area_p = area_prob(region['area'], ant.state.area)

    #prob = axis_p * theta_p * position_p
    prob = theta_p * position_p * a_area_p

    if "cont" in region:
        prob *= 1.1-region['overlap'] #1.1 to give chance for complete overlap...

    #if ant.state.lost:
    #    if prob > params.undefined_threshold:
    #        #this ensure that this valu will be higher than weight of undefined state
    #        #prob = np.nextafter(params.undefined_threshold, 1)
    #        #prob = params.undefined_threshold*2
    #        prob *= 0.01
    #        #prob -= 4

    #print "Th: ", theta_p, "P: ", position_p, "A: ", area_p, "PROD: ", prob

    return prob