__author__ = 'flip'

import math
import networkx as nx
import matplotlib.mlab as mlab
import my_utils as my_utils

def pred_distance_score(a, region):
    x = a.predicted_position(1).x - region["cx"]
    y = a.predicted_position(1).y - region["cy"]

    return math.sqrt(x*x + y*y)


def max_weight_matching(ants, regions, regions_idx, params):
    graph = nx.Graph()

    graph_add_ants(graph, ants)
    groups = prepare_and_add_region_groups(graph, regions, regions_idx)
    
    ants_groups_preferences = graph_add_edges(graph, ants, regions, groups, params)

    result = nx.max_weight_matching(graph, True)
    region_idxs, costs = interpret_results_of_max_weighted_matching(result, ants, ants_groups_preferences, graph)

    return region_idxs, costs


def interpret_results_of_max_weighted_matching(result, ants, ants_groups_preferences, graph):
    region_idxs = [None]*len(ants)
    costs = [None]*len(ants)

    for a in ants:
        node = result['a'+str(a.id)]
        if node[0] == 'u':
            region_idxs[a.id] = -1
            costs[a.id] = -1
        else:
            idx = int(node[1:])
            region_idxs[a.id] = ants_groups_preferences[a.id][idx]
            costs[a.id] = graph.get_edge_data('a'+str(a.id), node)['weight']

    return region_idxs, costs


def graph_add_ants(G, ants):
    for aidx in range(len(ants)):
        G.add_node('a'+str(aidx))
        G.add_node('u'+str(aidx))


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


def graph_add_edges(graph, ants, regions, groups, params):
    ants_groups_preferences = [[-1]*len(groups)]*len(ants)

    thresh = params.undefined_threshold
    for a in ants:
        graph.add_edge('a'+str(a.id), 'u'+str(a.id), weight=thresh)
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
                graph.add_edge('a'+str(a.id), 'g'+str(g), weight=max)

    return ants_groups_preferences


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

    if 'splitted' in region:
        if region['splitted']:
            theta = region['theta']
    else:
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
    #val = log_normpdf(x, u, s) / abs(max_val)

    return val


def position_prob(ant, region):
    u = 0
    s = 4.5669*2
    max_val = mlab.normpdf(u, u, s)
    #max_val = log_normpdf(u, u, s)
    x = pred_distance_score(ant, region)

    if ant.state.lost:
        x /= (math.log(ant.state.lost_time) + 1)

    val = mlab.normpdf(x, u, s) / max_val
    #val = log_normpdf(x, u, s) / abs(max_val)
    return val


def count_node_weight(ant, region, params):
    #tohle je spatne...
    #axis_p = axis_change_prob(ant, region)

    theta_p = theta_change_prob(ant, region)
    position_p = position_prob(ant, region)

    #prob = axis_p * theta_p * position_p
    prob = theta_p * position_p

    if ant.state.lost:
        if prob > params.undefined_threshold:
            #this ensure that this valu will be higher than weight of undefined state
            #prob = np.nextafter(params.undefined_threshold, 1)
            #prob = params.undefined_threshold*2
            prob *= 0.01
            #prob -= 4

    return prob