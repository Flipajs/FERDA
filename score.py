__author__ = 'flip'

import math
import networkx as nx


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


def max_weight_matching(ants, regions):
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

    thresh = 0.5
    for a in ants:
        G.add_edge('a'+str(a.id), 'u'+str(a.id), weight=thresh)
        for g in range(len(groups)):
            max = thresh
            max_id = -1
            for ridx in groups[g]:
                w = count_node_weight(a, regions[ridx])
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


def count_node_weight(ant, region):
    d = distance_prob(ant, region)
    a = area_prob(ant, region)
    val = d * a
    return val


def match(ants, regions, indexes):
    region_ants = [None] * len(regions)
    scores = count_scores(ants, regions, indexes)

    for i in sorted(range(len(scores)), key = lambda k: scores[k]):
        #TODO> allow settings for this value
        if scores[i] > 50:
            break

        ant_id = i / len(indexes)
        region_id = indexes[i % len(indexes)]

        if region_ants[region_id] is None:
            region_ants[region_id] = [ant_id, scores[i]]
        else:
            defender = ants[region_ants[region_id][0]]
            attacker = ants[ant_id]
            if ant_fight(regions[region_id], defender, attacker) == ant_id:
                region_ants[region_id] = [ant_id, scores[i]]

    ant_regions = [None] * len(ants)
    for i in range(len(regions)):
        r = region_ants[i]
        if r is None:
            continue
        if ant_regions[r[0]] is None:
            ant_regions[r[0]] = [i, r[1]]
        else:
            if ant_regions[r[0]][1] > r[1]:
                ant_regions[r[0]] = [i, r[1]]

    assigned = []
    for a in ant_regions:
        if a is None:
            assigned.append(-1)
        else:
            assigned.append(a[0])

    print assigned
    return assigned


#returns winners id, -1 if tie
def ant_fight(region, defender, attacker):
    #TODO> availiable depth settings
    depth = 5
    d_stability = defender.stability(depth)
    a_stability = attacker.stability(depth)

    if d_stability > a_stability:
        return defender.id
    elif a_stability < d_stability:
        return attacker.id
    else:
        return -1