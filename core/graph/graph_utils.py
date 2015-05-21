__author__ = 'fnaiser'

import numpy as np


def get_best_n_in_nodes(g, node, n, key='score', order='asc'):
    best = [0 for i in range(n)]
    best_n = [None for i in range(n)]

    scores = []
    nodes = []

    for n1, _, d in g.in_edges(node, data=True):
        scores.append(d[key])
        nodes.append(n1)

    scores = np.array(scores)
    nodes = np.array(nodes)

    if order == 'asc':
        ids = np.argsort(scores)
    else:
        ids = np.argsort(-scores)

    r = min(n, len(ids))
    for i in range(r):
        best[i] = scores[ids[i]]
        best_n[i] = nodes[ids[i]]

    return best, best_n


def get_best_n_out_nodes(g, node, n, key='score', order='asc'):
    best = [0 for i in range(n)]
    best_n = [None for i in range(n)]

    scores = []
    nodes = []

    for _, n2, d in g.out_edges(node, data=True):
        scores.append(d[key])
        nodes.append(n2)

    scores = np.array(scores)
    nodes = np.array(nodes)

    if order == 'asc':
        ids = np.argsort(scores)
    else:
        ids = np.argsort(-scores)

    r = min(n, len(ids))
    for i in range(r):
        best[i] = scores[ids[i]]
        best_n[i] = nodes[ids[i]]

    return best, best_n


def get_cc(g, n):
    s_t1 = set()
    s_t2 = set()

    process = [(n, 1)]

    while True:
        if not process:
            break

        n_, t_ = process.pop()

        s_test = s_t2
        if t_ == 1:
            s_test = s_t1

        if n_ in s_test:
            continue

        s_test.add(n_)

        if t_ == 1:
            for _, n2 in g.out_edges(n_):
                process.append((n2, 2))
        else:
            for n2, _ in g.in_edges(n_):
                process.append((n2, 1))

    return list(s_t1), list(s_t2)


def best_2_cc_configs(g, nodes1, nodes2):
    configurations = []
    conf_scores = []

    get_configurations(g, nodes1, nodes2, [], 0, configurations, conf_scores)

    if len(conf_scores) < 2:
        return conf_scores, configurations

    scores = [0, 0]
    confs = [None, None]
    ids = np.argsort(conf_scores)
    for i in range(2):
        scores[i] = conf_scores[ids[i]]
        confs[i] = configurations[ids[i]]

    return scores, confs


def get_configurations(g, nodes1, nodes2, c, s, configurations, conf_scores, use_undefined=True, undefined_edge_cost=0):
    if nodes1:
        n1 = nodes1.pop(0)
        for i in range(len(nodes2)):
            n2 = nodes2.pop(0)
            if n2 in g[n1]:
                get_configurations(g, nodes1, nodes2, c + [(n1, n2)], s+g[n1][n2]['score'], configurations, conf_scores)
            nodes2.append(n2)

        # undefined state
        if use_undefined:
            get_configurations(g, nodes1, nodes2, c + [(n1, None)], s + undefined_edge_cost, configurations, conf_scores)

        nodes1.append(n1)
    else:
        configurations.append(c)
        conf_scores.append(s)


def num_out_edges_of_type(G, n, type):
    num = 0
    last_n = None

    for _, n_, d in G.out_edges(n, data=True):
        if d['type'] == type:
            num += 1
            last_n = n_

    return num, last_n


def num_in_edges_of_type(G, n, type):
    num = 0
    last_n = None
    for n_, _, d in G.in_edges(n, data=True):
        if d['type'] == type:
            num += 1
            last_n = n_

    return num, last_n


def cc_certainty(g, c1, c2):
    configurations = []
    scores = []

    get_configurations(g, c1, c2, [], 0, configurations, scores)
    ids = np.argsort(scores)
    configurations = np.array(configurations)
    scores = np.array(scores)
    configurations = configurations[ids]
    scores = scores[ids]

    n_ = float(len(c1))
    cert = abs(scores[0] / n_)
    if len(scores) > 1:
        cert = abs(scores[0] / n_) * abs(scores[0]-scores[1])

    return cert, configurations, scores