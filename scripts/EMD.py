from pulp import *
import itertools
import re
import numpy as np


def tuple2str(t):
    return 'p'+str(t[0])+'q'+str(t[1])


def get_pid(s):
    return int(re.search('%s(.*)%s' % ('p', 'q'), s).group(1))


def get_qid(s):
    return int(re.search('%s(.*)' % ('q'), s).group(1))


def get_costs(flows, regions_P, regions_Q):
    costs = {}
    for fl in flows:
        p = get_pid(fl)
        q = get_qid(fl)

        c_p = np.array(regions_P[p][1])
        c_q = np.array(regions_Q[q][1])
        costs[fl] = np.linalg.norm(c_p - c_q)**1.3

    return costs

def get_areas_sums(r_P, r_Q):
    ap = 0
    aq = 0
    for it in r_P:
        ap += it[0]

    for it in r_Q:
        aq += it[0]

    return ap, aq


def build_EMD_lp(regions_P, regions_Q):
    #TODO: edges param to add support to not full bipartite graphs

    prob = LpProblem('EMD', LpMinimize)

    flows = [tuple2str(element) for element in itertools.product(range(len(regions_P)), range(len(regions_Q)))]

    costs = get_costs(flows, regions_P, regions_Q)

    # TODO what does 0 means?
    flow_vars = LpVariable.dicts("Vars",flows,0)

    sum_p, sum_q = get_areas_sums(regions_P, regions_Q)

    prob += lpSum([costs[i]*flow_vars[i] for i in flows])

    Q = len(regions_Q)
    P = len(regions_P)
    # j part...
    for i in range(P):
        fvs = [flow_vars[tuple2str((i, j))] for j in range(Q)]
        prob += lpSum(sum(fvs)) <= regions_P[i][0]

    # i part...
    for j in range(Q):
        fvs = [flow_vars[tuple2str((i, j))] for i in range(P)]
        prob += lpSum(sum(fvs)) <= regions_Q[j][0]

    prob += lpSum([flow_vars[i] for i in flows]) <= min(sum_p, sum_q)
    prob += lpSum([flow_vars[i] for i in flows]) >= min(sum_p, sum_q)

    return prob

def check_nodes_stability(regions_P, regions_Q, flows, threshold):
    # check outcomes
    out_max = np.max(flows, axis=1)
    in_max = np.max(flows, axis=0)

    stability_p = np.ones((len(regions_P), 1), dtype=np.bool)
    for r, m, i in zip(regions_P, out_max, range(len(out_max))):
        area = r[0]
        if m / float(area) < threshold:
            stability_p[i] = False

    stability_q = np.ones((len(regions_Q), 1), dtype=np.bool)
    for r, m, i in zip(regions_Q, in_max, range(len(in_max))):
        area = r[0]
        if m / float(area) < threshold:
            stability_q[i] = False

    unstable_num = len(regions_P) + len(regions_Q) - sum(stability_q) - sum(stability_p)
    return unstable_num[0], stability_p, stability_q

def detect_unstable(regions_P, regions_Q, thresh=0.8):
    prob = build_EMD_lp(regions_P, regions_Q)
    prob.solve()

    if LpStatus[prob.status] != 'Optimal':
        print "WARNING: there was not optimal solution computed in EMD split/merge predicate"

    flows = np.zeros((len(regions_P), len(regions_Q)), dtype=np.int)
    for v in prob.variables():
        # print(v.name, "=", v.varValue)
        p_id = get_pid(v.name)
        q_id = get_qid(v.name)

        flows[p_id, q_id] = v.varValue

    return check_nodes_stability(regions_P, regions_Q, flows, thresh)


def get_unstable_num(regions_P, regions_Q, thresh=0.8):
    num, _, _ = detect_unstable(regions_P, regions_Q, thresh=0.8)
    return num


# (area, (centroidX, Y) )
# regions_P = [(10, (0, 10)), (20, (0, 5)), (9, (0, 0))]
# regions_Q = [(17, (0, 7)), (18, (0, 3))]

# regions_P = [(80, (0, 4)), (80, (0, 1))]
# regions_Q = [(50, (0, 5)), (50, (0, 3)), (50, (0, 2)), (50, (0, 0))]
#
# regions_P = [(72, (0, 4)), (72, (0, 1))]
# regions_Q = [(80, (0, 5)), (80, (0, 4))]

# print get_result_flows(regions_P, regions_Q)