from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
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
        costs[fl] = np.linalg.norm(c_p - c_q)

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
    # TODO: edges param to add support to not full bipartite graphs

    prob = LpProblem('EMD', LpMinimize)

    flows = [tuple2str(element) for element in itertools.product(list(range(len(regions_P))), list(range(len(regions_Q))))]

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


def check_nodes_stability(regions_P, regions_Q, flows, threshold, area_med, area_med_w=0.5):
    # check outcomes
    out_max = np.max(flows, axis=1)
    i_out_max = np.argmax(flows, axis=1)

    in_max = np.max(flows, axis=0)
    i_in_max = np.argmax(flows, axis=0)

    preferences = {}

    stability_p = np.ones((len(regions_P), 1), dtype=np.bool)
    for r, r2_i, m, i in zip(regions_P, i_out_max, out_max, list(range(len(out_max)))):
        area = r[0]

        a1 = min(m, area)
        a2 = float(max(m, area))
        if old_div(a1, a2) < threshold or a2-a1 > area_med_w*area_med:
            stability_p[i] = False

        preferences[r[2]] = regions_Q[r2_i][2]

    stability_q = np.ones((len(regions_Q), 1), dtype=np.bool)
    for r, r1_i, m, i in zip(regions_Q, i_in_max, in_max, list(range(len(in_max)))):
        area = r[0]

        a1 = min(m, area)
        a2 = float(max(m, area))
        if old_div(a1, a2) < threshold or a2-a1 > area_med_w*area_med:
            stability_q[i] = False

        preferences[r[2]] = regions_P[r1_i][2]

    unstable_num = len(regions_P) + len(regions_Q) - sum(stability_q) - sum(stability_p)

    return unstable_num[0], stability_p, stability_q, preferences


def get_flows(regions_P, regions_Q):
    prob = build_EMD_lp(regions_P, regions_Q)
    prob.solve()

    if LpStatus[prob.status] != 'Optimal':
        print("WARNING: there was not optimal solution computed in EMD split/merge predicate")

    flows = np.zeros((len(regions_P), len(regions_Q)), dtype=np.int)
    for v in prob.variables():
        # print(v.name, "=", v.varValue)
        p_id = get_pid(v.name)
        q_id = get_qid(v.name)

        flows[p_id, q_id] = v.varValue

    return flows

def detect_stable(regions_P, regions_Q, thresh=0.8, area_med=0):
    flows = get_flows(regions_P, regions_Q)

    return check_nodes_stability(regions_P, regions_Q, flows, thresh, area_med)


def get_unstable_num(regions_P, regions_Q, thresh=0.8):
    num, _, _ = detect_stable(regions_P, regions_Q, thresh=0.8)
    return num



if __name__ == "__main__":
    pass
    # (area, (centroidX, Y) )
    # regions_P = [(10, (0, 10)), (20, (0, 5)), (9, (0, 0))]
    # regions_Q = [(17, (0, 7)), (18, (0, 3))]

    # regions_P = [(80, (0, 4)), (80, (0, 1))]
    # regions_Q = [(50, (0, 5)), (50, (0, 3)), (50, (0, 2)), (50, (0, 0))]
    #
    # regions_P = [(72, (0, 4)), (72, (0, 1))]
    # regions_Q = [(80, (0, 5)), (80, (0, 4))]

    # regions_P = [(397, (967, 584)), (1293, (910, 695)), (1058, (946, 600))]
    # regions_Q = [(1211, (910, 696)), (470, (841, 816)), (963, (945, 600))]
    #
    # print detect_stable(regions_P, regions_Q)


