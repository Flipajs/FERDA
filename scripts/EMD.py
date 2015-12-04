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

# (area, (centroidX, Y) )
# regions_P = [(10, (0, 10)), (20, (0, 5)), (9, (0, 0))]
# regions_Q = [(17, (0, 7)), (18, (0, 3))]

regions_P = [(80, (0, 4)), (80, (0, 1))]
regions_Q = [(50, (0, 5)), (50, (0, 3)), (50, (0, 2)), (50, (0, 0))]

prob = build_EMD_lp(regions_P, regions_Q)

prob.solve()
print("Status:", LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)