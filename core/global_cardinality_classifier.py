import pulp
import time
from sys import maxsize
import numpy as np


def get_tracklet_cardinalities(project):
    """

    :param project: Project()
    :return: dict, {tracklet id: number of objects}
    """
    collateral_sets = get_collateral_sets(project, max_frame=5000)
    tracklets = set([t for cs in collateral_sets for t in cs])

    predecessors = generate_predecessor_map(tracklets, project)
    successors = generate_successor_map(tracklets, project)
    # print("predecessors t: {}".format(time.time() - t))

    num_animals = len(project.animals)
    t = time.time()
    areas = get_median_areas(tracklets, project)
    # TODO: medain_area should probably be median from all areas...
    median_area = np.median(areas.values())
    print("area median extraction time: {}".format(time.time() - t))
    area_relaxation_coef = 0

    print("#animals {}, #tracklets: {}, #collateral sets: {}, mean area: {}".format(num_animals, len(tracklets),
                                                                                    len(collateral_sets), median_area))

    cardinalities = build_ilp_and_solve(tracklets, collateral_sets, predecessors, successors,
                                                          num_animals,
                                                          median_area, areas,
                                                          print_ilp=False)  # area_relaxation_coef=area_relaxation_coef,
    return cardinalities


def fill_tracklet_cardinalites(project):
    tracklet_cardinalities = get_tracklet_cardinalities(project)
    for t_id, cardinality in tracklet_cardinalities.iteritems():
        project.chm[t_id].cardinality = cardinality
        if cardinality == 1:
            project.chm[t_id].segmentation_class = 0  # single
        elif cardinality >= 1:
            project.chm[t_id].segmentation_class = 1  # multi
        else:
            project.chm[t_id].segmentation_class = 2  # noise


def get_collateral_sets(project, max_frame=maxsize):
    max_frame = min(max_frame, project.num_frames())

    collateral_sets = []

    all_end_frames = []
    for t in project.chm.tracklet_gen():
        # if len(t) == 1:
        #     all_end_frames.append(t.end_frame())

        all_end_frames.append(t.start_frame())
        all_end_frames.append(t.end_frame())

    all_end_frames = sorted(all_end_frames)
    # all_end_frames = sorted([t.end_frame() ])

    for frame in all_end_frames:
        if frame > max_frame:
            break

        collateral_tracklets = project.chm.tracklets_in_frame(frame)
        collateral_sets.append(collateral_tracklets)

        min_end_frame = maxsize
        for tracklet in collateral_tracklets:
            min_end_frame = min(min_end_frame, tracklet.end_frame())

    return list(collateral_sets)


def get_median_areas(tracklets, project):
    import numpy as np
    areas = {}
    for t in tracklets:
        areas[t] = np.median([r.area() for r in t.r_gen(project.rm)])

    return areas


def remap_instances_to_ids(tracklets, collateral_sets, predecessors, successors, areas):
    new_tracklets = [t.id() for t in tracklets]
    new_collateral_sets = [[t.id() for t in tracklets] for tracklets in collateral_sets]
    new_predecessors = {t.id(): [t_p.id() for t_p in pred] for t, pred in predecessors.items()}
    new_successors = {t.id(): [t_p.id() for t_p in succ] for t, succ in successors.items()}
    new_areas = {t.id(): area for t, area in areas.items()}

    return new_tracklets, new_collateral_sets, new_predecessors, new_successors, new_areas


def relax_area_cost(cost, k, median_area, area_relaxation_coef):
    return max(0, cost - median_area * area_relaxation_coef * (k - 1))


def build_ilp(tracklets, collateral_sets, predecessors, successors, K, median_area, areas, remap_to_ids=True,
              gamma=2.0, tracklet_weights=None):
    prob = pulp.LpProblem("Cardinality classifier", pulp.LpMinimize)
    print("gamma: {}".format(gamma))

    if tracklet_weights is None:
        tracklet_weights = {t.id(): len(t) for t in tracklets}

    if remap_to_ids:
        tracklets, collateral_sets, predecessors, successors, areas = remap_instances_to_ids(tracklets,
                                                                                             collateral_sets,
                                                                                             predecessors,
                                                                                             successors,
                                                                                             areas)
    t_start = time.time()
    full_KK = range(0, K + 1)
    KK = range(1, K + 1)
    x = {}
    for t in tracklets:
        for k in full_KK:
            x[(t, k)] = pulp.LpVariable("tid:{}^{}".format(t, k), cat=pulp.LpBinary)
    print("var registration t: {}".format(time.time() - t_start))

    t_start = time.time()
    # objective function
    # prob += sum([x[(t, k)] * relax_area_cost(abs(k * median_area - areas[t]), k, median_area, area_relaxation_coef) *
    #              tracklets_len[t] for t in tracklets for k in KK]) \
    #         + sum([x[(t, 0)] * max(0, median_area - abs(median_area - areas[t])) * tracklets_len[t] * gamma for t in tracklets])  # closer to mean the worst..
    #         # + sum([x[(t, 0)] * not_used_penalty for t in tracklets])

    prob += sum([x[(t, k)] * abs(k * median_area - areas[t]) *
                 tracklet_weights[t] for t in tracklets for k in KK]) \
            + sum([x[(t, 0)] * max(0, median_area - abs(median_area - areas[t])) * tracklet_weights[t] * gamma for t in
                   tracklets])  # closer to mean the worst..
    print("objective function: {}".format(time.time() - t_start))

    t_start = time.time()
    # for each complete set
    for cs_i, cs in enumerate(collateral_sets):
        prob += sum([k * x[(t, k)] for t in cs for k in KK]) == K, "CS{}; ID perservation rule in complete sets".format(
            cs_i)

    # flow rule
    for t, pred in predecessors.items():
        prob += sum([k * x[(t_, k)] for t_ in pred for k in KK]) >= \
                sum([k * x[(t, k)] for k in KK])

    for t, succ in successors.items():
        prob += sum([k * x[(t_, k)] for t_ in succ for k in KK]) >= \
                sum([k * x[(t, k)] for k in KK])

    # for each tracklet - only one cardinality option is possible
    for t in tracklets:
        prob += sum([x[(t, k)] for k in full_KK]) == 1
    print("constraints: {}".format(time.time() - t_start))

    return prob, x


def generate_predecessor_map(tracklets, project):
    predecessors = {}

    for t in tracklets:
        for t_pred in t.entering_tracklets():
            if t not in predecessors:
                predecessors[t] = []

            predecessors[t].append(t_pred)

    return predecessors


def generate_successor_map(tracklets, project):
    successors = {}

    for t in tracklets:
        for t_pred in t.entering_tracklets():
            if t not in successors:
                successors[t] = []

            successors[t].append(t_pred)

    return successors


def solve(prob, x, print_ilp):
    if print_ilp:
        print(prob)

    import time

    t = time.time()
    status = prob.solve()
    print("Status:", pulp.LpStatus[prob.status])
    print("Total Cost = ", pulp.value(prob.objective))

    print("time: {}s".format(time.time() - t))
    print(status)

    cardinalities = {}
    for (id, card), var in x.items():
        # id, card = str(xx)[1:-1].split(',')
        # id = int(id)
        # card = int(card[1:])
        if var.varValue > 0:
            cardinalities[id] = card
        if id not in cardinalities:
            cardinalities[id] = 0

        # print(id, card, var, var.varValue)

    # print("\n")

    # for id in sorted(cardinalities.keys()):
    #     print("{}:\t{}".format(id, cardinalities[id]))

    return cardinalities


def build_ilp_and_solve(tracklets, collateral_sets, predecesors, successors, K, median_area, areas, remap_to_ids=True,
                        print_ilp=False, gamma=2.0, tracklet_weights=None):
    """
    Build the Integral linear program and solve the cardinality problem.

    :param tracklets:
    :param collateral_sets:
    :param predecesors:
    :param successors:
    :param K:
    :param median_area:
    :param areas:
    :param remap_to_ids:
    :param print_ilp:
    :param gamma:
    :param tracklet_weights:
    :return: dict, {tracklet id: number of objects}
    """
    t = time.time()
    ilp, variable_mapping = build_ilp(tracklets, collateral_sets, predecesors, successors, K, median_area, areas,
                                      remap_to_ids,
                                      gamma, tracklet_weights=tracklet_weights)
    print("ILP construction tooks: {}s".format(time.time() - t))

    if print_ilp:
        print(ilp)

    return solve(ilp, variable_mapping, print_ilp)


def eval(tracklets, cardinalities, cardinalities_gt):
    tracklets_len = {t.id(): len(t) for t in tracklets}

    # for coef in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for coef in [0]:
        # cardinalities = g_cardinality_clf.build_ilp_and_solve(tracklets, collateral_sets, predecessors, num_animals,
        #                                                       median_area, areas, area_relaxation_coef=coef)
        num_mistakes = 0

        tracklets_len_sum = 0

        for id in cardinalities_gt.keys():
            if cardinalities_gt[id] != cardinalities[id]:
                print("mistake, id: {}, correct cardinality: {}, estimated: {}".format(id, cardinalities_gt[id],
                                                                                       cardinalities[id]))
                num_mistakes += 1
                tracklets_len_sum += tracklets_len[id]

        print("COEF: {}, #mistakes: {}, len: {}".format(coef, num_mistakes, tracklets_len_sum))
        print("------------------------------\n")


def toy_example():
    class Test():
        def __init__(self, name):
            self.id_ = name

        def id(self):
            return self.id_

    a = Test('A')
    b = Test('B')
    c = Test('C')
    d = Test('D')

    collateral_sets = [[a.id(), b.id(), c.id()], [c.id(), d.id()]]
    tracklets = set([t for cs in collateral_sets for t in cs])
    predecesors = {d.id(): [a.id(), b.id()]}
    median_area = 100
    areas = {a.id(): 110, b.id(): 85, c.id(): 400, d.id(): 180}
    K = 2

    build_ilp_and_solve(tracklets, collateral_sets, predecesors, K, median_area, areas)


if __name__ == '__main__':
    toy_example()
