__author__ = 'fnaiser'

from core.region.distance_map import DistanceMap
##############
# all point lists are in format [y, x]
###

def fit(region, ants, num_of_iterations=10, use_settled_heuristics=True):
    d_map_region = DistanceMap(region.pts())

    if use_settled_heuristics:
        unsettled = get_unsettled()

        for i in range(num_of_iterations/2):
            for each unsetttled ant
                d_map_angs = compute_distance_map()

            t = estimate_transformation()
            for each unsettled ant
                apply_transformation()

            if termination_test():
                break

    for i in range(num_of_iterations):
        t = estimate_transformation()
        apply_transformation()
        if termination_test():
            break

    return ants, status


def estimate_transformation():

w