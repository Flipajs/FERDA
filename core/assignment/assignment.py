__author__ = 'filip@naiser.cz'

import score


def ant2region_assignment(ant, regions, params, score_functions, expression, r_idx=None):
    """ Finds best assignment based on evaluation of expression.

    Args:
        ant: class Ant from ants.py
        regions: list of dicts describing MSERs regions
        params: class Params from experiment_parameters.py
        score_functions: list of function names(str)
        expression: str with expression to be evaluated.
        r_idx: list of int (region ids), only regions in this list are processed. If None, all regions are used

    Returns:
        id of region with highest score for ant assignment based on score function described by expression


    Example:
        score_functions = ['f1', 'f2'], where f1 and f2 are defined in score.py
        expression should be in form '2*f1 + sin(f2)' or '2*$0 + sin($1)' using functions from math
    """

    if r_idx is None:
        r_idx = range(len(regions))

    best_s = -1
    best_r_id = -1

    for r_id in r_idx:
        r = regions[r_id]
        s = score.evolve_score_functions(r, ant, params, score_functions, expression)

        if s > best_s:
            best_s = s
            best_r_id = r_id

    return best_r_id