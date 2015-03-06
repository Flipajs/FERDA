__author__ = 'fnaiser'


def get_region_groups(regions):
    """
    Returns regions ids sorted into groups based on region label
    :param regions:
    :return: list of lists of region ids
    """
    prev = -1
    groups = []
    i = -1
    for r_id in range(len(regions)):
        r = regions[r_id]

        if r.label() > prev:
            prev = r.label()
            groups.append([r_id])
            i += 1
        else:
            groups[i].append(r_id)

    return groups


def best_margin(regions, r_ids):
    """
    For given list of ids selects region with max margin
    :param regions:
    :param r_ids:
    :return: (margin_value, region_id)
    """
    best_margin = -1
    best_margin_id = -1
    for r_id in r_ids:
        if regions[r_id].margin() > best_margin:
            best_margin = regions[r_id].margin()
            best_margin_id = r_id

    return best_margin, best_margin_id


def margin_filter(regions, groups, min_margin=0):
    """
    Selects region with best margin for each group
    Returns list of ids (one for each group).
    :param regions:
    :param groups:
    :param min_margin:
    :return:
    """
    ids = []
    for g in groups:
        margin, region_id = best_margin(regions, g)
        if margin > min_margin:
            ids.append(region_id)

    return ids