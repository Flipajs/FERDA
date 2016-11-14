__author__ = 'fnaiser'

import numpy as np

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

def get_region_groups_dict_(regions):
    prev = -1
    groups = []
    i = -1
    for r_id in range(len(regions)):
        r = regions[r_id]

        if r['label'] > prev:
            prev = r['label']
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

def best_margin_dict_(regions, r_ids):
    """
    For given list of ids selects region with max margin
    :param regions:
    :param r_ids:
    :return: (margin_value, region_id)
    """
    best_margin = -1
    best_margin_id = -1
    for r_id in r_ids:
        if regions[r_id]['margin'] > best_margin:
            best_margin = regions[r_id]['margin']
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

def margin_filter_dict_(regions, groups, min_margin=0):
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
        margin, region_id = best_margin_dict_(regions, g)
        if margin > min_margin:
            ids.append(region_id)

    return ids


def area_filter(regions, r_ids, min_area):
    ids = []
    for i in r_ids:
        if regions[i].area() > min_area:
            ids.append(i)

    return ids

def min_intensity_filter_dict_(regions, r_ids, min_intensity):
    ids = []
    for i in r_ids:
        if regions[i]['minI'] < min_intensity:
            ids.append(i)

    return ids


def is_child_of(child, parent, tolerance=0, tolerance_percents=0.1):
    if child.area() > parent.area():
        return False

    num_miss = 0
    max_num_miss = tolerance_percents*parent.area()

    ch_r = child.roi()
    p_r = parent.roi()
    if p_r.is_inside(ch_r.top_left_corner()) and p_r.is_inside(ch_r.bottom_right_corner(), tolerance=tolerance):
        img = np.zeros((p_r.height(), p_r.width()), dtype=np.bool)
        offset = np.array([p_r.y(), p_r.x()])
        p_pts = parent.pts() - offset
        img[p_pts[:,0], p_pts[:,1]] = True

        for p in child.pts() - offset:
            if not img[p[0], p[1]]:
                num_miss += 1

                if num_miss > max_num_miss:
                    return False

        return True
    else:
        return False


def children_filter(regions, indexes, tolerance=0):
    ids = []
    for r_id in indexes:
        is_child = False
        for parent_id in indexes:
            if r_id == parent_id:
                continue

            if is_child_of(regions[r_id], regions[parent_id]):
                is_child = True
                break

        if not is_child:
            ids.append(r_id)

    return ids


def antlikeness_filter(svm, thresh, regions, indexes):
    ids = []
    for r_id in indexes:
        p = svm.get_prob(regions[r_id])[1]

        if p > thresh:
            ids.append(r_id)

    return ids


