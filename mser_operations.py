__author__ = 'flip'

import sys
from libs import cyMser
import cv2
import my_utils
from time import time
from numpy import *


#mser flags>
# arena_kill
# max_area_diff_kill
# better_mser_nearby_kill
# axis_kill
class MserOperations():
    def __init__(self, params):
        self.mser = cyMser.PyMser()
        self.params = params
        self.msers_last_time = 0
        self.msers_sum_time = 0

    def process_image(self, img, intensity_threshold=256, ignore_arena=False):
        if img.shape[2] > 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img[:, :, 0]

        # cv2.imshow("before msers", gray)
        t0 = time()

        if intensity_threshold > 256:
            intensity_threshold = 256

        intensity_threshold = 256

        self.mser.set_min_margin(self.params.min_margin)
        self.mser.set_max_area(0.005)
        self.mser.process_image(gray, intensity_threshold)

        t1 = time()
        self.msers_last_time = (t1-t0)
        self.msers_sum_time += (t1-t0)

        regions = self.mser.get_regions()

        self.arena_filter(regions, ignore_arena)
        regions = self.prepare_regions(regions)
        groups = get_region_groups(regions)
        ids = margin_filter(regions, groups, self.params.min_margin)
        if self.params.skip_big_regions:
            ids = self.area_filter(regions, ids)

        if self.params.skip_high_intensity_regions > -1:
            ids = self.intensity_filter(regions, ids)

        return regions, ids

    def prepare_regions(self, regions):
        regions = self.count_thetas(regions)
        regions = self.count_axis(regions)
        regions = self.count_roi_corners(regions)

        return regions
    def count_thetas(self, regions):
        for r in regions:
            r['theta'] = my_utils.mser_theta(r["sxy"], r["sxx"], r["syy"])

        return regions

    def count_axis(self, regions):
        for r in regions:
            axis_ratio, a, b = my_utils.mser_main_axis_ratio(r["sxy"], r["sxx"], r["syy"])
            b_ = math.sqrt(r['area'] / (axis_ratio * math.pi))
            a_ = b_ * axis_ratio
            r['a'] = a_
            r['b'] = b_

        return regions

    def count_roi_corners(self, regions):
        for i in range(len(regions)):
            min_c = sys.maxint
            min_r = sys.maxint
            max_c = 0
            max_r = 0
            for r in regions[i]['rle']:
                if r['line'] < min_r:
                    min_r = r['line']
                if r['line'] > max_r:
                    max_r = r['line']
                if r['col1'] < min_c:
                    min_c = r['col1']
                if r['col2'] > max_c:
                    max_c = r['col2']

            regions[i]['roi_tl'] = [min_c, min_r]
            regions[i]['roi_br'] = [max_c, max_r]

        return regions

    def arena_filter(self, regions, ignore_arena=False):
        if ignore_arena:
            for i in range(len(regions)):
                regions[i]['flags'] = None

            return range(len(regions))

        indexes = []
        for i in range(len(regions)):
            reg = regions[i]
            reg["flags"] = "arena_kill"

            if my_utils.is_inside_ellipse(self.params.arena, my_utils.Point(reg['cx'], reg['cy'])):
                indexes.append(i)
                reg["flags"] = None

            #if reg["minI"] > 75:
            #    reg["flags"] = "minI_kill"

        return indexes

    def intensity_filter(self, regions, indexes):
        filtered_indexes = []

        for i in range(len(indexes)):
            ri = regions[indexes[i]]
            if ri['minI'] < self.params.skip_high_intensity_regions:
                filtered_indexes.append(indexes[i])

        return filtered_indexes

    def area_filter(self, regions, indexes):
        filtered_indexes = []

        for i in range(len(indexes)):
            ri = regions[indexes[i]]
            if self.params.skip_big_regions > 0:
                if ri['area'] < self.params.skip_big_regions_thresh:
                    filtered_indexes.append(indexes[i])
            else:
                if ri['area'] > -self.params.skip_big_regions_thresh:
                    filtered_indexes.append(indexes[i])
            #
            # d_area = ri["area"] / float(self.params.avg_ant_area)
            # if d_area < self.params.max_area_diff:
            #     ri["flags"] = "max_area_diff_kill_small"
            #     continue
            # else:
            #     d_area = float(ri["area"]) / float(self.params.avg_ant_area)
            #     if d_area > 1 + self.params.max_area_diff:
            #         ri["flags"] = "max_area_diff_kill_big"
            #         continue
            #
            # filtered_indexes.append(indexes[i])

        return filtered_indexes

    def axis_filter(self, regions, filtered_indexes):
        indexes = []
        for i in range(0, len(filtered_indexes)):
            ri = regions[filtered_indexes[i]]

            ratio, a, b = my_utils.mser_main_axis_ratio(ri['sxy'], ri['sxx'], ri['syy'])
            if abs(ratio - self.params.avg_ant_axis_ratio) < self.params.max_axis_ratio_diff:
                indexes.append(filtered_indexes[i])
            else:
                ri["flags"] = "axis_kill"

        return indexes

    def area_ratio_filter(self, regions, filtered_indexes):
        indexes = []
        for i in range(0, len(filtered_indexes)):
            ri = regions[filtered_indexes[i]]

            ratio = my_utils.mser_main_axis_ratio(ri['sxy'], ri['sxx'], ri['syy'])
            if abs(ratio - self.params.avg_ant_axis_ratio) < self.params.max_axis_ratio_diff:
                add = True

                for j in range(0, len(filtered_indexes)):
                    rj = regions[filtered_indexes[j]]
                    if my_utils.e_distance(my_utils.Point(ri['cx'], ri['cy']), my_utils.Point(rj['cx'], rj['cy'])) < 5:
                        if ri['area'] > rj['area']:
                            add = False
                            break

                if add:
                    indexes.append(filtered_indexes[i])
        return indexes


def region_size(self, rle):
    row_start = rle[0]['line']
    col_start = sys.maxint
    col_end = 0

    for l in rle:
        if l['col1'] < col_start:
            col_start = l['col1']
        if l['col2'] > col_end:
            col_end = l['col2']

    row_end = l['line']

    return row_start, col_start, row_end, col_end


def get_region_groups(regions):
    prev = -1
    groups = []
    i = -1
    for ridx in range(len(regions)):
        r = regions[ridx]
        if r["flags"] == "arena_kill":
            continue
        if r["flags"] == "minI_kill":
            continue

        if r["label"] > prev:
            prev = r["label"]
            groups.append([ridx])
            i += 1
        else:
            groups[i].append(ridx)

    return groups


def get_region_groups2(regions, check_flags=True):
    prev = -1
    groups = []
    groups_avg_pos = []
    i = -1
    for ridx in range(len(regions)):
        r = regions[ridx]
        if check_flags:
            if r["flags"] == "arena_kill":
                continue
            if r["flags"] == "max_area_diff_kill_small":
                continue
            if r["flags"] == "minI_kill":
                continue

        if r["label"] > prev:
            prev = r["label"]
            groups.append([ridx])
            groups_avg_pos.append([r["cx"], r["cy"]])
            i += 1
        else:
            groups[i].append(ridx)
            groups_avg_pos[i][0] += r["cx"]
            groups_avg_pos[i][1] += r["cy"]

    for i in range(len(groups)):
        groups_avg_pos[i][0] /= len(groups[i])
        groups_avg_pos[i][1] /= len(groups[i])

    return groups, groups_avg_pos


def margin_filter(regions, groups, min_margin):
    ids = []
    for g in groups:
        margin, region_id = my_utils.best_margin(regions, g)
        if margin > min_margin:
            ids.append(region_id)

    return ids


def prepare_region_for_splitting(region, img, reduce_factor):
    #pxs = [[0, 0, 0] for i in range(region['area'])] #x y intensity
    pxs = zeros((region['area'], 3), dtype=int)

    i = 0
    for rle in region['rle']:
        for c in range(rle['col1'], rle['col2'] + 1):
            pxs[i][0] = c
            pxs[i][1] = rle['line']
            pxs[i][2] = img[rle['line']][c][0]
            i += 1

    #pxs = np.array(pxs)
    pxs = pxs[pxs[:,2].argsort()]

    crop = region['area'] - region['area'] * reduce_factor - 1
    return pxs[0:crop, 0:2]


def region_roi_img(region):
    rows = region['roi_br'][1] - region['roi_tl'][1]
    cols = region['roi_br'][0] - region['roi_tl'][0]

    img = zeros((rows+1, cols+1, 1), dtype=uint8)

    for r in region['rle']:
        row = r['line'] - region['roi_tl'][1]
        col1 = r['col1'] - region['roi_tl'][0]
        col2 = r['col2'] - region['roi_tl'][0]
        img[row][col1:col2+1] = 255

    return img


def is_child_of(child, parent):
    if child['area'] > parent['area']:
        return False

    if child['roi_tl'][0] >= parent['roi_tl'][0] and child['roi_tl'][1] >= parent['roi_tl'][1] and\
                    child['roi_br'][0] <= parent['roi_br'][0] and child['roi_br'][1] <= parent['roi_br'][1]:
        img = region_roi_img(parent)
        for r in child['rle']:
            row = r['line'] - parent['roi_tl'][1]
            col1 = r['col1'] - parent['roi_tl'][0]
            col2 = r['col2'] - parent['roi_tl'][0]

            if img[row][col1] == 0 or img[row][col2] == 0:
                return False

        return True
    else:
        return False    # full intersection is impossible


def filter_out_children_of_best_margin(regions, indexes):
    ids = []
    for r_id in indexes:
        is_child = False
        for parent_id in indexes:
            if r_id == parent_id:
                continue

            if is_child_of(regions[r_id], regions[parent_id]):
                is_child = True
                continue
        if not is_child:
            ids.append(r_id)

    return ids