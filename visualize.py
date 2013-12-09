__author__ = 'flip'

import cv2
import utils as my_utils
from numpy import *


def draw_region(img, region, color):
    for r in region["rle"]:
        cv2.line(img, (r["col1"], r["line"]), (r["col2"], r["line"]), color, 1)


def draw_all_regions(img, regions):
    for i in range(0, len(regions)):
        draw_region(img, regions[i], (0, 255, 0))


def draw_selected_regions(img, regions, indexes, color):
    for i in range(0, len(indexes)):
        draw_region(img, regions[indexes[i]], color)


def draw_ants(img, ants, regions, filled):
    for i in range(len(ants)):
        c = ants[i].color

        for j in range(len(ants[i].history)):
            if j > 50:
                break

            path_strength = 1
            if j < 15:
                path_strength = 2

            cv2.circle(img, ants[i].history[j].position.int_tuple(), path_strength, c, -1)

        if ants[i].state.mser_id < 0:
            continue

        reg = regions[ants[i].state.mser_id]

        for j in range(len(reg["rle"])):
            a = reg["rle"][j]

            if filled:
                cv2.line(img, (a["col1"], a["line"]), (a["col2"], a["line"]), c, 1)
            else:
                cv2.circle(img, (a["col1"], a["line"]), 0, c, -1)
                cv2.circle(img, (a["col2"], a["line"]), 0, c, -1)

        cv2.line(img, ants[i].state.position.int_tuple(), ants[i].predicted_position(1).int_tuple(), (255, 255, 255), 1)

    return img


def draw_region_collection(img, regions, params, cols=10, rows=10, cell_size=50):
    collection = zeros((rows * cell_size, cols * cell_size, 3), dtype=uint8)
    border = cell_size

    img_ = zeros((shape(img)[0] + 2 * border, shape(img)[1] + 2 * border, 3), dtype=uint8)
    for i in range(len(regions)):
        img_[border:-border, border:-border] = img.copy()
        r = regions[i]
        c = (0, 255, 0)

        if r["flags"] == "arena_kill":
            c = (0, 0, 255)
        elif r["flags"] == "max_area_diff_kill":
            c = (0, 128, 255)
        elif r["flags"] == "better_mser_nearby_kill":
            c = (200, 255, 0)
        elif r["flags"] == "axis_kill":
            c = (200, 0, 255)

        draw_region(img_[border:-border, border:-border], r, c)

        #rate = my_utils.mser_main_axis_rate(r["sxy"], r["sxx"], r["syy"])
        #d_area = abs(r["area"] - params.avg_ant_area) / float(params.avg_ant_area)
        #print i, r["cx"], r["cy"], r["area"], d_area, params.avg_ant_area, rate, params.avg_ant_axis_ratio

        row = i / cols
        col = i % cols
        img_small = img_[border + r[
            "cy"] - cell_size / 2:border + r["cy"] + cell_size / 2, border + r["cx"] - cell_size / 2:border + r[
            "cx"] + cell_size / 2].copy()

        cv2.putText(img_small, str(i), (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.CV_AA)
        collection[row * cell_size:(row + 1) * cell_size, col * cell_size:(col + 1) * cell_size, :] = img_small

    return collection
