__author__ = 'flip'

import cv2
import my_utils as my_utils
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


def draw_ants(img, ants, regions, filled, history=0):
    for i in range(len(ants)):
        c = ants[i].color

        for j in range(history, len(ants[i].history)):
            if j > 50:
                break

            path_strength = 0
            if j < 15:
                path_strength = 1

            cv2.circle(img, ants[i].history[j].position.int_tuple(), path_strength, c, -1)

        mser_id = ants[i].state.mser_id
        if history > 0:
            mser_id = ants[i].history[history-1].mser_id

        if mser_id < 0:
            continue

        reg = regions[mser_id]

        #for j in range(len(reg["rle"])):
        #    a = reg["rle"][j]
        #
        #    if filled:
        #        cv2.line(img, (a["col1"], a["line"]), (a["col2"], a["line"]), c, 1)
        #    else:
        #        cv2.circle(img, (a["col1"], a["line"]), 0, c, -1)
        #        cv2.circle(img, (a["col2"], a["line"]), 0, c, -1)

        if history == 0:
            cv2.line(img, ants[i].state.position.int_tuple(), ants[i].predicted_position(1).int_tuple(), (255, 255, 255), 1)

        #c = (0, 0, 0)
        #if ants[i].state.orientation:
        #    c = (255, 255, 255)

        a = ants[i].state
        if history > 0:
            a = ants[i].history[history-1]

        cv2.circle(img, a.position.int_tuple(), 3, (255,255,255), -1)
        cv2.circle(img, a.head.int_tuple(), 2, (255,255,255), -1)
        cv2.circle(img, a.back.int_tuple(), 2, (255,255,255), -1)
        cv2.circle(img, a.head.int_tuple(), 1, c, -1)
        cv2.circle(img, a.back.int_tuple(), 1, c, -1)
        cv2.circle(img, a.position.int_tuple(), 2, c, -1)

    return img


def draw_region_collection(img, regions, params, cols=10, rows=10, cell_size=50):
    collection = zeros((rows * cell_size, cols * cell_size, 3), dtype=uint8)
    border = cell_size

    img_ = zeros((shape(img)[0] + 2 * border, shape(img)[1] + 2 * border, 3), dtype=uint8)
    for i in range(len(regions)):
        img_[border:-border, border:-border] = img.copy()
        r = regions[i]
        if r["cx"] == inf or r["cy"] == inf:
            continue

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

        row = i / cols
        col = i % cols

        img_small = img_[border + r[
            "cy"] - cell_size / 2:border + r["cy"] + cell_size / 2, border + r["cx"] - cell_size / 2:border + r[
            "cx"] + cell_size / 2].copy()

        cv2.putText(img_small, str(i), (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.CV_AA)
        collection[row * cell_size:(row + 1) * cell_size, col * cell_size:(col + 1) * cell_size, :] = img_small

    return collection


def draw_ants_collection(img, ants, cell_size=60, history=0):
    c_width = 6*cell_size
    c_height = len(ants) * cell_size
    collection = zeros((c_height, c_width, 3), dtype=uint8)
    border = cell_size
    img_ = zeros((shape(img)[0] + 2 * border, shape(img)[1] + 2 * border, 3), dtype=uint8)
    img_[border:-border, border:-border] = img.copy()

    line_color = (53, 53, 53)

    color_stripe_width = 7

    for i in range(len(ants)):
        a = ants[i].state
        if history > 0:
            a = ants[i].history[history-1]

        y = a.position.y
        x = a.position.x
        c = img_[border + y - cell_size / 2:border + y + cell_size / 2, border + x - cell_size / 2:border + x + cell_size / 2].copy()

        if a.lost:
            collection[i * cell_size:(i + 1)*cell_size, color_stripe_width:cell_size+color_stripe_width, :] = c*0.5
        else:
            collection[i * cell_size:(i + 1)*cell_size, color_stripe_width:cell_size+color_stripe_width, :] = c

        collection[i * cell_size:(i + 1)*cell_size, 0:color_stripe_width, :] = ants[i].color

        if i > 0:
           cv2.line(collection, (0, i*cell_size), (c_width - 1, i*cell_size), line_color, 1)

        w = cell_size + color_stripe_width
        w2 = w + 135
        h = i*(cell_size)
        font_scale = 0.80
        font = cv2.FONT_HERSHEY_PLAIN
        thick = 1
        h1 = 11
        h2 = 26
        h3 = 40
        h4 = 53
        cv2.putText(collection, ants[i].name, (w + 3, h1+h), font, font_scale, (255, 255, 255), thickness=1, linetype=cv2.CV_AA)
        cv2.putText(collection, "[" + str(a.position.x)[0:6] + ", " + str(a.position.y)[0:6] + "]", (w + 3, h2+h), font, font_scale, (255, 255, 255), thickness=thick, linetype=cv2.CV_AA)
        cv2.putText(collection, "theta: " + str(a.theta*180/3.14)[0:6], (w + 3, h3+h), font, font_scale, (255, 255, 255), thickness=thick, linetype=cv2.CV_AA)
        cv2.putText(collection, "area: " + str(a.area), (w + 3, h4+h), font, font_scale, (255, 255, 255), thickness=thick, linetype=cv2.CV_AA)
        cv2.putText(collection, "p: " + str(a.score)[0:6], (w2, h1+h), font, font_scale, (255, 255, 255), thickness=thick, linetype=cv2.CV_AA)
        cv2.putText(collection, "[" + str(a.a)[0:6] + ", " + str(a.b)[0:6] + "]", (w2, h2+h), font, font_scale, (255, 255, 255), thickness=thick, linetype=cv2.CV_AA)
        cv2.putText(collection, str(a.a / a.b)[0:6], (w2, h3+h), font, font_scale, (255, 255, 255), thickness=thick, linetype=cv2.CV_AA)
        cv2.putText(collection, str(a.mser_id), (w2, h4+h), font, font_scale, (255, 255, 255), thickness=thick, linetype=cv2.CV_AA)

    cv2.line(collection, (color_stripe_width - 1, 0), (color_stripe_width - 1, c_height - 1), line_color, 1)
    return collection


def draw_collision_risks(img, ants, collisions, history):
    for c in collisions:
        a1 = ants[c[0]].state
        a2 = ants[c[1]].state
        if history > 0:
            a1 = ants[c[0]].history[history-1]
            a2 = ants[c[1]].history[history-1]

        if c[3] % 3 == 0:
            p2 = a2.head
        elif c[3] % 3 == 1:
            p2 = a2.position
        else:
            p2 = a2.back

        if c[3] < 3:
            p1 = a1.head
        elif c[3] < 6:
            p1 = a1.position
        else:
            p1 = a1.back

        cv2.line(img, p1.int_tuple(), p2.int_tuple(), (0, 0, 255), 1)


    return img