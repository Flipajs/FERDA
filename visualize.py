import mser_operations

__author__ = 'flip'

import cv2
import my_utils
from numpy import *
import score
import mser_operations

def draw_region(img, region, color, contour=False):
    if 'splitted' in region:
        color = (220, 100, 0)
        for pt in region["points"]:
            if contour:
                cv2.circle(img, (pt[0], pt[1]), 0, color, -1)
            else:
                cv2.line(img, (pt[0], pt[1]), (pt[0], pt[1]), color, 1)
    elif "contour" in region:
        color = (220, 0, 255)
        for pt in region["points"]:
            if contour:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 0, color, -1)
            else:
                cv2.line(img, (int(pt[0]), int(pt[1])), (int(pt[0]), int(pt[1])), color, 1)
    else:
        for r in region["rle"]:
            if contour:
                cv2.circle(img, (r['col1'], r['line']), 0, color, -1)
                cv2.circle(img, (r['col2'], r['line']), 0, color, -1)
            else:
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

        radius = 1
        for j in range(history+1, min(len(ants[i].history), 250)):
            path_strength = radius-2
            if path_strength < 0:
                path_strength = 0
            if j < 15:
                path_strength = radius-1
                if path_strength < 1:
                    path_strength = 1


            cv2.circle(img, ants[i].history[j].position.int_tuple(), path_strength, c, -1)


        mser_id = ants[i].state.mser_id
        if history > 0:
            mser_id = ants[i].history[history-1].mser_id

        if mser_id < 0:
            continue

        reg = regions[mser_id]

        if filled:
            for j in range(len(reg["rle"])):
                a = reg["rle"][j]

                if filled:
                    cv2.line(img, (a["col1"], a["line"]), (a["col2"], a["line"]), c, 1)
                else:
                    cv2.circle(img, (a["col1"], a["line"]), 0, c, -1)
                    cv2.circle(img, (a["col2"], a["line"]), 0, c, -1)
        else:
            #if history == 0:
            #    cv2.line(img, ants[i].state.position.int_tuple(), ants[i].predicted_position(1).int_tuple(), (255, 255, 255), 1)

            #if len(ants[0].history) > 0:
            #    cv2.line(img, ants[i].history[0].position.int_tuple(), ants[i].state.position.int_tuple(), (0, 255, 255), 2)

            #c = (0, 0, 0)
            #if ants[i].state.orientation:
            #    c = (255, 255, 255)

            a = ants[i].state
            if history > 0:
                a = ants[i].history[history-1]

            cv2.circle(img, a.position.int_tuple(), radius+2, (255,255,255), -1)
            cv2.circle(img, a.head.int_tuple(), radius+1, (255,255,255), -1)
            cv2.circle(img, a.back.int_tuple(), radius+1, (255,255,255), -1)
            cv2.circle(img, a.head.int_tuple(), radius, c, -1)
            cv2.circle(img, a.back.int_tuple(), radius, c, -1)
            cv2.circle(img, a.position.int_tuple(), radius+1, c, -1)

    return img


def draw_region_collection(img, regions, params, cols=15, rows=10, cell_size=50):
    if len(regions) > rows*cols:
        rows = math.ceil(len(regions) / float(cols))

    collection = zeros((rows * cell_size, cols * cell_size, 3), dtype=uint8)
    border = cell_size

    img_ = zeros((shape(img)[0] + 2 * border, shape(img)[1] + 2 * border, 3), dtype=uint8)
    for i in range(len(regions)):
        img_[border:-border, border:-border] = img.copy()
        r = regions[i]
        if r["cx"] == inf or r["cy"] == inf:
            continue

        c = (0, 255, 0)

        if r["parent_label"] > -1:
            c = (0, 0, 255)

        # if r["flags"] == "arena_kill":
        #     c = (0, 0, 255)
        # elif r["flags"] == "max_area_diff_kill_small":
        #     c = (0, 100, 200)
        # elif r["flags"] == "max_area_diff_kill_big":
        #     c = (0, 128, 255)
        # elif r["flags"] == "better_mser_nearby_kill":
        #     c = (200, 255, 0)
        # elif r["flags"] == "axis_kill":
        #     c = (200, 0, 255)

        draw_region(img_[border:-border, border:-border], r, c)

        row = i / cols
        col = i % cols

        img_small = img_[border + r[
            "cy"] - cell_size / 2:border + r["cy"] + cell_size / 2, border + r["cx"] - cell_size / 2:border + r[
            "cx"] + cell_size / 2].copy()

        cv2.putText(img_small, str(r['label']), (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_small, str(i), (3, 20), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_small, str(r['parent_label']), (3, 30), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        collection[row * cell_size:(row + 1) * cell_size, col * cell_size:(col + 1) * cell_size, :] = img_small

    return collection

def draw_region_best_margins_collection(img, regions, indexes, ants, cols=5, cell_size=70):
    rows = int(math.ceil(len(indexes) / cols)) + 1

    collection = zeros((rows * cell_size, (cols * cell_size), 3), dtype=uint8)
    border = cell_size

    img_ = zeros((shape(img)[0] + 2 * border, shape(img)[1] + 2 * border, 3), dtype=uint8)

    for row in range(rows):
        for col in range(cols):
            index = row*cols + col
            if index >= len(indexes):
                break

            img_[border:-border, border:-border] = img.copy()
            r = regions[indexes[index]]
            if r["cx"] == inf or r["cy"] == inf:
                continue

            c = (230, 230, 230)
            for a in ants:
                if a.state.mser_id == indexes[index]:
                    c = a.color

            draw_region(img_[border:-border, border:-border], r, c, contour=False)


            img_small = img_[border + r["cy"] - cell_size / 2:border + r["cy"] + cell_size / 2, border + r["cx"] - cell_size / 2:border + r["cx"] + cell_size / 2].copy()

            cv2.putText(img_small, str(indexes[row*cols + col]), (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            collection[(row) * cell_size:((row) + 1) * cell_size, (col) * cell_size:((col) + 1) * cell_size, :] = img_small

    return collection


def draw_region_group_collection(img, regions, groups, params, cell_size=70):
    rows = int(math.ceil(len(groups) / 2.))
    cols = 0
    for g in groups:
        if len(g) > cols:
            cols = len(g)

    num_strip = 20
    collection = zeros((rows * cell_size, 2*(cols * cell_size) + num_strip + cell_size, 3), dtype=uint8)
    border = cell_size

    col_p = 0
    row_p = 0

    img_ = zeros((shape(img)[0] + 2 * border, shape(img)[1] + 2 * border, 3), dtype=uint8)
    counter = 0
    for row in range(len(groups)):
        if row >= rows:
                row_p = -rows
                col_p = cols+1

        cv2.putText(collection, str(row), (3 + cell_size*col_p, 30 + cell_size*(row+row_p)), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        best_id = -1
        margins = [0]*len(groups[row])
        for col in range(len(groups[row])):
            r = regions[groups[row][col]]

            margins[col] = r['margin']

        best_margin_id = argmax(array(margins))


        for col in range(len(groups[row])):

            img_[border:-border, border:-border] = img.copy()
            r = regions[groups[row][col]]
            if r["cx"] == inf or r["cy"] == inf:
                continue

            c = (0, 255, 0)

            if best_margin_id == col and margins[best_margin_id] != 0:
                c = (0, 138, 212)

            draw_region(img_[border:-border, border:-border], r, c)


            img_small = img_[border + r[
                "cy"] - cell_size / 2:border + r["cy"] + cell_size / 2, border + r["cx"] - cell_size / 2:border + r[
                "cx"] + cell_size / 2].copy()


            _, a, b = my_utils.mser_main_axis_ratio(r["sxy"], r["sxx"], r["syy"])
            a, b = my_utils.count_head_tail(r["area"], a, b)

            #cv2.putText(img_small, str(groups[row][col]), (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            #if col == best_id:
            #cv2.putText(img_small, str(r['a'])[0:5], (3, 35), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
            #cv2.putText(img_small, str(r['area']/(r['a']*2))[0:5], (35, 35), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
            #cv2.putText(img_small, str(r['area']), (3, 55), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
            #cv2.putText(img_small, str(r['maxI']), (3, 45), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 255), 1, cv2.LINE_AA)
            #cv2.putText(img_small, str(r['margin']), (3, 65), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
            collection[(row + row_p) * cell_size:((row + row_p) + 1) * cell_size, num_strip + (col + col_p) * cell_size:num_strip + ((col + col_p) + 1) * cell_size, :] = img_small

    #print "COUNTER: ", counter

    return collection

def draw_region_group_collection2(img, regions, groups, params, cell_size=70):
    rows = int(math.ceil(len(groups) / 2.))
    cols = 0
    for g in groups:
        if len(g) > cols:
            cols = len(g)

    num_strip = 20
    collection = zeros((rows * cell_size, 2*(cols * cell_size) + num_strip + cell_size, 3), dtype=uint8)
    border = cell_size

    col_p = 0
    row_p = 0

    img_ = zeros((shape(img)[0] + 2 * border, shape(img)[1] + 2 * border, 3), dtype=uint8)
    counter = 0
    for row in range(len(groups)):
        if row >= rows:
                row_p = -rows
                col_p = cols+1

        cv2.putText(collection, str(row), (3 + cell_size*col_p, 30 + cell_size*(row+row_p)), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        best_id = -1
        margins = [0]*len(groups[row])
        for col in range(len(groups[row])):
            r = regions[groups[row][col]]

            margins[col] = r['margin']

        best_margin_id = argmax(array(margins))


        for col in range(len(groups[row])):

            img_[border:-border, border:-border] = img.copy()
            r = regions[groups[row][col]]
            if r["cx"] == inf or r["cy"] == inf:
                continue

            c = (0, 255, 0)

            #if best_margin_id == col and margins[best_margin_id] != 0:
            #    c = (0, 138, 212)

            draw_region(img_[border:-border, border:-border], r, c)


            img_small = img_[border + r[
                "cy"] - cell_size / 2:border + r["cy"] + cell_size / 2, border + r["cx"] - cell_size / 2:border + r[
                "cx"] + cell_size / 2].copy()


            _, a, b = my_utils.mser_main_axis_ratio(r["sxy"], r["sxx"], r["syy"])
            a, b = my_utils.count_head_tail(r["area"], a, b)

            #cv2.putText(img_small, str(groups[row][col]), (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            #if col == best_id:
            #cv2.putText(img_small, str(r['a'])[0:5], (3, 35), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
            #cv2.putText(img_small, str(r['area']/(r['a']*2))[0:5], (35, 35), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
            #cv2.putText(img_small, str(r['area']), (3, 55), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
            #cv2.putText(img_small, str(r['maxI']), (3, 45), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 255), 1, cv2.LINE_AA)
            #cv2.putText(img_small, str(r['margin']), (3, 65), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_small, str(r['margin']), (3, 15), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            collection[(row + row_p) * cell_size:((row + row_p) + 1) * cell_size, num_strip + (col + col_p) * cell_size:num_strip + ((col + col_p) + 1) * cell_size, :] = img_small

    #print "COUNTER: ", counter

    return collection

def draw_assignment_problem(prev_img, img, ants, regions, indexes, params, cell_size=60, history=0):
    num_best = 3
    best_regions_id = [[-1 for x in range(num_best)] for x in range(len(ants))]
    #best_regions_id = [len(ants)][num_best]

    for a_id in range(len(ants)):
        help = [0] * len(indexes)
        for i in range(len(indexes)):
            #_, region_id = my_utils.best_margin(regions, indexes[i])

            help[i] = score.count_node_weight(ants[a_id], regions[indexes[i]], params)

        s = argsort(array(help))[::-1]
        for i in range(num_best):
            #_, region_id = my_utils.best_margin(regions, groups[s[i]])
            best_regions_id[a_id][i] = indexes[s[i]]

    color_stripe_width = 7

    c_width = color_stripe_width + cell_size + (num_best) * cell_size * 2
    c_height = len(ants) * cell_size
    collection = zeros((c_height, c_width, 3), dtype=uint8)
    border = cell_size
    img_ = zeros((shape(img)[0] + 2 * border, shape(img)[1] + 2 * border, 3), dtype=uint8)

    line_color = (53, 53, 53)


    for i in range(len(ants)):
        img_[border:-border, border:-border] = prev_img.copy()
        a = ants[i].state
        if history > 0:
            a = ants[i].history[history-1]

        y = a.position.y
        x = a.position.x
        c = img_[border + y - cell_size / 2:border + y + cell_size / 2, border + x - cell_size / 2:border + x + cell_size / 2].copy()

        color = [255, 255, 255]

        collection[i * cell_size:(i + 1)*cell_size, color_stripe_width:cell_size+color_stripe_width, :] = c

        collection[i * cell_size:(i + 1)*cell_size, 0:color_stripe_width, :] = ants[i].color

        if i > 0:
           cv2.line(collection, (0, i*cell_size), (c_width - 1, i*cell_size), line_color, 1)

        w = cell_size + color_stripe_width
        color = tuple(color)
        w2 = w + 135
        h = i*(cell_size)
        font_scale = 0.80
        font = cv2.FONT_HERSHEY_PLAIN
        thick = 1
        h1 = 11
        h2 = 26
        h3 = 40
        h4 = 53

        img_ = zeros((shape(img)[0] + 2 * border, shape(img)[1] + 2 * border, 3), dtype=uint8)


        for best_id in range(num_best):
            r = regions[best_regions_id[i][best_id]]
            img_[border:-border, border:-border] = img.copy()
            draw_region(img_[border:-border, border:-border], r, (180, 100, 0), contour=False)
            img_small = img_[border + r[
                "cy"] - cell_size / 2:border + r["cy"] + cell_size / 2, border + r["cx"] - cell_size / 2:border + r[
                "cx"] + cell_size / 2].copy()

            collection[i * cell_size:(i + 1)*cell_size, color_stripe_width + cell_size * (2 * (best_id + 1) - 1):color_stripe_width + cell_size * (2 * (best_id + 1)), :] = img_small

            sc = score.count_node_weight(ants[i], r, params)
            s_sc =  "%.9f" % (sc*100)
            cv2.putText(collection, s_sc[0:7], (3 + color_stripe_width + cell_size * (2 * (best_id + 1)), i * cell_size + 12), font, font_scale, color, 1, cv2.LINE_AA)

            th = score.theta_change_prob(ants[i], r)
            s_th =  "t %.9f" % (th*100)
            cv2.putText(collection, s_th[0:7], (3 + color_stripe_width + cell_size * (2 * (best_id + 1)), i * cell_size + 25), font, font_scale, color, 1, cv2.LINE_AA)

            po = score.position_prob(ants[i], r, params)
            s_po =  "p %.9f" % (po*100)
            cv2.putText(collection, s_po[0:7], (3 + color_stripe_width + cell_size * (2 * (best_id + 1)), i * cell_size + 37), font, font_scale, color, 1, cv2.LINE_AA)

            ab = score.a_area_prob(r, params)
            s_ab =  "a %.9f" % (ab*100)
            cv2.putText(collection, s_ab[0:7], (3 + color_stripe_width + cell_size * (2 * (best_id + 1)), i * cell_size + 50), font, font_scale, color, 1, cv2.LINE_AA)


        #cv2.putText(collection, ants[i].name, (w + 3, h1+h), font, font_scale, color, 1, cv2.LINE_AA)
        #cv2.putText(collection, "[" + str(a.position.x)[0:6] + ", " + str(a.position.y)[0:6] + "]", (w + 3, h2+h), font, font_scale, color, thick, cv2.LINE_AA)
        #cv2.putText(collection, "theta: " + str(a.theta*180/3.14)[0:6], (w + 3, h3+h), font, font_scale, color, thick, cv2.LINE_AA)
        #cv2.putText(collection, "area: " + str(a.area), (w + 3, h4+h), font, font_scale, color, thick, cv2.LINE_AA)
        #cv2.putText(collection, "p: " + str(a.score)[0:6], (w2, h1+h), font, font_scale, color, thick, cv2.LINE_AA)
        #cv2.putText(collection, "[" + str(a.a)[0:6] + ", " + str(a.b)[0:6] + "]", (w2, h2+h), font, font_scale, color, thick, cv2.LINE_AA)
        #cv2.putText(collection, str(a.a / a.b)[0:6], (w2, h3+h), font, font_scale, color, thick, cv2.LINE_AA)
        #cv2.putText(collection, str(a.mser_id), (w2, h4+h), font, font_scale, color, thick, cv2.LINE_AA)

    cv2.line(collection, (color_stripe_width - 1, 0), (color_stripe_width - 1, c_height - 1), line_color, 1)
    cv2.line(collection, (color_stripe_width + cell_size - 1, 0), (color_stripe_width + cell_size - 1, c_height - 1), line_color, 1)
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

        color = [255, 255, 255]

        if a.collision_predicted:
            color[0] = 0
            color[1] = 128

            c[:, :, 1] += 17
            c[:, :, 2] += 35

        if a.lost:
            color[0] = round(color[0] * 0.7)
            color[1] = round(color[1] * 0.7)
            color[2] = round(color[2] * 0.7)

            collection[i * cell_size:(i + 1)*cell_size, color_stripe_width:cell_size+color_stripe_width, :] = c*0.5
        else:
            collection[i * cell_size:(i + 1)*cell_size, color_stripe_width:cell_size+color_stripe_width, :] = c

        collection[i * cell_size:(i + 1)*cell_size, 0:color_stripe_width, :] = ants[i].color

        if i > 0:
           cv2.line(collection, (0, i*cell_size), (c_width - 1, i*cell_size), line_color, 1)

        w = cell_size + color_stripe_width
        color = tuple(color)
        w2 = w + 135
        h = i*(cell_size)
        font_scale = 0.80
        font = cv2.FONT_HERSHEY_PLAIN
        thick = 1
        h1 = 11
        h2 = 26
        h3 = 40
        h4 = 53

        cv2.putText(collection, ants[i].name, (w + 3, h1+h), font, font_scale, color, 1, cv2.LINE_AA)
        cv2.putText(collection, "[" + str(a.position.x)[0:6] + ", " + str(a.position.y)[0:6] + "]", (w + 3, h2+h), font, font_scale, color, thick, cv2.LINE_AA)
        cv2.putText(collection, "theta: " + str(a.theta*180/3.14)[0:6], (w + 3, h3+h), font, font_scale, color, thick, cv2.LINE_AA)
        cv2.putText(collection, "area: " + str(a.area), (w + 3, h4+h), font, font_scale, color, thick, cv2.LINE_AA)
        cv2.putText(collection, "p: " + str(a.score)[0:6], (w2, h1+h), font, font_scale, color, thick, cv2.LINE_AA)
        cv2.putText(collection, "[" + str(a.a)[0:6] + ", " + str(a.b)[0:6] + "]", (w2, h2+h), font, font_scale, color, thick, cv2.LINE_AA)
        if a.b > 0:
            cv2.putText(collection, str(a.a / a.b)[0:6], (w2, h3+h), font, font_scale, color, thick, cv2.LINE_AA)
        else:
            cv2.putText(collection, str(0), (w2, h3+h), font, font_scale, color, thick, cv2.LINE_AA)

        cv2.putText(collection, str(a.mser_id), (w2, h4+h), font, font_scale, color, thick, cv2.LINE_AA)

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

