import math

__author__ = 'flip'

import my_utils as my_utils
from collections import deque
import copy

ant_names = ["Arnold", "Bob", "Cenek", "Dusan", "Emil", "Ferda", "Gustav", "Hugo", "Igor", "Julius", "Kamil",
                 "Ludek", "Milos", "Narcis", "Oliver", "Prokop", "Quido", "Rene", "Servac", "Tadeas", "1", "2",
                 "3", "4", "5", "6", "7", "8", "9", "10"]

ant_colors = [(145, 95, 22), (54, 38, 227), (0, 191, 255), (204, 102, 153), (117, 149, 105),
                  (0, 182, 141), (255, 255, 0), (32, 83, 78), (0, 238, 256), (128, 127, 0),
                  (190, 140, 238), (32, 39, 89), (99, 49, 222), (139, 0, 0), (0, 0, 139),
                  (60, 192, 3), (0, 79, 255), (128, 128, 128), (255, 255, 255), (0, 0, 0),
                  (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                  (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]

class AntState():
    position = [0, 0]
    theta = 0
    orientation = 0 #if 1... means theta + 180
    head = [0, 0]
    back = [0, 0]
    a = 0
    b = 0
    axis_ratio = 0
    area = -1
    size = my_utils.Size(0, 0)
    mser_id = -1
    info = None
    lost = False
    lost_time = 0

    def __init__(self):
        pass


class Ant():
    def __init__(self, id):
        self.id = id
        self.area_weighted = -1
        self.name = ant_names[id]
        self.color = ant_colors[id]
        self.state = AntState()
        self.history = deque()
        pass

    def velocity(self, history_depth):
        if len(self.history) == 0:
            return (0, 0)

        pos_ti = self.state.position
        velocity = [0, 0]
        counter = 0

        for i in range(history_depth):
            if i > len(self.history):
                break

            pos_ti_minus1 = self.history[i]
            velocity[0] += pos_ti[0] - pos_ti_minus1.position[0]
            velocity[1] += pos_ti[1] - pos_ti_minus1.position[1]

            counter += 1

            pos_ti = pos_ti_minus1.position

        velocity[0] /= counter
        velocity[1] /= counter

        return velocity

    def predicted_position(self, history_depth):
        pred = self.velocity(history_depth)
        return [self.state.position[0] + pred[0], self.state.position[0] + pred[1]]

    #returns stability (1 - #losts/history_depth)
    def stability(self, history_depth):
        if len(self.history) <= history_depth:
            return 0

        counter = 0
        for i in range(history_depth):
            if not self.history[i].lost:
                counter += 1

        return counter / history_depth

    def estimate_orientation(self, region):
        c = self.state.position
        theta = self.state.theta

        best_head = (c[0], c[1])
        best_head_val = c[0]
        best_back = (c[0], c[1])
        best_back_val = c[0]

        wide = True
        if theta > 45 < 135:
            wide = False
            best_head_val = c[1]
            best_back_val = c[1]

        for r in region["rle"]:
            h = r["line"]
            l = r["col1"]
            r = r["col2"]

            alpha = math.asin((h-c[1]) / math.sqrt(pow(l - c[0], 2) + pow(h - c[1], 2)))
            beta = math.asin((h-c[1]) / math.sqrt(pow(r - c[0], 2) + pow(h - c[1], 2)))

            pos = l + round(beta/(beta+alpha) * (r - l))
            if pos < l > r:
                continue

            comp = h
            if wide:
                comp = pos
                
            if comp > best_head_val:
                best_head_val = comp
                best_head = (pos, h)

            if comp < best_back_val:
                best_back_val = comp
                best_back = (pos, h)

        self.state.head = best_head
        self.state.back = best_back

        self.state.orientation = 1
        if my_utils.e_distance(c, best_head) > my_utils.e_distance(c, best_back):
            self.state.orientation = 0

    def buffer_history(self, first_frame = 0, last_frame = -1):
        if last_frame > len(self.history):
            last_frame = -1
        if last_frame == -1:
            last_frame = len(self.history)

        history_len = last_frame - first_frame
        x = [0] * history_len
        y = [0] * history_len
        a = [0] * history_len
        b = [0] * history_len
        theta = [0] * history_len
        for i in range(first_frame, last_frame):
            state = self.history[i]
            pos = state.position
            x[i] = pos[0]
            y[i] = pos[1]
            a[i] = state.a
            b[i] = state.b

            theta[i] = state.theta

        a = {'x': x,
             'y': y,
             'theta': theta,
             'a': a,
             'b': b,
             'id': self.id,
             'moviename': '...',
             'firstframe': first_frame,
             'arena': {'x': 0, 'y': 0, 'r': 0},
             'off': 0,
             'nframes': history_len,
             'endframe': last_frame,
             'timestamps': [0],
             'matname': '...',
             'x_mm': x,
             'y_mm': y,
             'a_mm': a,
             'b_mm': b,
             'pxpermm': 0,
             'fps': 0,
        }

        return a


def count_head_tail(ant):
    ast = ant.state
    b_ = math.sqrt(ast.area / (ast.axis_ratio * math.pi))
    a_ = b_ * ast.axis_ratio

    ant.state.a = a_
    ant.state.b = b_

    x = a_ * math.cos(ast.theta)
    y = a_ * math.sin(ast.theta)

    ant.state.head[0] = ast.position[0] + x
    ant.state.head[1] = ast.position[1] + y

    ant.state.back[0] = ast.position[0] - x
    ant.state.back[1] = ast.position[1] - y


def set_ant_state(ant, mser_id, region, add_history=True):
    if ant.state.area > 0 and add_history:
        ant.history.appendleft(copy.copy(ant.state))

    area_weight = 0.01
    ant.state.mser_id = mser_id
    ant.state.area = region["area"]
    if ant.area_weighted < 0:
        ant.area_weighted = region["area"]
    else:
        ant.area_weighted = ant.area_weighted*(1-area_weight) + region["area"] * area_weight

    ant.state.position = [region["cx"], region["cy"]]
    ant.state.axis_ratio, ant.state.a, ant.state.b = my_utils.mser_main_axis_ratio(region["sxy"], region["sxx"], region["syy"])
    ant.state.theta = my_utils.mser_theta(region["sxy"], region["sxx"], region["syy"])
    ant.state.info = ""

    #so the big jumps will not appears
    if ant.state.lost:
        ant.history[0].position[0] = ant.state.position[0]
        ant.history[0].position[1] = ant.state.position[1]

    ant.state.lost = False
    ant.state.lost_time = 0
    count_head_tail(ant)
    #ant.estimate_orientation(region)


def set_ant_state_undefined(ant, mser_id):
    if ant.state.area > 0:
        ant.history.appendleft(copy.copy(ant.state))

    if mser_id < 0:
        ant.state.info = "NASM"
        #TODO> depends on if lost in colission mode
        vel = ant.velocity(1)
        ant.state.position[0] += vel[0]
        ant.state.position[1] += vel[1]
        ant.state.mser_id = mser_id
        ant.state.lost = True
        ant.state.lost_time += 1
