__author__ = 'flipajs'

import pickle
import math
import numpy as np


ferda_path = '../out/eight_with_gt_repair.m4v-1505-ants.pkl'
file = open(ferda_path)
ants = pickle.load(file)
file.close()

frames = 1506
ant_num = len(ants)

def get_ants_stats(ants):
    x = [[0] * (frames) for i in range(len(ants))]
    y = [[0] * (frames) for i in range(len(ants))]
    a = [[0] * (frames) for i in range(len(ants))]
    b = [[0] * (frames) for i in range(len(ants))]
    area = [[0] * (frames) for i in range(len(ants))]
    theta = [[0] * (frames) for i in range(len(ants))]

    for ant_id in range(len(ants)):
        x[ant_id][0] = ants[ant_id].state.position.x
        y[ant_id][0] = ants[ant_id].state.position.y
        a[ant_id][0] = ants[ant_id].state.a
        b[ant_id][0] = ants[ant_id].state.b
        area[ant_id][0] = ants[ant_id].state.area
        theta[ant_id][0] = ants[ant_id].state.theta

        for frame_id in range(1, frames):
            x[ant_id][frame_id] = ants[ant_id].history[frame_id-1].position.x
            y[ant_id][frame_id] = ants[ant_id].history[frame_id-1].position.y
            a[ant_id][frame_id] = ants[ant_id].history[frame_id-1].a
            b[ant_id][frame_id] = ants[ant_id].history[frame_id-1].b
            area[ant_id][frame_id] = ants[ant_id].history[frame_id-1].area
            theta[ant_id][frame_id] = ants[ant_id].history[frame_id-1].theta

    return x[::-1], y[::-1], a[::-1], b[::-1], theta[::-1], area[::-1]

def get_ants_speed(x, y):
    prev_position = [0, 0] * ant_num
    speed = [[0] * (frames-1) for i in range(ant_num)]

    for i in range(ant_num):
        prev_position[i] = [x[i][0], y[i][0]]

    for frame_i in range(frames-1):
        for ant_i in range(ant_num):
            xx = x[ant_i][frame_i+1]
            yy = y[ant_i][frame_i+1]

            spx = prev_position[ant_i][0] - xx
            spy = prev_position[ant_i][1] - yy

            speed[ant_i][frame_i] = math.sqrt(spx*spx + spy*spy)

            prev_position[ant_i] = [xx, yy]

    return speed


x, y, a, b, theta, area = get_ants_stats(ants)
speed = get_ants_speed(x, y)

avg_speed = [0] * ant_num
med_speed = [0] * ant_num
avg_area = [0] * ant_num
med_area = [0] * ant_num
for i in range(ant_num):
    avg_speed[i] = sum(speed[i]) / float(frames-1)
    med_speed[i] = np.median(speed[i])

    avg_area[i] = np.average(area[i])
    med_area[i] = np.median(area[i])


print avg_speed
print med_speed
print avg_area
print med_area