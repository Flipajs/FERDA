from __future__ import print_function
from utils import video_manager

__author__ = 'filip@naiser.cz'

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import math


def x(trx, id):
    return trx[0][id].x[0]

def y(trx, id):
    return trx[0][id].y[0]

def y_flip(trx, id, im_height):
    return im_height - trx[0][id].y[0]

def th(trx, id):
    return trx[0][id].theta[0]

def th_deg(trx, id, i):
    return ((trx[0][id].theta[0][i]) * 57.2957795)

def firstframe(trx, id):
    return trx[0][id].firstframe[0][0]

def nframes(trx, id):
    return trx[0][id].nframes[0][0]

def a(trx, id):
    return trx[0][id].a[0]

def b(trx, id):
    return trx[0][id].b[0]

def get_ants_speed(trx):
    frames = len(x(trx, 0))
    ants = len(trx[0])

    prev_position = [0, 0] * ants
    speed = [[0] * (frames-1) for i in range(ants)]

    for i in range(ants):
        xx = x(trx, i)[0]
        yy = y(trx, i)[0]
        prev_position[i] = [xx, yy]

    for frame_i in range(frames-1):
        for ant_i in range(ants):
            xx = x(trx, ant_i)[frame_i+1]
            yy = y(trx, ant_i)[frame_i+1]

            spx = prev_position[ant_i][0] - xx
            spy = prev_position[ant_i][1] - yy

            speed[ant_i][frame_i] = math.sqrt(spx*spx + spy*spy)

            prev_position[ant_i] = [xx, yy]

    return speed


ferda_path = '../out/gt_ctrax_ferda/ferda_trx.mat'
gt_path = '../out/gt_ctrax_ferda/fixed_uncompressedmovie.mat'
ctrax_path = '../out/gt_ctrax_ferda/uncompressedmovie.mat'
video_file_name = '/home/flipajs/Dropbox/PycharmProjects/data/eight/eight.m4v'

gt_mapping = [5, 1, 0, 3, 6, 2, 7, 4]

ferda = sio.loadmat(ferda_path, struct_as_record=False)
f_trx = ferda['trx']

gt = sio.loadmat(gt_path, struct_as_record=False)
g_trx = gt['trx']

ctrax = sio.loadmat(ctrax_path, struct_as_record=False)
c_trx = ctrax['trx']

for i in range(8):
    print(i, x(f_trx, i)[0], x(g_trx, gt_mapping[i])[0], x(c_trx, gt_mapping[i])[0])

print(" ")

a_num = len(f_trx[0])
frames = len(x(f_trx, 0))

vid = video_manager.VideoManager(video_file_name)

print(a(g_trx, 0)[0])


speed = get_ants_speed(g_trx)

avg_speed = [0] * a_num
med_speed = [0] * a_num
avg_speed100 = [0] * a_num
avg_speed200 = [0] * a_num
for i in range(a_num):
    avg_speed[i] = sum(speed[i]) / float(frames-1)
    avg_speed[i] = np.median(speed[i])
    avg_speed100[i] = sum(speed[i][0:100]) / float(100)
    avg_speed200[i] = sum(speed[i][100:200]) / float(100)

print(avg_speed)
print(med_speed)
print(avg_speed100)
print(avg_speed200)

plt.plot(speed[6])
plt.plot(speed[7])

plt.show()

#for i in range(frames):
#    img = vid.next_img()
#    im_height = img.shape[0]
#
#    if i < 500:
#        continue
#
#    print i
#
#    for a_id in range(a_num):
#        gc = (int(x(g_trx, gt_mapping[a_id])[i]), int(y_flip(g_trx, gt_mapping[a_id], im_height)[i]))
#        gab = (int(2 * a(g_trx, gt_mapping[a_id])[i]), int(2 * b(g_trx, gt_mapping[a_id])[i]))
#        cv2.ellipse(img, gc, gab, -th_deg(g_trx, gt_mapping[a_id], i), 0, 360, (0, 0, 255), 4)
#
#        fc = (int(x(f_trx, a_id)[i]), int(y(f_trx, a_id)[i]))
#        fab = (int(2 * a(f_trx, a_id)[i]), int(2 * b(f_trx, a_id)[i]))
#        cv2.ellipse(img, fc, fab, th_deg(f_trx, a_id, i), 0, 360, (255, 120, 0), 3)
#
#    for a_id in range(len(c_trx[0])):
#        ff = firstframe(c_trx, a_id)
#        nf = nframes(c_trx, a_id)
#        if i >= ff-1 and i < ff+nf-1:
#            cc = (int(x(c_trx, a_id)[i-ff+1]), int(y_flip(c_trx, a_id, im_height)[i-ff+1]))
#            cab = (int(2 * a(c_trx, a_id)[i-ff+1]), int(2 * b(c_trx, a_id)[i-ff+1]))
#            cv2.ellipse(img, cc, cab, -th_deg(c_trx, a_id, i-ff+1), 0, 360, (0, 125, 125), 2)
#
#
#    cv2.imshow("test", img)
#    cv2.waitKey(0)