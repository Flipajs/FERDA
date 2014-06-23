__author__ = 'flipajs'

from clearmetrics import clearmetrics
from trajectories_data import eight_ctrax, eight_gt, eight_idtracker, eight_ktrack
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import *
import pickle
import os

f = open('trajectories_data/eight_ferda_stable_split_dmaps2_xy.arr', 'rb')
measurements = pickle.load(f)
f.close()

#clear = clearmetrics.ClearMetrics(eight_gt.data, measurements, 7)
#clear.match_sequence()
#evaluation = [clear.get_mota(), clear.get_motp(), clear.get_fn_count(), clear.get_fp_count(), clear.get_mismatches_count(), clear.get_object_count(), clear.get_matches_count()]
#
#f = open('trajectories_data/dump', 'wb')
#package = [evaluation, clear.gt_distances, clear.gt_matches, clear.measurements_matches]
#pickle.dump(package, f)
#f.close()

f = open('trajectories_data/dump', 'rb')
package = pickle.load(f)
evaluation = package[0]
gt_distances = package[1]
gt_matches = package[2]
measurements_matches = package[3]
f.close()

#: *[0-9]
#: *\[[0-9]\]

gt = eight_gt.data
sequence_path = os.path.expanduser('~/dump/eight/frames')

methods = {
    'Ferda': {'data': measurements, 'mc': [673, 691], 'fp': [450, 803, 1284, 806, 807, 1180, 689, 692, 1174, 475, 1179, 1437, 767]}
}

frame = 100
ext = '.png'

method = methods['Ferda']

colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']


start = 1138
step = 1
seq_len = 12
cols = 4
scale_fact = 2

#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
f, axarr = plt.subplots(int(ceil(seq_len/cols)), cols, figsize=(4*scale_fact, ceil(seq_len/cols)*scale_fact), dpi=80, facecolor='w', edgecolor='k')
#y = 330
#h = 140
#x = 745
#w = 140

y = 280
h = 140
x = 700
w = 140

colors_num = len(colors)
m_size = 220
op = 0.9

for frame in range(start, start+seq_len*step):
    i = frame-start
    img = mpimg.imread(sequence_path+'/'+str(frame-1)+'.png')

    axarr[i/cols, i%cols].imshow(img)

    data = array(method['data'][frame])
    c_data = array(eight_ctrax.data[frame])
    #k_data = array(eight_ktrack.data[frame])
    i_data = array(eight_idtracker.data[frame])
    gt_data = array(gt[frame])
    gtm = array(gt_matches[frame])
    mm = array(measurements_matches[frame])

    for j in range(gt_data.shape[0]):
        axarr[i/cols, i%cols].scatter(gt_data[j, 0], gt_data[j, 1], color='r', marker='+', alpha=op, s=m_size)

    #for j in range(c_data.shape[0]):
    #    if c_data[j] is not None:
    #        axarr[i/cols, i%cols].scatter(c_data[j][0], c_data[j][1], color='c', marker='o', alpha=op, s=m_size, facecolor='none')
    #for j in range(k_data.shape[0]):
    #    if k_data[j] is not None:
    #        axarr[i/cols, i%cols].scatter(k_data[j][0], k_data[j][1], color='m', marker='o', alpha=op, s=m_size, facecolor='none')
    #for j in range(i_data.shape[0]):
    #    if i_data[j] is not None:
    #        axarr[i/cols, i%cols].scatter(i_data[j][0], i_data[j][1], color='y', marker='o', alpha=op, s=m_size, facecolor='none')

    for j in range(data.shape[0]):
        axarr[i/cols, i%cols].scatter(data[j, 0], data[j, 1], color=colors[j], marker='o', alpha=op, s=m_size, facecolor='none')


        #a = gt_data[j, :]
        #b = data[gtm[j], :]
        #if gtm[j] > -1:
        #    axarr[i/cols, i%cols].plot([a[0], b[0]], [a[1], b[1]], c='r')


    axarr[i/cols, i%cols].set_ylim((y, y+h))
    axarr[i/cols, i%cols].axes.set_xlim((x, x+w))
    axarr[i/cols, i%cols].invert_yaxis()
    axarr[i/cols, i%cols].get_xaxis().set_visible(False)
    axarr[i/cols, i%cols].get_yaxis().set_visible(False)

#plt.subplots_adjust(hspace=.01, wspace=.01, left=None, bottom=None, right=None, top=None)
plt.subplots_adjust(hspace=0, wspace=0, left=0, bottom=0, right=1, top=1)

plt.show()

#f, axarr = plt.subplots(1, 4)
#subplt_i = 0
#for frame in ran:
#    print gt_distances[frame]
#    print gt_matches[frame]
#    print measurements_matches[frame]
#    img=mpimg.imread(sequence_path+'/'+str(frame)+'.png')
#    axarr[0, subplt_i] = plt.imshow(img)
#
#    data = array(method['data'][frame])
#    gt_data = array(gt[frame])
#    gtm = array(gt_matches[frame])
#    mm = array(measurements_matches[frame])
#    for i in range(8):
#        axarr[0, subplt_i] = plt.scatter(data[i, 0], data[i, 1], color=colors[i], marker='o', alpha=op, s=100)
#        axarr[0, subplt_i] = plt.scatter(gt_data[i, 0], gt_data[i, 1], color=colors[i], marker='v', alpha=op, s=100)
#
#        a = gt_data[i, :]
#        b = data[gtm[i], :]
#        if gtm[i] > -1:
#            axarr[0, subplt_i] = plt.plot([a[0], b[0]], [a[1], b[1]], c='r')
#
#        #a = data[i, :]
#        #b = gt_data[mm[i], :]
#        #plt.plot([a[0], b[0]], [a[1], b[1]], c='b')
#
#    y = 340
#    h = 140
#    x = 750
#    w = 140
#
#    #axarr[i].ylim((y, y+h))
#    #axarr[i].xlim((x, x+w))
#    #plt.gca().invert_yaxis()
#    i += 1
#
#
#plt.show()
#
##for frame in range(689, 694):
##    plt.figure()
##    img=mpimg.imread(sequence_path+'/'+str(frame)+'.png')
##    imgplot = plt.imshow(img)
##    gt_data = array(gt[frame])
##    for i in range(8):
##        plt.scatter(gt_data[i, 0], gt_data[i, 1], color=colors[i], marker='o', alpha=0.75, s=100)
#
#
#