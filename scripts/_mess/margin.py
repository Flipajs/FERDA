from __future__ import print_function
__author__ = 'filip@naiser.cz'

import pickle
import math
import score
import my_utils
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

eight = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 78, 91, 98, 111, 115, 116, 117, 183, 184, 185, 186, 187, 188, 189, 223, 224, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250]
nolid = [53, 57, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 85, 109, 126, 136, 137, 140, 141, 142, 143, 145, 146, 147, 148, 149, 150, 151, 153, 161, 179, 180, 181, 182, 185, 186, 187, 188, 189, 190, 191, 192, 194, 195, 196, 197, 198, 199, 200]

def prepare_data(name, frames):
    regions_idx = {}
    regions_labels = []
    regions_vals = []

    counter = 0

    for id in frames:
        file = open('../out/ab/'+name+'/regions_'+str(id)+'.pkl', 'rb')
        regions = pickle.load(file)
        file.close()

        regions_idx[name[0]+str(id)] = counter

        regions_labels.append([0]*len(regions))
        regions_vals.append([])

        i = 0
        best_id = -1
        best_val = -1
        prev = -1
        for r in regions:
            if r['label'] > prev:
                regions_labels[counter][best_id] = 1
                best_id = -1
                best_val = -1
            else:
                if r['margin'] > best_val:
                    best_id = i
                    best_val = r['margin']

            i += 1

        regions_labels[counter][best_id] = 1
        counter += 1

    return regions_idx, regions_vals, regions_labels


e_regions_idx, e_regions_vals, e_regions_labels = prepare_data('eight', eight)
n_regions_idx, n_regions_vals, n_regions_labels = prepare_data('nolid', nolid)
#
#
#afile = open('../out/margin/eight-labels.pkl', 'wb')
#pickle.dump(e_regions_labels, afile)
#afile.close()
#
#afile = open('../out/margin/eight-values.pkl', 'wb')
#pickle.dump(e_regions_vals, afile)
#afile.close()
#
#afile = open('../out/margin/eight-idx.pkl', 'wb')
#pickle.dump(e_regions_idx, afile)
#afile.close()
#
#afile = open('../out/margin/nolid-labels.pkl', 'wb')
#pickle.dump(n_regions_labels, afile)
#afile.close()
#
#afile = open('../out/margin/nolid-values.pkl', 'wb')
#pickle.dump(n_regions_vals, afile)
#afile.close()
#
#afile = open('../out/margin/nolid-idx.pkl', 'wb')
#pickle.dump(n_regions_idx, afile)
#afile.close()


#afile = open('../out/margin/eight-labels.pkl', 'rb')
#e_regions_labels = pickle.load(afile)
#afile.close()
#
#afile = open('../out/margin/eight-values.pkl', 'rb')
#e_regions_vals = pickle.load(afile)
#afile.close()
#
#afile = open('../out/margin/nolid-labels.pkl', 'rb')
#n_regions_labels = pickle.load(afile)
#afile.close()
#
#afile = open('../out/margin/nolid-values.pkl', 'rb')
#n_regions_vals = pickle.load(afile)
#afile.close()


#e_region_ prepare_data('eight', eight)
max_x = 41
max_y = 26
my_hist = np.zeros((max_y, max_x))

frame_i = 0
x_start = 0.2
y_start = 0.2
step = 0.05
for frame in e_regions_vals:
    reg_i = 0
    for reg in frame:
        if e_regions_labels[frame_i][reg_i] == 1:
            x_id = int(math.floor((reg[0] - x_start) / step))
            y_id = int(math.floor((reg[1] - y_start) / step))

            if x_id > 0 and y_id > 0:
                if x_id < max_x and y_id < max_y:
                    my_hist[y_id][x_id] += 1

        reg_i += 1
    frame_i += 1

frame_i = 0
for frame in n_regions_vals:
    reg_i = 0
    for reg in frame:
        if n_regions_labels[frame_i][reg_i] == 1:
            plt.plot(reg[0], reg[1], 'mx')

            x_id = int(math.floor((reg[0] - x_start) / step))
            y_id = int(math.floor((reg[1] - y_start) / step))

            if x_id > 0 and y_id > 0:
                if x_id < max_x and y_id < max_y:
                    my_hist[y_id][x_id] += 1

        reg_i += 1

    frame_i += 1


#my_hist += 1

very_blurred = ndimage.gaussian_filter(my_hist, sigma=1)
print(very_blurred.max())
#
#afile = open('../out/margin/ab_area_hist_blurred.pkl', 'wb')
#e_regions_labels = pickle.dump(very_blurred, afile)
#afile.close()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(very_blurred, interpolation='nearest', cmap=plt.cm.bone_r)
plt.colorbar()
plt.gca().invert_yaxis()
plt.grid()
plt.show()

frame_i = 0
for frame in e_regions_vals:
    reg_i = 0
    for reg in frame:
        if e_regions_labels[frame_i][reg_i] == 0:
            plt.plot(reg[0], reg[1], 'rx')
        else:
            plt.plot(reg[0], reg[1], 'g^')

        reg_i += 1

    frame_i += 1

frame_i = 0
for frame in n_regions_vals:
    reg_i = 0
    for reg in frame:
        if n_regions_labels[frame_i][reg_i] == 0:
            plt.plot(reg[0], reg[1], 'mx')
        else:
            plt.plot(reg[0], reg[1], 'yv')

        reg_i += 1

    frame_i += 1


plt.axis('equal')
plt.xlabel('area / avg area')
plt.ylabel('ratio ab / avg ratio ab')
plt.grid()
plt.show()