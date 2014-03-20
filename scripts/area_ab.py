__author__ = 'flipajs'

import pickle
import math
import score
import my_utils
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

#eight = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 78, 91, 98, 111, 115, 116, 117, 183, 184, 185, 186, 187, 188, 189, 223, 224, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250]
#nolid = [53, 57, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 85, 109, 126, 136, 137, 140, 141, 142, 143, 145, 146, 147, 148, 149, 150, 151, 153, 161, 179, 180, 181, 182, 185, 186, 187, 188, 189, 190, 191, 192, 194, 195, 196, 197, 198, 199, 200]
#
#
#def correction_eight(regions_labels, regions_idx):
#    regions_labels[regions_idx['e5']][15] = 1
#    regions_labels[regions_idx['e6']][12] = 1
#    regions_labels[regions_idx['e7']][27] = 1
#    regions_labels[regions_idx['e8']][16] = 1
#    regions_labels[regions_idx['e9']][19] = 1
#    regions_labels[regions_idx['e10']][27] = 1
#    regions_labels[regions_idx['e11']][27] = 1
#    regions_labels[regions_idx['e12']][26] = 1
#    regions_labels[regions_idx['e13']][27] = 1
#    regions_labels[regions_idx['e17']][0] = 1
#    regions_labels[regions_idx['e78']][0] = 1
#    regions_labels[regions_idx['e91']][0] = 1
#    regions_labels[regions_idx['e98']][0] = 1
#    regions_labels[regions_idx['e111']][0] = 1
#    regions_labels[regions_idx['e115']][13] = 1
#    regions_labels[regions_idx['e116']][4] = 1
#    regions_labels[regions_idx['e117']][5] = 1
#    regions_labels[regions_idx['e183']][12] = 1
#    regions_labels[regions_idx['e184']][12] = 1
#    regions_labels[regions_idx['e185']][16] = 1
#    regions_labels[regions_idx['e186']][18] = 1
#    regions_labels[regions_idx['e187']][19] = 1
#    regions_labels[regions_idx['e188']][18] = 1
#    regions_labels[regions_idx['e189']][22] = 1
#    regions_labels[regions_idx['e223']][24] = 1
#    regions_labels[regions_idx['e224']][18] = 1
#    regions_labels[regions_idx['e235']][17] = 1
#    regions_labels[regions_idx['e236']][13] = 1
#    regions_labels[regions_idx['e237']][16] = 1
#    regions_labels[regions_idx['e238']][12] = 1
#    regions_labels[regions_idx['e239']][10] = 1
#    regions_labels[regions_idx['e240']][9] = 1
#    regions_labels[regions_idx['e241']][16] = 1
#    regions_labels[regions_idx['e242']][23] = 1
#    regions_labels[regions_idx['e243']][22] = 1
#    regions_labels[regions_idx['e244']][23] = 1
#    regions_labels[regions_idx['e245']][24] = 1
#    regions_labels[regions_idx['e246']][24] = 1
#    regions_labels[regions_idx['e247']][25] = 1
#    regions_labels[regions_idx['e248']][26] = 1
#    regions_labels[regions_idx['e249']][16] = 1
#    regions_labels[regions_idx['e250']][20] = 1
#
#    return regions_labels
#
#def correction_nolid(regions_labels, regions_idx):
#    regions_labels[regions_idx['n53']][42] = 1
#    regions_labels[regions_idx['n57']][13] = 1
#    regions_labels[regions_idx['n68']][39] = 1
#    regions_labels[regions_idx['n69']][20] = 1
#    regions_labels[regions_idx['n70']][35] = 1
#    regions_labels[regions_idx['n71']][26] = 1
#
#    regions_labels[regions_idx['n72']][31] = 1
#    regions_labels[regions_idx['n73']][28] = 1
#    regions_labels[regions_idx['n74']][32] = 1
#    regions_labels[regions_idx['n75']][0] = 1
#    regions_labels[regions_idx['n75']][38] = 1
#    regions_labels[regions_idx['n76']][41] = 1
#
#    regions_labels[regions_idx['n77']][44] = 1
#    regions_labels[regions_idx['n78']][11] = 1
#    regions_labels[regions_idx['n78']][45] = 1
#    regions_labels[regions_idx['n85']][38] = 1
#    regions_labels[regions_idx['n109']][1] = 1
#    regions_labels[regions_idx['n126']][51] = 1
#
#    regions_labels[regions_idx['n136']][13] = 1
#    regions_labels[regions_idx['n136']][22] = 1
#    regions_labels[regions_idx['n137']][2] = 1
#    regions_labels[regions_idx['n137']][47] = 1
#    regions_labels[regions_idx['n140']][10] = 1
#    regions_labels[regions_idx['n140']][49] = 1
#
#    regions_labels[regions_idx['n141']][10] = 1
#    regions_labels[regions_idx['n141']][46] = 1
#    regions_labels[regions_idx['n142']][0] = 1
#    regions_labels[regions_idx['n142']][13] = 1
#    regions_labels[regions_idx['n142']][20] = 1
#    regions_labels[regions_idx['n143']][36] = 1
#
#    regions_labels[regions_idx['n146']][59] = 1
#    regions_labels[regions_idx['n146']][60] = 1
#    regions_labels[regions_idx['n147']][56] = 1
#    regions_labels[regions_idx['n147']][57] = 1
#    regions_labels[regions_idx['n148']][60] = 1
#    regions_labels[regions_idx['n148']][61] = 1
#
#    regions_labels[regions_idx['n149']][53] = 1
#    regions_labels[regions_idx['n149']][54] = 1
#    regions_labels[regions_idx['n150']][53] = 1
#    regions_labels[regions_idx['n150']][54] = 1
#    regions_labels[regions_idx['n151']][52] = 1
#    regions_labels[regions_idx['n151']][53] = 1
#
#    regions_labels[regions_idx['n153']][36] = 1
#    regions_labels[regions_idx['n161']][40] = 1
#    regions_labels[regions_idx['n179']][36] = 1
#    regions_labels[regions_idx['n180']][37] = 1
#    regions_labels[regions_idx['n180']][0] = 1
#    regions_labels[regions_idx['n181']][38] = 1
#
#    regions_labels[regions_idx['n182']][36] = 1
#    regions_labels[regions_idx['n185']][0] = 1
#    regions_labels[regions_idx['n186']][5] = 1
#    regions_labels[regions_idx['n187']][2] = 1
#    regions_labels[regions_idx['n188']][4] = 1
#    regions_labels[regions_idx['n189']][6] = 1
#
#    regions_labels[regions_idx['n190']][5] = 1
#    regions_labels[regions_idx['n191']][25] = 1
#    regions_labels[regions_idx['n192']][11] = 1
#    regions_labels[regions_idx['n194']][9] = 1
#    regions_labels[regions_idx['n195']][4] = 1
#    regions_labels[regions_idx['n196']][0] = 1
#
#    regions_labels[regions_idx['n197']][2] = 1
#    regions_labels[regions_idx['n198']][2] = 1
#    regions_labels[regions_idx['n199']][3] = 1
#    regions_labels[regions_idx['n200']][8] = 1
#
#    return regions_labels
#
#def prepare_data(name, frames):
#    regions_idx = {}
#    regions_labels = []
#    regions_vals = []
#
#    counter = 0
#    if name == 'eight':
#        avg_a = 244
#        avg_ab = 4.25897163403
#    elif name == 'nolid':
#        avg_ab = 4.16476763396
#        avg_a = 122
#
#    for id in frames:
#        file = open('../out/ab/'+name+'/regions_'+str(id)+'.pkl', 'rb')
#        regions = pickle.load(file)
#        file.close()
#
#        regions_idx[name[0]+str(id)] = counter
#
#        regions_labels.append([0]*len(regions))
#        regions_vals.append([])
#
#        i = 0
#        for r in regions:
#            ratio, _, _ = my_utils.mser_main_axis_ratio(r['sxy'], r['sxx'], r['syy'])
#
#            val = score.area_prob(r['area'], avg_a)
#            val *= score.axis_ratio_prob(ratio, avg_ab)
#
#            regions_vals[counter].append([r['area'] / float(avg_a), ratio / avg_ab])
#
#            if val >= 0.8:
#                regions_labels[counter][i] = 1
#
#            i += 1
#
#        counter += 1
#
#    return regions_idx, regions_vals, regions_labels
#
#
#e_regions_idx, e_regions_vals, e_regions_labels = prepare_data('eight', eight)
#n_regions_idx, n_regions_vals, n_regions_labels = prepare_data('nolid', nolid)
#
#e_regions_labels = correction_eight(e_regions_labels, e_regions_idx)
#n_regions_labels = correction_nolid(n_regions_labels, n_regions_idx)
#
#afile = open('../out/ab/eight-labels.pkl', 'wb')
#pickle.dump(e_regions_labels, afile)
#afile.close()
#
#afile = open('../out/ab/eight-values.pkl', 'wb')
#pickle.dump(e_regions_vals, afile)
#afile.close()
#
#afile = open('../out/ab/eight-idx.pkl', 'wb')
#pickle.dump(e_regions_idx, afile)
#afile.close()
#
#afile = open('../out/ab/nolid-labels.pkl', 'wb')
#pickle.dump(n_regions_labels, afile)
#afile.close()
#
#afile = open('../out/ab/nolid-values.pkl', 'wb')
#pickle.dump(n_regions_vals, afile)
#afile.close()
#
#afile = open('../out/ab/nolid-idx.pkl', 'wb')
#pickle.dump(n_regions_idx, afile)
#afile.close()


afile = open('../out/ab/eight-labels.pkl', 'rb')
e_regions_labels = pickle.load(afile)
afile.close()

afile = open('../out/ab/eight-values.pkl', 'rb')
e_regions_vals = pickle.load(afile)
afile.close()

afile = open('../out/ab/nolid-labels.pkl', 'rb')
n_regions_labels = pickle.load(afile)
afile.close()

afile = open('../out/ab/nolid-values.pkl', 'rb')
n_regions_vals = pickle.load(afile)
afile.close()

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


from scipy import misc
lena = misc.lena()
#blurred_lena = ndimage.gaussian_filter(lena, sigma=3)
very_blurred = ndimage.gaussian_filter(my_hist, sigma=1)
print very_blurred.max()

afile = open('../out/ab/ab_area_hist_blurred.pkl', 'wb')
e_regions_labels = pickle.dump(very_blurred, afile)
afile.close()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(very_blurred, interpolation='nearest', cmap=plt.cm.bone_r)
plt.colorbar()
plt.gca().invert_yaxis()
plt.grid()
plt.show()

#frame_i = 0
#for frame in e_regions_vals:
#    reg_i = 0
#    for reg in frame:
#        if e_regions_labels[frame_i][reg_i] == 0:
#            plt.plot(reg[0], reg[1], 'rx')
#        else:
#            plt.plot(reg[0], reg[1], 'g^')
#
#        reg_i += 1
#
#    frame_i += 1
#
#frame_i = 0
#for frame in n_regions_vals:
#    reg_i = 0
#    for reg in frame:
#        if n_regions_labels[frame_i][reg_i] == 0:
#            plt.plot(reg[0], reg[1], 'mx')
#        else:
#            plt.plot(reg[0], reg[1], 'yv')
#
#        reg_i += 1
#
#    frame_i += 1
#
#
#plt.axis('equal')
#plt.xlabel('area / avg area')
#plt.ylabel('ratio ab / avg ratio ab')
#plt.grid()
#plt.show()