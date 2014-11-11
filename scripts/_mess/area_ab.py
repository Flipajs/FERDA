__author__ = 'filip@naiser.cz'

import sys
sys.path.append('../libs')
import pickle
import math
import score
import my_utils
import matplotlib.pyplot as plt
import numpy as np
import mser_operations
import visualize
from scipy import ndimage
import cv2
from numpy import *

eight = range(1, 101)
noplast = range(1, 101)

def draw_region_group_collection(img, regions, groups, labels, cell_size=70):
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
    r_id = 0

    for row in range(len(groups)):
        if row >= rows:
                row_p = -rows
                col_p = cols+1

        cv2.putText(collection, str(row), (3 + cell_size*col_p, 30 + cell_size*(row+row_p)), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.CV_AA)

        margins = [0]*len(groups[row])
        for col in range(len(groups[row])):
            r = regions[groups[row][col]]
            ratio, _, _ = my_utils.mser_main_axis_ratio(r['sxy'], r['sxx'], r['syy'])

            margins[col] = r['margin']

        for col in range(len(groups[row])):
            img_[border:-border, border:-border] = img.copy()
            r = regions[groups[row][col]]
            if r["cx"] == inf or r["cy"] == inf:
                continue

            c = (0, 255, 0)

            cont = True
            if labels[r_id] == 1:
                c = (0, 0, 212)
                cont = False

            if labels[r_id] > 1:
                c = (255, 255, 0)
                cont = False

            visualize.draw_region(img_[border:-border, border:-border], r, c, contour=cont)

            img_small = img_[border + r[
                "cy"] - cell_size / 2:border + r["cy"] + cell_size / 2, border + r["cx"] - cell_size / 2:border + r[
                "cx"] + cell_size / 2].copy()

            cv2.putText(img_small, str(groups[row][col]), (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.CV_AA)
            cv2.putText(img_small, str(r['area']), (3, 45), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.CV_AA)
            cv2.putText(img_small, str(r['margin']), (3, 55), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.CV_AA)
            collection[(row + row_p) * cell_size:((row + row_p) + 1) * cell_size, num_strip + (col + col_p) * cell_size:num_strip + ((col + col_p) + 1) * cell_size, :] = img_small

            r_id += 1

    return collection

def correction_eight(regions_labels, regions_idx):
    #1:100 .. checked


    #regions_labels[regions_idx['e2']][] = 1

    return regions_labels

def correction_noplast(regions_labels, regions_idx):
    regions_labels[regions_idx['n8']][10] = 0
    regions_labels[regions_idx['n8']][11] = 1

    regions_labels[regions_idx['n9']][15] = 0
    regions_labels[regions_idx['n9']][16] = 1

    regions_labels[regions_idx['n28']][2] = 2
    regions_labels[regions_idx['n28']][78] = 0

    regions_labels[regions_idx['n29']][2] = 2
    regions_labels[regions_idx['n29']][75] = 0

    regions_labels[regions_idx['n30']][2] = 2
    regions_labels[regions_idx['n30']][74] = 0

    regions_labels[regions_idx['n43']][0] = 0 #rozteklej

    regions_labels[regions_idx['n44']][1] = 0 #rozteklej

    regions_labels[regions_idx['n45']][8] = 0 #rozteklej

    regions_labels[regions_idx['n46']][8] = 0 #rozteklej

    regions_labels[regions_idx['n47']][8] = 0 #rozteklej

    regions_labels[regions_idx['n48']][12] = 0 #rozteklej

    regions_labels[regions_idx['n49']][9] = 0 #rozteklej

    regions_labels[regions_idx['n50']][8] = 0 #rozteklej

    regions_labels[regions_idx['n51']][19] = 0 #rozteklej

    regions_labels[regions_idx['n52']][9] = 0 #rozteklej

    regions_labels[regions_idx['n53']][75] = 0

    regions_labels[regions_idx['n54']][39] = 0 #rozteklej

    regions_labels[regions_idx['n55']][24] = 0 #rozteklej

    regions_labels[regions_idx['n56']][17] = 0 #rozteklej

    regions_labels[regions_idx['n57']][18] = 0 #rozteklej

    regions_labels[regions_idx['n58']][14] = 0 #rozteklej

    regions_labels[regions_idx['n59']][18] = 0 #rozteklej

    regions_labels[regions_idx['n60']][20] = 0 #rozteklej
    regions_labels[regions_idx['n60']][22] = 2
    regions_labels[regions_idx['n60']][61] = 0

    regions_labels[regions_idx['n61']][36] = 0 #rozteklej

    regions_labels[regions_idx['n62']][17] = 0 #rozteklej

    regions_labels[regions_idx['n63']][8] = 0 #rozteklej
    regions_labels[regions_idx['n63']][21] = 2

    regions_labels[regions_idx['n64']][16] = 2
    regions_labels[regions_idx['n64']][19] = 0 #rozteklej

    regions_labels[regions_idx['n65']][36] = 0 #rozteklej

    regions_labels[regions_idx['n66']][21] = 0 #rozteklej

    regions_labels[regions_idx['n67']][22] = 0 #rozteklej

    regions_labels[regions_idx['n68']][71] = 0

    regions_labels[regions_idx['n71']][17] = 0 #rozteklej

    regions_labels[regions_idx['n72']][12] = 0 #rozteklej

    regions_labels[regions_idx['n73']][1] = 2
    regions_labels[regions_idx['n73']][36] = 0 #rozteklej

    regions_labels[regions_idx['n74']][2] = 2
    regions_labels[regions_idx['n74']][23] = 0 #rozteklej
    regions_labels[regions_idx['n74']][79] = 0

    regions_labels[regions_idx['n75']][2] = 2
    regions_labels[regions_idx['n75']][43] = 0 #rozteklej

    regions_labels[regions_idx['n76']][36] = 0 #rozteklej

    regions_labels[regions_idx['n77']][6] = 0 #rozteklej

    regions_labels[regions_idx['n78']][18] = 0 #rozteklej

    regions_labels[regions_idx['n79']][5] = 0 #rozteklej

    regions_labels[regions_idx['n80']][8] = 0 #rozteklej

    regions_labels[regions_idx['n81']][21] = 0 #rozteklej

    regions_labels[regions_idx['n82']][14] = 0 #rozteklej

    regions_labels[regions_idx['n83']][24] = 0 #rozteklej

    regions_labels[regions_idx['n84']][21] = 0 #rozteklej

    regions_labels[regions_idx['n85']][25] = 0 #rozteklej

    regions_labels[regions_idx['n86']][27] = 0 #rozteklej

    regions_labels[regions_idx['n100']][10] = 2

    #1..100 checked

    return regions_labels

def prepare_data(name, frames):
    regions_idx = {}
    regions_labels = []
    regions_vals = []

    counter = 0
    if name == 'eight':
        avg_a = 244
        avg_ab = 4.25897163403
        ant_num = 8
    elif name == 'noplast':
        avg_ab = 4.16476763396
        avg_a = 117
        ant_num = 15

    for id in frames:
        file = open('../out/'+name+'_dump/regions/regions_'+str(id)+'.pkl', 'rb')
        regions = pickle.load(file)
        file.close()

        groups = mser_operations.get_region_groups(regions)
        groups_max_margin_id = [0] * len(groups)
        groups_max_margin = [0] * len(groups)
        group_vals = [0] * len(groups)

        for i in range(len(groups)):
            g = groups[i]
            groups_max_margin[i], groups_max_margin_id[i] = my_utils.best_margin(regions, g)

        regions_idx[name[0]+str(id)] = counter
        regions_labels.append([0]*len(regions))
        regions_vals.append([])

        sorted_ids = np.argsort(np.array(groups_max_margin))[::-1]

        c = 0
        for id in sorted_ids:
            regions_labels[counter][groups_max_margin_id[id]] = 1
            c += 1
            if c == ant_num:
                break

        i = 0
        for r in regions:
            ratio, a, b = my_utils.mser_main_axis_ratio(r['sxy'], r['sxx'], r['syy'])
            a, b = my_utils.count_head_tail(r['area'], a, b)
            regions_vals[counter].append([r['area'], a, b])

        counter += 1

    return regions_idx, regions_vals, regions_labels


def log_hist(hist):
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            hist[i][j] = math.log(hist[i][j]+1)

    return hist

#
#e_regions_idx, e_regions_vals, e_regions_labels = prepare_data('eight', eight)
#n_regions_idx, n_regions_vals, n_regions_labels = prepare_data('noplast', noplast)
#
#preview = True
#
#e_regions_labels = correction_eight(e_regions_labels, e_regions_idx)
#n_regions_labels = correction_noplast(n_regions_labels, n_regions_idx)
#
#if preview:
#    name = 'eight'
#    counter = 0
#    start_from = 101
#    for id in eight:
#        print "ID: ", id
#        if counter < start_from-1:
#            counter += 1
#            continue
#
#        img = cv2.imread('../out/'+name+'_dump/frames/frame'+str(id)+'.png')
#
#        file = open('../out/'+name+'_dump/regions/regions_'+str(id)+'.pkl', 'rb')
#        regions = pickle.load(file)
#        file.close()
#
#        groups = mser_operations.get_region_groups(regions)
#
#        collection = draw_region_group_collection(img, regions, groups, e_regions_labels[counter])
#
#        cv2.imshow("collection", collection)
#        cv2.waitKey(0)
#
#        counter += 1
#
#    name = 'noplast'
#    counter = 0
#    start_from = 101
#    for id in eight:
#        print "ID: ", id
#        if counter < start_from-1:
#            counter += 1
#            continue
#
#        img = cv2.imread('../out/'+name+'_dump/frames/frame'+str(id)+'.png')
#
#        file = open('../out/'+name+'_dump/regions/regions_'+str(id)+'.pkl', 'rb')
#        regions = pickle.load(file)
#        file.close()
#
#        groups = mser_operations.get_region_groups(regions)
#
#        collection = draw_region_group_collection(img, regions, groups, n_regions_labels[counter])
#
#        cv2.imshow("collection", collection)
#
#        w = True
#        while w:
#            key = cv2.waitKey(1)
#            if key == 110 or key == 32:
#                w = False
#
#        counter += 1
#
#afile = open('../out/ab_margin/eight-labels.pkl', 'wb')
#pickle.dump(e_regions_labels, afile)
#afile.close()
#
#afile = open('../out/ab_margin/eight-values.pkl', 'wb')
#pickle.dump(e_regions_vals, afile)
#afile.close()
#
#afile = open('../out/ab_margin/eight-idx.pkl', 'wb')
#pickle.dump(e_regions_idx, afile)
#afile.close()
#
#afile = open('../out/ab_margin/noplaster-labels.pkl', 'wb')
#pickle.dump(n_regions_labels, afile)
#afile.close()
#
#afile = open('../out/ab_margin/noplaster-values.pkl', 'wb')
#pickle.dump(n_regions_vals, afile)
#afile.close()
#
#afile = open('../out/ab_margin/noplaster-idx.pkl', 'wb')
#pickle.dump(n_regions_idx, afile)
#afile.close()



afile = open('../out/ab_margin/eight-labels.pkl', 'rb')
e_regions_labels = pickle.load(afile)
afile.close()

afile = open('../out/ab_margin/eight-values.pkl', 'rb')
e_regions_vals = pickle.load(afile)
afile.close()

afile = open('../out/ab_margin/noplaster-labels.pkl', 'rb')
n_regions_labels = pickle.load(afile)
afile.close()

afile = open('../out/ab_margin/noplaster-values.pkl', 'rb')
n_regions_vals = pickle.load(afile)
afile.close()

max_x = 41
max_y = 26
my_hist = np.zeros((max_y, max_x))

frame_i = 0
x_start = 0.2
y_start = 0.2
step = 0.05

plt.figure()

areas = []
aa = []
for frame in e_regions_vals:
    reg_i = 0
    for reg in frame:
        if e_regions_labels[frame_i][reg_i] == 1:
            area = reg[0] / float(244)
            a = reg[1] / float(17.97)

            areas.append(area)
            aa.append(a)
            x_id = int(math.floor((area - x_start) / step))
            y_id = int(math.floor((a - y_start) / step))

            if x_id > 0 and y_id > 0:
                if x_id < max_x and y_id < max_y:
                    my_hist[y_id][x_id] += 1

        reg_i += 1
    frame_i += 1


plt.plot(areas, aa, 'cx', label='ist-eight')

frame_i = 0
areas = []
aa = []
for frame in n_regions_vals:
    reg_i = 0
    for reg in frame:
        if n_regions_labels[frame_i][reg_i] == 1:
            area = reg[0] / float(117)
            a = reg[1] / float(12.7)

            areas.append(area)
            aa.append(a)
            x_id = int(math.floor((area - x_start) / step))
            y_id = int(math.floor((a - y_start) / step))

            if x_id > 0 and y_id > 0:
                if x_id < max_x and y_id < max_y:
                    my_hist[y_id][x_id] += 1

        reg_i += 1

    frame_i += 1

plt.hold(True)
plt.plot(areas, aa, 'mx', label='ist-fifteen')

plt.legend(loc='bottom center', shadow=True, fontsize=13)

#plt.legend((eight_l, noplast_l), ('ist-eight', 'ist-fifteen'),
#           loc='top right',
#           ncol=2,
#           fontsize=13)

plt.ylabel('main axis / avg main axis')
plt.xlabel('area / avg area')
plt.grid()

log_hist(my_hist)

very_blurred = ndimage.gaussian_filter(my_hist, sigma=2)
very_blurred = very_blurred / very_blurred.max()
print "MAX: ", very_blurred.max()

afile = open('../out/ab/ab_area_hist_blurred.pkl', 'wb')
e_regions_labels = pickle.dump(very_blurred, afile)
afile.close()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(very_blurred, interpolation='nearest', cmap=plt.cm.bone)
plt.colorbar()
plt.gca().invert_yaxis()
ylabels=[0, 0.2, 0.5, 0.75, 1.0, 1.25]
ax.set_yticklabels(ylabels)
xlabels=[0, 0.2, 0.7, 1.2, 1.7, 2.2]
ax.set_xticklabels(xlabels)
plt.grid(c='white')
plt.ylabel('main axis / avg main axis')
plt.xlabel('area / avg area')
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