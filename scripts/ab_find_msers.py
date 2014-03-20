__author__ = 'flipajs'

import pickle
import math

afile = open('../out/ab/eight-labels.pkl', 'rb')
e_regions_labels = pickle.load(afile)
afile.close()

afile = open('../out/ab/eight-values.pkl', 'rb')
e_regions_vals = pickle.load(afile)
afile.close()

afile = open('../out/ab/eight-idx.pkl', 'rb')
e_regions_idx = pickle.load(afile)
afile.close()

afile = open('../out/ab/nolid-labels.pkl', 'rb')
n_regions_labels = pickle.load(afile)
afile.close()

afile = open('../out/ab/nolid-values.pkl', 'rb')
n_regions_vals = pickle.load(afile)
afile.close()

afile = open('../out/ab/nolid-idx.pkl', 'rb')
n_regions_idx = pickle.load(afile)
afile.close()


ab = 0.5
a = 1

e_best = float('inf')
e_best_frame = -1
e_best_i = -1

for frame in e_regions_vals:
    for i in frame:
        if math.sqrt((i[0]-a) * (i[0]-a) + (i[1] - ab) * (i[1] - ab))