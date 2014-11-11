__author__ = 'filip@naiser.cz'

import pickle
import os

path = '../out/msers/eight/'

def update_e_labels(labels, frame_idx):
    labels[frame_idx['209']][8] = 2
    labels[frame_idx['346']][9] = 2
    labels[frame_idx['449']][20] = 2
    labels[frame_idx['450']][8] = 2
    labels[frame_idx['450']][9] = 2
    labels[frame_idx['451']][7] = 2
    labels[frame_idx['451']][8] = 2
    labels[frame_idx['452']][5] = 2
    labels[frame_idx['452']][7] = 2
    labels[frame_idx['453']][13] = 2
    labels[frame_idx['472']][1] = 2
    labels[frame_idx['473']][0] = 2
    labels[frame_idx['473']][1] = 2
    labels[frame_idx['474']][5] = 2
    labels[frame_idx['474']][6] = 2
    labels[frame_idx['475']][0] = 2
    labels[frame_idx['475']][1] = 2
    labels[frame_idx['577']][5] = 2
    labels[frame_idx['579']][8] = 2
    labels[frame_idx['602']][7] = 2
    labels[frame_idx['603']][0] = 2
    labels[frame_idx['603']][1] = 2
    labels[frame_idx['604']][5] = 2
    labels[frame_idx['604']][6] = 2
    labels[frame_idx['605']][12] = 2
    labels[frame_idx['605']][13] = 2
    labels[frame_idx['606']][8] = 2
    labels[frame_idx['606']][9] = 2
    labels[frame_idx['607']][0] = 2
    labels[frame_idx['607']][1] = 2
    labels[frame_idx['607']][2] = 2
    labels[frame_idx['608']][4] = 2
    labels[frame_idx['608']][5] = 2
    labels[frame_idx['609']][9] = 2
    labels[frame_idx['609']][10] = 2
    labels[frame_idx['635']][19] = 2
    labels[frame_idx['635']][20] = 2
    labels[frame_idx['636']][5] = 2
    labels[frame_idx['636']][6] = 2
    labels[frame_idx['637']][16] = 2
    labels[frame_idx['637']][17] = 2
    labels[frame_idx['638']][6] = 2
    labels[frame_idx['638']][7] = 2
    labels[frame_idx['649']][9] = 2
    labels[frame_idx['649']][10] = 2



frame_idx = {}
regions = []
labels = []

i = 0
for file in os.listdir(path):
    if file.endswith(".pkl"):
        frame_id = file[8:len(file)-4]

        afile = open(path+'eight-values.pkl', 'rb')
        regions = pickle.load(afile)
        afile.close()

        regions.append([regions])
        labels.append([1] * len(regions))
        frame_idx[str(frame_idx)] = i
        i += 1

print frame_idx

labels = update_e_labels(labels, frame_idx)