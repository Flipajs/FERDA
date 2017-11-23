import numpy as np
import sys, os, re, random
import h5py
import string
from tqdm import tqdm
from imageio import imread
import csv

OUT_DIR = '/Users/flipajs/Downloads/double_regions'
NUM_PARAMS = 2


def get_data(type):
    imgs = []
    results = []
    print type
    with open(OUT_DIR + '/'+ type +'.csv') as f:
        data = csv.reader(f)

        for i, row in tqdm(enumerate(data)):
            if i == 0:
                continue

            im1 = imread(OUT_DIR + '/images_'+ type +'/' + row[0])
            imgs.append(im1)
            ant1_x = string.atof(row[1])
            ant1_y = string.atof(row[2])
            ant1_major = string.atof(row[3])
            ant1_minor = string.atof(row[4])
            ant1_angle = string.atof(row[5]) % 360

            ant2_x = string.atof(row[6])
            ant2_y = string.atof(row[7])
            ant2_major = string.atof(row[8])
            ant2_minor = string.atof(row[9])
            ant2_angle = string.atof(row[10]) % 360

            # test swap

            if (ant1_x**2 + ant1_y**2)**0.5 > (ant2_x**2 + ant2_y**2)**0.5:
                ant1_x, ant2_x = ant2_x, ant1_x
                ant1_y, ant2_y = ant2_y, ant1_y
                ant1_angle, ant2_angle = ant2_angle, ant1_angle
                ant1_major, ant2_major = ant2_major, ant1_major
                ant1_minor, ant2_minor = ant2_minor, ant1_minor

            results.append([ant1_x,
                           ant1_y,
                           ant1_major,
                           ant1_minor,
                           ant1_angle,
                           ant2_x,
                           ant2_y,
                           ant2_major,
                           ant2_minor,
                           ant2_angle
                            ])

    imgs = np.array(imgs)
    imgs = imgs.astype('float32')
    imgs /= 255

    results = np.array(results)
    return imgs, results


if __name__ == '__main__':
    if len(sys.argv) > 1:
        OUT_DIR = sys.argv[1]

    imgs_test, results_test = get_data('test')
    imgs_train, results_train = get_data('train')

    np.set_printoptions(precision=2)

    print "train:"
    print "MIN: ", np.min(results_train, axis=0)
    print "MAX: ", np.max(results_train, axis=0)
    print "MEAN: ", np.mean(results_train, axis=0)
    print "MED: ", np.median(results_train, axis=0)
    print "STD: ", np.std(results_train, axis=0)

    print "test"
    print "MIN: ", np.min(results_test, axis=0)
    print "MAX: ", np.max(results_test, axis=0)
    print "MEAN: ", np.mean(results_test, axis=0)
    print "MED: ", np.median(results_test, axis=0)
    print "STD: ", np.std(results_test, axis=0)

    print "test: ", imgs_test.shape, results_test.shape
    print "train: ", imgs_train.shape, results_train.shape

    with h5py.File(OUT_DIR+'/imgs_inter_train.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_train)
    with h5py.File(OUT_DIR+'/imgs_inter_test.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_test)
    with h5py.File(OUT_DIR+'/results_inter_train.h5', 'w') as hf:
        hf.create_dataset("data", data=results_train)
    with h5py.File(OUT_DIR+'/results_inter_test.h5', 'w') as hf:
        hf.create_dataset("data", data=results_test)