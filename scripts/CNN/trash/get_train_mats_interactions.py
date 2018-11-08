from __future__ import unicode_literals
import numpy as np
import sys, os, re, random
import h5py
import string
from tqdm import tqdm
from imageio import imread
from os.path import join
import pandas as pd
import fire

OUT_DIR = '/home/matej/prace/ferda/data/interactions'
# OUT_DIR = '/datagrid/personal/smidm1/ferda/iteractions/'

fields = ['ant1_x',  # 0
            'ant1_y',  # 1
            'ant1_major',  # 2
            'ant1_minor',  # 3
            'ant1_angle',  # 4
            'ant2_x',  # 5
            'ant2_y',  # 6
            'ant2_major',  # 7
            'ant2_minor',  # 8
            'ant2_angle',  # 9
          ]


def tohdf5(in_csv_filename, image_dir, out_hdf5, dataset_name='train'):
    data = pd.read_csv(in_csv_filename)
    img_shape = imread(join(image_dir, data.iloc[0]['filename'])).shape
    out_hdf5.create_dataset(dataset_name, (len(data), ) + img_shape, np.uint8)

    for i, (_, row) in enumerate(tqdm(data.iterrows())):
        out_hdf5[dataset_name][i, ...] = imread(join(image_dir, row['filename']))

        # test swap
        # if (ant1_x**2 + ant1_y**2)**0.5 > (ant2_x**2 + ant2_y**2)**0.5:
        #     ant1_x, ant2_x = ant2_x, ant1_x
        #     ant1_y, ant2_y = ant2_y, ant1_y
        #     ant1_angle, ant2_angle = ant2_angle, ant1_angle
        #     ant1_major, ant2_major = ant2_major, ant1_major
        #     ant1_minor, ant2_minor = ant2_minor, ant1_minor


    # imgs = np.array(imgs)
    # imgs = imgs.astype('float32')
    # imgs /= 255

def train_test_to_hdf5(in_csv, image_dir, out_hdf5):
    hdf5_file = h5py.File(out_hdf5, mode='w')
    tohdf5(in_csv, image_dir, hdf5_file, dataset_name='train')
    # tohdf5(in_csv, image_dir, hdf5_file, dataset_prefix='test')


if __name__ == '__main__':
    fire.Fire(train_test_to_hdf5)
    # np.set_printoptions(precision=2)
    # print "train:"
    # print "MIN: ", np.min(results_train, axis=0)
    # print "MAX: ", np.max(results_train, axis=0)
    # print "MEAN: ", np.mean(results_train, axis=0)
    # print "MED: ", np.median(results_train, axis=0)
    # print "STD: ", np.std(results_train, axis=0)
    #
    # print "test"
    # print "MIN: ", np.min(results_test, axis=0)
    # print "MAX: ", np.max(results_test, axis=0)
    # print "MEAN: ", np.mean(results_test, axis=0)
    # print "MED: ", np.median(results_test, axis=0)
    # print "STD: ", np.std(results_test, axis=0)
    #
    # print "test: ", imgs_test.shape, results_test.shape
    # print "train: ", imgs_train.shape, results_train.shape
