from __future__ import print_function
from __future__ import unicode_literals
from builtins import str
from builtins import range
import numpy as np
import sys, os, re, random
import h5py
import string
from scipy import misc
import tqdm

OUT_DIR = '/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1'
NUM_ANIMALS = 6

# NUM_EXAMPLES x NUM_A
NUM_EXAMPLES = 10
NEGATIVE_EXA_RATIO = 1
TRAIN_TEST_RATIO = 0.1

if __name__ == '__main__':
    if len(sys.argv) > 1:
        OUT_DIR = sys.argv[1]
        NUM_ANIMALS = string.atoi(sys.argv[2])
        NUM_EXAMPLES = string.atoi(sys.argv[3])
        NEGATIVE_EXA_RATIO = string.atoi(sys.argv[2])

    images_f = []

    imgs_a = []
    imgs_b = []
    labels = []

    ids_set = set(range(NUM_ANIMALS))

    for i in range(NUM_ANIMALS):
        images_f.append([])

        pattern = re.compile(r"(.)*\.jpg")

        for fname in os.listdir(OUT_DIR+'/'+str(i)+''):
            if pattern.match(fname):
                images_f[i].append(fname)

    for k in tqdm.tqdm(list(range(NUM_EXAMPLES))):
        for i in range(NUM_ANIMALS):
            ai, aj = random.sample(range(0, len(images_f[i])), 2)

            im1 = misc.imread(OUT_DIR+'/'+str(i)+'/'+images_f[i][ai])
            im2 = misc.imread(OUT_DIR+'/'+str(i)+'/'+images_f[i][aj])

            imgs_a.append(im1)
            imgs_b.append(im2)
            labels.append(1)

            # for j in range(NEGATIVE_EXA_RATIO):
            #     neg_i = random.choice(list(ids_set - set([i])))
            #     neg_f = random.choice(images_f[neg_i])
            #
            #     im_negative = misc.imread(OUT_DIR+'/'+str(neg_i)+'/'+neg_f)
            #
            #     imgs_a.append(im1)
            #     imgs_b.append(im_negative)
            #     labels.append(0)

            for neg_i in list(ids_set - set([i])):
                # neg_i = random.choice(list(ids_set - set([i])))
                neg_f = random.choice(images_f[neg_i])

                im_negative = misc.imread(OUT_DIR+'/'+str(neg_i)+'/'+neg_f)

                imgs_a.append(im1)
                imgs_b.append(im_negative)
                labels.append(0)

    split_idx = int(TRAIN_TEST_RATIO * len(imgs_a))

    imgs_a_test = np.array(imgs_a[:split_idx])
    imgs_a_train = np.array(imgs_a[split_idx:])
    imgs_b_test = np.array(imgs_b[:split_idx])
    imgs_b_train = np.array(imgs_b[split_idx:])
    labels_test = np.array(labels[:split_idx])
    labels_train = np.array(labels[split_idx:])

    with h5py.File(OUT_DIR+'/imgs_a_train.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_a_train)
    with h5py.File(OUT_DIR+'/imgs_a_test.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_a_test)
    with h5py.File(OUT_DIR+'/imgs_b_train.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_b_train)
    with h5py.File(OUT_DIR+'/imgs_b_test.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_b_test)
    with h5py.File(OUT_DIR+'/labels_train.h5', 'w') as hf:
        hf.create_dataset("data", data=labels_train)
    with h5py.File(OUT_DIR+'/labels_test.h5', 'w') as hf:
        hf.create_dataset("data", data=labels_test)


    with h5py.File(OUT_DIR+'/imgs_a_train.h5', 'r') as hf:
        data = hf['data'][:]
        print(data.shape)

    with h5py.File(OUT_DIR+'/imgs_a_test.h5', 'r') as hf:
        data = hf['data'][:]
        print(data.shape)

    with h5py.File(OUT_DIR+'/imgs_b_train.h5', 'r') as hf:
        data = hf['data'][:]
        print(data.shape)

    with h5py.File(OUT_DIR+'/imgs_b_test.h5', 'r') as hf:
        data = hf['data'][:]
        print(data.shape)

    with h5py.File(OUT_DIR+'/labels_train.h5', 'r') as hf:
        data = hf['data'][:]
        print(data.shape)

    with h5py.File(OUT_DIR+'/labels_test.h5', 'r') as hf:
        data = hf['data'][:]
        print(data.shape)


