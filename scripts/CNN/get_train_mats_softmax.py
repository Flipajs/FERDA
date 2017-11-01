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
    labels = []

    ids_set = set(range(NUM_ANIMALS))

    for i in range(NUM_ANIMALS):
        images_f.append([])

        pattern = re.compile(r"(.)*\.jpg")

        for fname in os.listdir(OUT_DIR+'/'+str(i)+''):
            if pattern.match(fname):
                images_f[i].append(fname)

    for k in tqdm.tqdm(range(NUM_EXAMPLES)):
        for i in range(NUM_ANIMALS):
            ai = random.randint(0, len(images_f[i])-1)

            im1 = misc.imread(OUT_DIR+'/'+str(i)+'/'+images_f[i][ai])

            imgs_a.append(im1)
            labels.append(i)

    split_idx = int(TRAIN_TEST_RATIO * len(imgs_a))

    imgs_a_test = np.array(imgs_a[:split_idx])
    imgs_a_train = np.array(imgs_a[split_idx:])
    labels_test = np.array(labels[:split_idx])
    labels_train = np.array(labels[split_idx:])

    with h5py.File(OUT_DIR+'/imgs_multi_train.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_a_train)
    with h5py.File(OUT_DIR+'/imgs_multi_test.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_a_test)
    with h5py.File(OUT_DIR+'/labels_multi_train.h5', 'w') as hf:
        hf.create_dataset("data", data=labels_train)
    with h5py.File(OUT_DIR+'/labels_multi_test.h5', 'w') as hf:
        hf.create_dataset("data", data=labels_test)


