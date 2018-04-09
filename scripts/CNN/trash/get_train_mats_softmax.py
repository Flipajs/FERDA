import numpy as np
import sys, os, re, random
import h5py
import string
import tqdm
from imageio import imread

OUT_DIR = '/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1'
NUM_ANIMALS = 6

# NUM_EXAMPLES x NUM_A
NUM_EXAMPLES = 10
TRAIN_TEST_RATIO = 0.1
RANDOM = False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        OUT_DIR = sys.argv[1]
        NUM_ANIMALS = string.atoi(sys.argv[2])
        NUM_EXAMPLES = string.atoi(sys.argv[3])
        RANDOM = bool(string.atoi(sys.argv[4]))
        TRAIN_TEST_RATIO = float(string.atof(sys.argv[5]))

    images_f = []

    imgs_a = []
    labels = []

    ids_set = set(range(NUM_ANIMALS))

    for i in range(NUM_ANIMALS):
        images_f.append([])

        pattern = re.compile(r"(.)*\.jpg")

        for fname in os.listdir(OUT_DIR+'/'+str(i)+''):
            if pattern.match(fname):
                images_f[i].append(int(fname[:-4]))

        # sort by ID
        images_f[i] = sorted(images_f[i])

    split_idx = int((1-TRAIN_TEST_RATIO) * NUM_EXAMPLES)
    print "SPLIT: ", split_idx

    for k in tqdm.tqdm(range(NUM_EXAMPLES)):
        for i in range(NUM_ANIMALS):
            ai = k

            if len(images_f) <= ai:
                ai = random.randint(0, len(images_f[i])     - 1)

            if ai >= split_idx:
                im1 = imread(OUT_DIR + '/' + str(i) + '/' + str(images_f[i][ai]) + '.jpg')
                imgs_a.append(im1)
                labels.append(i)

            if RANDOM or ai >= split_idx:
                ai = random.randint(0, len(images_f[i])-1)


            # print str(images_f[i][ai])+'.jpg'
            im1 = imread(OUT_DIR+'/'+str(i)+'/'+str(images_f[i][ai])+'.jpg')

            imgs_a.append(im1)
            labels.append(i)

    split_idx *= NUM_ANIMALS
    imgs_test = np.array(imgs_a[split_idx:])
    imgs_train = np.array(imgs_a[:split_idx])
    labels_test = np.array(labels[split_idx:])
    labels_train = np.array(labels[:split_idx])

    print "test: ", imgs_test.shape, labels_test.shape
    print "train: ", imgs_train.shape, labels_train.shape

    with h5py.File(OUT_DIR+'/imgs_multi_train.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_train)
    with h5py.File(OUT_DIR+'/imgs_multi_test.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_test)
    with h5py.File(OUT_DIR+'/labels_multi_train.h5', 'w') as hf:
        hf.create_dataset("data", data=labels_train)
    with h5py.File(OUT_DIR+'/labels_multi_test.h5', 'w') as hf:
        hf.create_dataset("data", data=labels_test)