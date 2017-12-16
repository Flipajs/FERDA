import numpy as np
import sys, os, re, random
import h5py
import string
from scipy import misc
import tqdm
import argparse

# creates dataset of size args.num_examples * (2+args.num_negative), data will
# be ordered positive1_1, positive2_1, negative1_1, .... negative_num_negative, positive1_2, positive2_2, negative1_2....

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare mats for hard')

    parser.add_argument('--datadir', type=str,
                        default='/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_zebrafish',
                        help='path to dataset')
    parser.add_argument('--num_animals', type=int,
                        default=6,
                        help='number of IDs')
    parser.add_argument('--num_examples', type=int, default=100,
                        help='num examples')
    parser.add_argument('--num_negative', type=int, default=1,
                        help='number of negative examples')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='number of negative examples')

    args = parser.parse_args()

    images_f = []

    imgs = []
    labels = []

    ids_set = set(range(args.num_animals))

    for i in range(args.num_animals):
        images_f.append([])

        pattern = re.compile(r"(.)*\.jpg")

        for fname in os.listdir(args.datadir+'/'+str(i)+''):
            if pattern.match(fname):
                images_f[i].append(fname)

    for k in tqdm.tqdm(range(args.num_examples)):
        for i in range(args.num_animals):
            ai, aj = random.sample(xrange(0, len(images_f[i])), 2)

            im1 = misc.imread(args.datadir+'/'+str(i)+'/'+images_f[i][ai])
            im2 = misc.imread(args.datadir+'/'+str(i)+'/'+images_f[i][ai])

            imgs.append(im1)
            imgs.append(im2)
            labels.append(1)
            labels.append(1)

            for l in range(args.num_negative):
                neg_ids = list(ids_set - set([i]))
                neg_id = random.choice(neg_ids)
                neg_f = random.choice(images_f[neg_id])

                im_negative = misc.imread(args.datadir+'/'+str(neg_id)+'/'+neg_f)

                imgs.append(im_negative)
                labels.append(0)

    imgs = np.array(imgs)
    # normalize..
    imgs = imgs.astype('float32')
    imgs /= 255

    split_idx = int(args.test_ratio * len(imgs))

    imgs_test = np.array(imgs[:split_idx])
    imgs_train = np.array(imgs[split_idx:])
    labels_test = np.array(labels[:split_idx])
    labels_train = np.array(labels[split_idx:])

    print "imgs TEST: {}, TRAIN: {}".format(imgs_test.shape, imgs_train.shape)
    print "labels TEST: {}, TRAIN: {}".format(labels_test.shape, labels_train.shape)

    with h5py.File(args.datadir+'/imgs_train_hard_'+str(args.num_negative)+'.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_train)
    with h5py.File(args.datadir+'/imgs_test_hard_'+str(args.num_negative)+'.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_test)
    with h5py.File(args.datadir+'/labels_train_hard_'+str(args.num_negative)+'.h5', 'w') as hf:
        hf.create_dataset("data", data=labels_train)
    with h5py.File(args.datadir+'/labels_test_hard_'+str(args.num_negative)+'.h5', 'w') as hf:
        hf.create_dataset("data", data=labels_test)

