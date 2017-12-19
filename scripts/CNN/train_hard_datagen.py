import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import random
from random import randint as ri

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare mats for hard')

    parser.add_argument('--datadir', type=str,
                        default='/Users/flipajs/Documents/wd/FERDA/CNN_hard_datagen',
                        help='path to dataset')
    parser.add_argument('--num_examples', type=int, default=200,
                        help='num examples')
    parser.add_argument('--im_size', type=int, default=32,
                        help='im size')
    parser.add_argument('--num_negative', type=int, default=1,
                        help='im size')
    parser.add_argument('--test_ratio', type=float, default=0.5,
                        help='number of negative examples')

    args = parser.parse_args()

    imgs = []
    labels = []
    num_swap = 0
    for i in range(args.num_examples):
        r, g, b = ri(0, 255), ri(0, 255), ri(0, 255)

        radius = 10
        circle1 = np.zeros((args.im_size, args.im_size, 3), dtype=np.uint8)
        # cv2.circle(circle1, (ri(5, 27), ri(5, 27)), ri(2, 13), (r, g, b), -1)
        cv2.circle(circle1, (ri(5, 27), ri(5, 27)), radius, (r, g, b), -1)

        circle2 = np.zeros((args.im_size, args.im_size, 3), dtype=np.uint8)
        cv2.circle(circle2, (ri(5, 27), ri(5, 27)), radius, (r, g, b), -1)

        r, g, b = ri(0, 255), ri(0, 255), ri(0, 255)
        size = 15
        rectangle1 = np.zeros((args.im_size, args.im_size, 3), dtype=np.uint8)
        # size = ri(2, 30)
        x = ri(0, 31-size)
        y = ri(0, 31-size)
        rectangle1[x:x+size, y:y+size, :] = [r, g, b]

        rectangle2 = np.zeros((args.im_size, args.im_size, 3), dtype=np.uint8)
        # size = ri(2, 30)
        x = ri(0, 31 - size)
        y = ri(0, 31 - size)
        rectangle2[x:x + size, y:y + size, :] = [r, g, b]

        noise_amount = 0
        circle1 = np.clip(circle1 + np.asarray(np.random.rand(args.im_size, args.im_size, 3) * noise_amount, dtype=np.uint8), 0, 255)
        circle2 = np.clip(circle2 + np.asarray(np.random.rand(args.im_size, args.im_size, 3) * noise_amount, dtype=np.uint8), 0, 255)
        rectangle1 = np.clip(rectangle1 + np.asarray(np.random.rand(args.im_size, args.im_size, 3) * noise_amount, dtype=np.uint8), 0, 255)
        rectangle2 = np.clip(rectangle2 + np.asarray(np.random.rand(args.im_size, args.im_size, 3) * noise_amount, dtype=np.uint8), 0, 255)

        # plt.imshow(circle)
        # plt.figure()
        # plt.imshow(rectangle)
        # plt.show()

        # random swap...
        if random.random() <= 0.5:
            num_swap += 1
            imgs.append(rectangle1)
            imgs.append(rectangle2)
            imgs.append(circle1)
        else:
            imgs.append(circle1)
            imgs.append(circle2)
            imgs.append(rectangle1)

        labels.append(1)
        labels.append(1)
        labels.append(0)

    print "#swap: {:.2%}".format(num_swap / float(args.num_examples))

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

    with h5py.File(args.datadir + '/imgs_train_hard_' + str(args.num_negative) + '.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_train)
    with h5py.File(args.datadir + '/imgs_test_hard_' + str(args.num_negative) + '.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs_test)
    with h5py.File(args.datadir + '/labels_train_hard_' + str(args.num_negative) + '.h5', 'w') as hf:
        hf.create_dataset("data", data=labels_train)
    with h5py.File(args.datadir + '/labels_test_hard_' + str(args.num_negative) + '.h5', 'w') as hf:
        hf.create_dataset("data", data=labels_test)