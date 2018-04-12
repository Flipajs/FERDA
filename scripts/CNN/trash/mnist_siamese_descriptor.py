# based on https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py

'''Trains a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

# References

- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 97.2% test accuracy after 20 epochs.
2 seconds per epoch on a Titan X Maxwell GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def contrastive_loss2(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(K.maximum(y_pred, 0)) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_base_network1(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3))(input)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(64, (3, 3))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)
    x = Conv2D(8, (3, 3))(x)
    # out_a = vision_model(animal_a)
    x = Flatten()(x)
    # out = Dense(128, activation='relu')(out_a)
    # out = Dense(K, activation='softmax')(x)

    # x = Flatten()(input)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    m = Model(input, x)
    m.summary()
    return m

def create_base_network2(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3))(input)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(64, (3, 3))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)
    x = Conv2D(8, (3, 3))(x)
    # out_a = vision_model(animal_a)
    x = Flatten()(x)
    # out = Dense(128, activation='relu')(out_a)
    # out = Dense(K, activation='softmax')(x)

    # x = Flatten()(input)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    m = Model(input, x)
    m.summary()
    return m

def create_base_network3(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3))(input)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(64, (3, 3))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = Conv2D(32, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    m = Model(input, x)
    m.summary()
    return m

def create_base_network4(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3))(input)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(64, (3, 3))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = Conv2D(32, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    x = Dense(128, activation='linear')(x)
    m = Model(input, x)
    m.summary()
    return m

def create_base_network5(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = Conv2D(32, (3, 3), activation='relu', dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(64, (3, 3))(x)
    x = Conv2D(32, (3, 3), activation='relu', dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    x = Dense(128, activation='linear')(x)
    m = Model(input, x)
    m.summary()
    return m

def create_base_network6(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = Conv2D(32, (3, 3), activation='relu', dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(64, (3, 3))(x)
    x = Conv2D(32, (3, 3), activation='relu', dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='linear')(x)
    m = Model(input, x)
    m.summary()
    return m

def create_base_network7(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(64, (3, 3))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='linear')(x)
    m = Model(input, x)
    m.summary()
    return m

def create_base_network8(input_shape):
    ''' Basiacly network 5 + padding added...
    '''
    input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(64, (3, 3))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    x = Dense(128, activation='linear')(x)
    m = Model(input, x)
    m.summary()
    return m

def create_base_network9(input_shape):
    ''' Basiacly network 5 + padding added...
    '''
    input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(64, (3, 3))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    x = Dense(128, activation='linear')(x)
    m = Model(input, x)
    m.summary()
    return m

def create_base_network10(input_shape):
    ''' Basiacly network 5 + padding added...
    '''
    input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(64, (3, 3))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    x = Dense(32, activation='linear')(x)
    m = Model(input, x)
    m.summary()
    return m


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


import h5py
import argparse

class DataGenerator(object):
    """docstring for DataGenerator"""
    def __init__(self, batch_sz, tr_pairs, tr_y):
        self.tr_pairs = tr_pairs
        self.tr_pairs_0 = tr_pairs[:, 0]
        self.tr_pairs_1 = tr_pairs[:, 1]
        self.tr_y = tr_y

        self.datagen_0 = ImageDataGenerator(rotation_range=360,
                                            # width_shift_range=0.02,
                                            # height_shift_range=0.02
                                            ).flow(self.tr_pairs_0, self.tr_y, batch_size=batch_sz, shuffle=False)

        self.datagen_1 = ImageDataGenerator(rotation_range=360,
                                            # width_shift_range=0.02,
                                            # height_shift_range=0.02
                                            ).flow(self.tr_pairs_1, self.tr_y, batch_size=batch_sz, shuffle=False)

        self.batch_sz = batch_sz
        self.samples_per_train  = (self.tr_pairs.shape[0]/self.batch_sz)

        self.cur_train_index=0
        self.cur_val_index=0

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_sz
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index=0

            p0, y = self.datagen_0.next()
            p1, _ = self.datagen_1.next()

            yield([p0, p1], y)
            # yield ([
            #             self.tr_pairs_0[self.cur_train_index:self.cur_train_index+self.batch_sz],
            #             self.tr_pairs_1[self.cur_train_index:self.cur_train_index+self.batch_sz]
            #         ],
            #         self.tr_y[self.cur_train_index:self.cur_train_index+self.batch_sz]
            #     )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train siamese CNN with contrastive loss')

    parser.add_argument('--datadir', type=str,
                        default='/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/',
                        help='path to dataset')
    parser.add_argument('--epochs', type=int,
                        default=10,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=128,
                        help='batch size')
    parser.add_argument('--weights_name', type=str, default='best_weights',
                        help='name used for saving intermediate results')
    parser.add_argument('--num_negative', type=int, default=1,
                        help='name used for saving intermediate results')
    parser.add_argument('--continue_training', type=bool, default=False,
                        help='if True, use --weights as initialisation')

    args = parser.parse_args()

    m = load_model(args.datadir+"/best_model_on6_300_ft.h5", compile=False)

    # with h5py.File(args.datadir + '/imgs_a_train.h5', 'r') as hf:
    #     test = hf['data'][:]
    #
    # with h5py.File(args.datadir + '/labels_test.h5', 'r') as hf:
    #     y = hf['data'][:]
    #
    # test = test.astype('float32')
    # test /= 255

    new_model = m.layers[2]
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist, squareform

    import os
    from scipy import misc

    descriptors = {}
    from tqdm import tqdm
    for i in range(6):
        print(i)
        imgs = []
        r_ids = []
        for f_name in tqdm(os.listdir(args.datadir + str(i))):
            # .DS_Store.. =/
            if f_name[0] == '.':
                continue
            imgs.append(misc.imread(args.datadir+str(i)+'/'+f_name))
            r_ids.append(int(f_name[:-4]))

        imgs = np.array(imgs)
        imgs = imgs.astype('float32')
        imgs /= 255

        batch = 200
        for j in tqdm(range(imgs.shape[0]/batch)):
            descs = new_model.predict(imgs[j*batch:(j+1)*batch, :, :, :])

            for k in range(min(batch, len(imgs))):
                r_id = r_ids[j*batch + k]
                descriptors[r_id] = descs[k, :]

    import pickle
    with open(args.datadir+'descriptors.pkl', 'wb') as f:
        pickle.dump(descriptors, f)

    print("DONE")

        # f, axarr = plt.subplots(6, 8)
        # axarr = axarr.flatten()
        #
        # d = pdist(descs)
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # cax = ax1.imshow(squareform(d))
        #
        # fig.colorbar(cax)
        #
        # for j in range(batch):
        #     axarr[j].imshow(imgs[j, :, :, :])
        #     axarr[j].title.set_text(str(j))
        #     axarr[j].set_axis_off()
        #
        # plt.show()

