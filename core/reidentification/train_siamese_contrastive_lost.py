# train a dimensionality reduction network for re-identification

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


import numpy as np
import yaml
from os.path import join

import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


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


def create_base_network11(input_shape):
    ''' Basiacly network 5 + padding added...
    '''
    input = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(64, (3, 3))(x)
    x = Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    x = Dense(32, activation='linear')(x)
    m = Model(input, x)
    m.summary()

    return m


def create_base_network_mobilenet_like(input_shape):
    from keras.applications.mobilenet import _conv_block, _depthwise_conv_block

    input = Input(shape=input_shape)
    alpha = 1.0
    depth_multiplier = 1
    x = _conv_block(input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 32, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 32, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 32, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 32, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 32, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 32, alpha, depth_multiplier, block_id=6)
    x = _depthwise_conv_block(x, 8, alpha, depth_multiplier, block_id=7)

    from keras.layers import GlobalAveragePooling2D
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.1)(x)
    # x = _depthwise_conv_block(x, 32, alpha, depth_multiplier,
    #                           strides=(2, 2), block_id=6)
    # x = _depthwise_conv_block(x, 32, alpha, depth_multiplier, block_id=7)
    # x = _depthwise_conv_block(x, 16, alpha, depth_multiplier, block_id=8)
    # x = _depthwise_conv_block(x, 16, alpha, depth_multiplier, block_id=9)
    # x = _depthwise_conv_block(x, 16, alpha, depth_multiplier, block_id=10)
    # x = _depthwise_conv_block(x, 16, alpha, depth_multiplier, block_id=11)
    # x = Flatten()(x)
    x = Dense(32, activation='linear')(x)
    m = Model(input, x)

    m.summary()
    return m


def create_basenetwork_squeezenet_like(input_shape):
    from keras_squeezenet.squeezenet import fire_module
    input = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)

    x = fire_module(x, fire_id=8, squeeze=64, expand=256)

    # x = Dropout(0.5, name='drop9')(x)
    #
    # x = Conv2D(32, (1, 1), padding='valid', name='conv10')(x)
    # x = Activation('linear', name='relu_conv10')(x)

    x = Flatten()(x)
    x = Dense(32, activation='linear')(x)

# x = GlobalAveragePooling2D()(x)
#
#     x = Dense(32, activation='linear')(x)
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

        self.datagen_0 = ImageDataGenerator(# rotation_range=360,
                                            # width_shift_range=0.02,
                                            # height_shift_range=0.02
                                            ).flow(self.tr_pairs_0, self.tr_y, batch_size=batch_sz, shuffle=False)

        self.datagen_1 = ImageDataGenerator(# rotation_range=360,
                                            # width_shift_range=0.02,
                                            # height_shift_range=0.02
                                            ).flow(self.tr_pairs_1, self.tr_y, batch_size=batch_sz, shuffle=False)

        self.batch_sz = batch_sz
        self.samples_per_train = (self.tr_pairs.shape[0]/self.batch_sz)

        self.cur_train_index=0
        self.cur_val_index=0

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_sz
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index=0

            p0, y = next(self.datagen_0)
            p1, _ = next(self.datagen_1)

            yield([p0, p1], y)
            # yield ([
            #             self.tr_pairs_0[self.cur_train_index:self.cur_train_index+self.batch_sz],
            #             self.tr_pairs_1[self.cur_train_index:self.cur_train_index+self.batch_sz]
            #         ],
            #         self.tr_y[self.cur_train_index:self.cur_train_index+self.batch_sz]
            #     )


def train(datadir, epochs=10, batch_size=128, continue_training=False):
    with h5py.File(datadir + '/descriptor_cnn_imgs_train.h5', 'r') as hf:
        tr_pairs = hf['data'][:]
    with h5py.File(datadir + '/descriptor_cnn_imgs_test.h5', 'r') as hf:
        te_pairs = hf['data'][:]
    with h5py.File(datadir + '/descriptor_cnn_labels_train.h5', 'r') as hf:
        tr_y = hf['data'][:]
    with h5py.File(datadir + '/descriptor_cnn_labels_test.h5', 'r') as hf:
        te_y = hf['data'][:]
    parameters = yaml.safe_load(open(join(datadir, 'descriptor_cnn_params.yaml'), 'r'))
    # # normalize..
    tr_pairs = tr_pairs.astype('float32')
    tr_pairs /= 255
    te_pairs = te_pairs.astype('float32')
    te_pairs /= 255
    print("train shape {}, min: {} max: {}".format(tr_pairs.shape, tr_pairs.min(), tr_pairs.max()))
    print("test shape {}, min: {} max: {}".format(te_pairs.shape, te_pairs.min(), te_pairs.max()))
    print("y_train shape {}, min: {} max: {}".format(tr_y.shape, tr_y.min(), tr_y.max()))
    input_shape = tr_pairs.shape[2:]
    print(input_shape)
    architectures = [
        # create_base_network1,
        # create_base_network2,
        # create_base_network3,
        # create_base_network4,
        # create_base_network5,
        # create_base_network6,
        # create_base_network7,
        # create_base_network8,
        # create_base_network9,
        create_base_network10,
        # create_base_network11,
        # create_base_network_mobilenet_like,
        # create_basenetwork_squeezenet_like
    ]
    datagen = DataGenerator(batch_size, tr_pairs, tr_y)
    for architecture in architectures:
        print("")
        print("")
        print("###################################")
        print(architecture)
        # network definition
        base_network = architecture(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

        model = Model([input_a, input_b], distance)
        model.summary()

        if continue_training:
            from keras.models import load_model
            model = load_model(datadir + '/weights.h5', compile=False)

        # train
        # rms = RMSprop()
        model.compile(loss=contrastive_loss2, optimizer='adam', metrics=[accuracy])

        checkpoint = ModelCheckpoint(datadir + '/weights.h5', monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        model.fit_generator(generator=datagen.next_train(), samples_per_epoch=datagen.samples_per_train,
                            nb_epoch=epochs, validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
                            callbacks=callbacks_list)

        # model.fit_generator(datagen.flow(te_pairs, tr_y, batch_size=batch_size),
        #                     steps_per_epoch=,
        #                     epochs=epochs,
        #                     validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

        # model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
        #           batch_size=batch_size,
        #           epochs=epochs,
        #           validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

        # compute final accuracy on training and test sets
        y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
        tr_acc = compute_accuracy(tr_y, y_pred)
        y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
        te_acc = compute_accuracy(te_y, y_pred)

        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train siamese reidentification with contrastive loss')

    parser.add_argument('--datadir', type=str, help='path to dataset')
    parser.add_argument('--epochs', type=int,
                        default=10,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=128,
                        help='batch size')
    parser.add_argument('--continue_training', default=False, action='store_true',
                        help='if True, use --weights as initialisation')

    args = parser.parse_args()
    print(args)

    train(args.datadir, args.epochs, args.batch_size, args.continue_training)
