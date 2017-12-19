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
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, BatchNormalization, Activation


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train siamese CNN with HARD loss (https://github.com/DagnyT/hardnet/)')

    parser.add_argument('--datadir', type=str,
                        default='/Users/flipajs/Documents/wd/FERDA/CNN_hard_datagen',
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

    with h5py.File(args.datadir + '/imgs_train_hard_' + str(args.num_negative) + '.h5', 'r') as hf:
        tr_pairs = hf['data'][:]

    with h5py.File(args.datadir + '/imgs_test_hard_' + str(args.num_negative) + '.h5', 'r') as hf:
        te_pairs = hf['data'][:]

    with h5py.File(args.datadir + '/labels_train_hard_' + str(args.num_negative) + '.h5', 'r') as hf:
        tr_y = hf['data'][:]

    with h5py.File(args.datadir + '/labels_test_hard_' + str(args.num_negative) + '.h5', 'r') as hf:
        te_y = hf['data'][:]

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
                     create_base_network8,
                     ]

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
        # train
        # rms = RMSprop()
        model.compile(loss=contrastive_loss, optimizer='adam', metrics=[accuracy])
        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

        # compute final accuracy on training and test sets
        y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
        tr_acc = compute_accuracy(tr_y, y_pred)
        y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
        te_acc = compute_accuracy(te_y, y_pred)

        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))