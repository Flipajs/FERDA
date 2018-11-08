from __future__ import print_function
import h5py
import keras
import sys
import string
import numpy as np
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import sys

if __name__ == '__main__':
    WEIGHTS_FILE = 'weights.h5'
    ROOT_DIR = '/home/threedoid/cnn_descriptor/'
    if len(sys.argv) > 1:
        WEIGHTS_FILE = sys.argv[1]

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(WEIGHTS_FILE)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    classification_model = loaded_model

    N = 100

    # and on cam1 sequence
    print("Cam1")
    DATA_DIR = ROOT_DIR + '/data_cam1'



    with h5py.File(DATA_DIR + '/imgs_a_test.h5', 'r') as hf:
        X_test_a = hf['data'][:]

    with h5py.File(DATA_DIR + '/imgs_b_test.h5', 'r') as hf:
        X_test_b = hf['data'][:]

    with h5py.File(DATA_DIR + '/labels_test.h5', 'r') as hf:
        y_test = hf['data'][:]

    X_test_a = X_test_a.astype('float32')
    X_test_b = X_test_b.astype('float32')
    X_test_a /= 255
    X_test_b /= 255

    results = classification_model.evaluate([X_test_a, X_test_b], y_test, verbose=1)
    print(results)

    print("cam3")
    DATA_DIR = ROOT_DIR + '/data_cam3'
    with h5py.File(DATA_DIR + '/imgs_a_test.h5', 'r') as hf:
        X_test_a = hf['data'][:]

    with h5py.File(DATA_DIR + '/imgs_b_test.h5', 'r') as hf:
        X_test_b = hf['data'][:]

    with h5py.File(DATA_DIR + '/labels_test.h5', 'r') as hf:
        y_test = hf['data'][:]

    X_test_a = X_test_a.astype('float32')
    X_test_b = X_test_b.astype('float32')
    X_test_a /= 255
    X_test_b /= 255

    results = classification_model.evaluate([X_test_a, X_test_b], y_test, verbose=1)
    print(results)


    print("zebrafish")
    # and on zebrafish sequence
    DATA_DIR = ROOT_DIR + '/data_zebrafish'
    with h5py.File(DATA_DIR + '/imgs_a_test.h5', 'r') as hf:
        X_test_a = hf['data'][:]

    with h5py.File(DATA_DIR + '/imgs_b_test.h5', 'r') as hf:
        X_test_b = hf['data'][:]

    with h5py.File(DATA_DIR + '/labels_test.h5', 'r') as hf:
        y_test = hf['data'][:]

    X_test_a = X_test_a.astype('float32')
    X_test_b = X_test_b.astype('float32')
    X_test_a /= 255
    X_test_b /= 255

    results = classification_model.evaluate([X_test_a, X_test_b], y_test, verbose=1)
    print(results)