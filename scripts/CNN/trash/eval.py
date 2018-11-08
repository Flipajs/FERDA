from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from past.utils import old_div
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
    WEIGHTS_FILE = 'best_weights.h5'
    if len(sys.argv) > 1:
        WEIGHTS_FILE = sys.argv[1]


    ROOT_DIR = '/home/threedoid/cnn_descriptor/'

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

    y_predict = classification_model.predict([X_test_a, X_test_b])
    y_predict = np.reshape(y_predict, (y_predict.shape[0], ))
    y_predict = y_predict > 0.975
    print(y_predict)
    print(y_predict.min(), y_predict.max(), np.mean(y_predict))
    print(y_predict.shape)
    correct = y_predict == y_test
    print(correct.shape)
    print(correct[0])
    print(np.sum(correct))

    total_correct = 0
    i = 0
    while i < len(y_predict):
        if y_predict[i] == True:
            if y_predict[i+1] == False and y_predict[i+2] == False and y_predict[i+3] == False and y_predict[i+4] == False and y_predict[i+5] == False:
                total_correct += 1

        i += 6

    print("Total correct: ", total_correct)

    correct_match = y_test[correct] == 1
    correct_match = np.sum(correct_match)
    print(correct_match, old_div(correct_match,float(y_predict.shape[0])))
    print()

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