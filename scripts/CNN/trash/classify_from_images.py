from __future__ import print_function
import numpy as np
import sys, os, re, random
import h5py
import string
from scipy import misc
import tqdm

import h5py
import keras
import sys
import string
import numpy as np
import shutil
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

def classify_imgs(results_map):
    global DATA_DIR, MODEL_NAME
    X_test = np.array(imgs)

    print(X_test.shape)

    X_test = X_test.astype('float32')
    X_test /= 255

    from keras.models import model_from_json
    json_file = open(DATA_DIR + "/" + MODEL_NAME + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(DATA_DIR + "/" + MODEL_NAME + ".h5")
    classification_model = loaded_model

    y_pred = classification_model.predict(X_test, verbose=1)

    for i in range(len(imgs)):
        id_ = int(names[i][1][:-4])
        results_map[id_] = y_pred[i, :]


if __name__ == '__main__':
    DATA_DIR = sys.argv[1]
    MODEL_NAME = sys.argv[2]
    BATCH_SIZE = sys.argv[3]

    imgs = []
    names = []

    pattern = re.compile(r"(.)*\.jpg")

    results_map = {}

    i = 0
    for fname in tqdm.tqdm(os.listdir(DATA_DIR+ '/test')):
        if pattern.match(fname):
            i += 1

            im1 = misc.imread(DATA_DIR + '/test/' + fname)

            imgs.append(im1)
            names.append((i, fname))

            # TODO: fix...
            if i == BATCH_SIZE:
                classify_imgs(results_map)
                imgs = []
                names = []

    classify_imgs(results_map)

    import pickle
    with open(DATA_DIR+'/softmax_results_map.pkl', 'wb') as f:
        pickle.dump(results_map, f)
