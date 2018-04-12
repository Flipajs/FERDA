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


DATA_DIR = '/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1'
NUM_ANIMALS = 6

# NUM_EXAMPLES x NUM_A
NUM_EXAMPLES = 10
NEGATIVE_EXA_RATIO = 1

if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
        NUM_ANIMALS = string.atoi(sys.argv[2])

    images_f = []

    imgs = []
    labels = []
    names = []

    ids_set = set(range(NUM_ANIMALS))
    num_examples = []

    for i in tqdm.tqdm(range(NUM_ANIMALS)):
        # imgs.append([])

        pattern = re.compile(r"(.)*\.jpg")

        j = 0
        for fname in tqdm.tqdm(os.listdir(DATA_DIR+ '/'+str(i)+ '')):
            j += 1
            if j == 1000:
                break

            if pattern.match(fname):
                im1 = misc.imread(DATA_DIR + '/' + str(i) + '/' + fname)

                imgs.append(im1)
                labels.append(i)

        num_examples.append(j)

    labels = np.array(labels)

    X_test = np.array(imgs)
    y_test = np.array(labels)

    print X_test.shape
    print y_test.shape

    X_test = X_test.astype('float32')
    X_test /= 255

    from keras.models import model_from_json
    json_file = open("vision_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("vision_cam1_zebr_weights.h5")
    classification_model = loaded_model

    y_pred = classification_model.predict(X_test, verbose=1)


    animal_ids = []
    for i in range(NUM_ANIMALS):
        ids_ = np.argwhere(labels==i)
        animal_ids.append(ids_.flatten())

    correct = 0
    wrong = 0
    for i in range(NUM_ANIMALS):
        for j in range(1000):
            id_ = np.random.choice(animal_ids[i])

            y = y_pred[id_, :]
            best_d = np.inf
            best_i = None
            for k in range(NUM_ANIMALS):
                id_ = random.choice(animal_ids[k])
                y_ = y_pred[id_, :]
                # y = y.reshape((len(y), 1))
                # y_ = y_.reshape((len(y_), 1))

                d = np.linalg.norm(y-y_)

                if d < best_d:
                    best_d = d
                    best_i = k

            if i == best_i:
                correct += 1
            else:
                wrong += 1

    print
    print NUM_ANIMALS
    print "Correct: {}({:.2%}) Wrong: {}({:.2%})".format(correct, correct/float(correct+wrong), wrong, wrong/float(correct+wrong))

