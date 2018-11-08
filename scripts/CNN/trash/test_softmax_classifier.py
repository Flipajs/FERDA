from __future__ import print_function
from __future__ import absolute_import
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


DATA_DIR = '/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1'
NUM_ANIMALS = 6

# NUM_EXAMPLES x NUM_A
NUM_EXAMPLES = 10
NEGATIVE_EXA_RATIO = 1

if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
        NUM_ANIMALS = string.atoi(sys.argv[2])
        MODEL_NAME = sys.argv[3]

    images_f = []

    imgs_a = []
    labels = []
    names = []

    ids_set = set(range(NUM_ANIMALS))
    num_examples = []

    for i in tqdm.tqdm(range(NUM_ANIMALS)):
        images_f.append([])

        pattern = re.compile(r"(.)*\.jpg")

        j = 0
        for fname in tqdm.tqdm(os.listdir(DATA_DIR+ '/'+str(i)+ '')):
            j += 1
            # if j == 100:
            #     break

            if pattern.match(fname):
                im1 = misc.imread(DATA_DIR + '/' + str(i) + '/' + fname)

                imgs_a.append(im1)
                labels.append(i)
                names.append((i, fname))

        num_examples.append(j)

    X_test = np.array(imgs_a)
    y_test = np.array(labels)

    print(X_test.shape)
    print(y_test.shape)

    X_test = X_test.astype('float32')
    X_test /= 255

    from keras.models import model_from_json
    json_file = open(DATA_DIR+"/"+MODEL_NAME+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(DATA_DIR+"/"+MODEL_NAME+".h5")
    classification_model = loaded_model

    y_test_c = np_utils.to_categorical(y_test, NUM_ANIMALS)

    from sklearn.metrics import accuracy_score
    y_pred = classification_model.predict(X_test, verbose=1)

    y_pred_class = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_pred_class, y_test)
    print("Accuracy: {:.2%}".format(accuracy))

    np.set_printoptions(precision=2, suppress=True)

    not_classified = [0] * NUM_ANIMALS
    missclassified = [0] * NUM_ANIMALS

    try:
        shutil.rmtree(DATA_DIR+'/errors')
    except Exception as e:
        print(e)

    try:
        os.mkdir(DATA_DIR+'/errors')
    except:
        pass

    for i in range(NUM_ANIMALS):
        try:
            os.mkdir(DATA_DIR + '/errors/'+str(i))
        except:
            pass

    results_map = {}
    for i in range(len(y_test)):
        id_ = int(names[i][1][:-4])
        results_map[id_] = y_pred[i, :]

        if y_pred_class[i] != y_test[i]:
            shutil.copy(DATA_DIR+'/'+str(y_test[i])+'/'+names[i][1],
                        DATA_DIR + '/errors/'+str(y_test[i])+'/'+names[i][1])

            not_classified[y_test[i]] += 1
            missclassified[y_pred_class[i]] += 1
            # print names[i], y_pred[i]

    import pickle
    with open(DATA_DIR+'/softmax_results_map.pkl', 'wb') as f:
        pickle.dump(results_map, f)

    not_classified = np.array(not_classified)
    missclassified = np.array(missclassified)
    print("NOT CLASSIFIED: ", not_classified)
    print("MISSCLASSIFIED: ", missclassified)

    not_classified = not_classified.astype('float32')
    missclassified = missclassified.astype('float32')
    for i in range(NUM_ANIMALS):
        not_classified[i] /= float(num_examples[i])
        missclassified[i] /= float(num_examples[i])

    print()
    print("NOT CLASSIFIED: ", not_classified)
    print("MISSCLASSIFIED: ", missclassified)

    # results = classification_model.evaluate(X_test, y_test, verbose=1)
    # print results

    from .errors_web import make_web
    data = "{}, Accuracy: {:.2%}, FN: {}, FP: {}".format(DATA_DIR,
                                                                                 accuracy,
                                                                                 not_classified,
                                                                                 missclassified)

    make_web(DATA_DIR+"/errors", data)

