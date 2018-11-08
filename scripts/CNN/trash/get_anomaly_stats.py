from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import sys
import numpy as np
from keras.models import model_from_json
from keras.models import Model


ROOT_DIR = '/home/threedoid/cnn_descriptor/'
DATA_DIR = ROOT_DIR + '/data'
BATCH_SIZE = 32
TWO_TESTS = True

if __name__ == '__main__':
    NUM_EPOCHS = 5
    USE_PREVIOUS_AS_INIT = 0
    K = 6
    WEIGHTS = 'best_weights'
    MODEL_NAME = 'softmax'
    CONTINUE = False
    SAMPLES = 2000

    if len(sys.argv) > 1:
        DATA_DIR = ROOT_DIR + '/' + sys.argv[1]
    if len(sys.argv) > 2:
        MODEL_NAME = sys.argv[2]

    # 3. Import libraries and modules
    np.random.seed(123)  # for reproducibility

    im_dim = 3
    im_h = 90
    im_w = 90

    json_file = open(DATA_DIR +"/" + MODEL_NAME + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(DATA_DIR +"/" + MODEL_NAME + ".h5")

    classification_model = loaded_model
    classification_model.summary()
    # classification_model.layers.pop()
    #
    # print classification_model.layers[-1].output
    # # classification_model.outputs = [classification_model.layers[-1].output]
    # classification_model.output_layers = [classification_model.layers[-1]]  # added this line in addition to zo7 solution
    # classification_model.layers[-1].outbound_nodes = []
    #
    # classification_model.summary()

    model = Model(input=[classification_model.inputs[0]], output=[classification_model.output, classification_model.layers[2].input])
    model.summary()

    # 9. Fit model on training data
    best_eval_consecutive = 0
    best_eval_random = 0

    with h5py.File(DATA_DIR + '/imgs_multi_train.h5', 'r') as hf:
        X_train = hf['data'][:]

    with h5py.File(DATA_DIR + '/labels_multi_train.h5', 'r') as hf:
        y_train = hf['data'][:]

    X_train = X_train.astype('float32')
    X_train /= 255

    y = model.predict(X_train)
    print(y[0].shape)
    print(y[1].shape)

    with h5py.File(DATA_DIR+'/penultimate_layer.h5', 'w') as hf:
        hf.create_dataset("data", data=y[1])


