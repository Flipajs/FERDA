import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import sys
import string
import numpy as np
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, BatchNormalization, Activation, AveragePooling2D, ZeroPadding2D
from keras import backend as K
from keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


ROOT_DIR = '/home/threedoid/cnn_descriptor/'
# ROOT_DIR = '/Users/flipajs/Documents/wd/FERDA/cnn_exp'
DATA_DIR = ROOT_DIR + '/data'
BATCH_SIZE = 32
TWO_TESTS = True

WEIGHTS = 'cam3_zebr_weights_vgg'
NUM_PARAMS = 6



def model():
    global NUM_PARAMS, DATA_DIR, CONTINUE

    img_input = Input(shape=(200, 200, 3))

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), dilation_rate=(4, 4), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(8, 8), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), dilation_rate=(8, 8), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(16, 16), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), dilation_rate=(16, 16), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    # x = Dense(64, activation='relu')(x)
    x = Dense(NUM_PARAMS, activation='linear')(x)
    model = Model(img_input, x)

    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    return model


if __name__ == '__main__':
    NUM_EPOCHS = 5
    USE_PREVIOUS_AS_INIT = 0
    # K = 6
    WEIGHTS = 'best_weights'
    CONTINUE = False
    SAMPLES = 2000

    if len(sys.argv) > 1:
        DATA_DIR = ROOT_DIR + '/' + sys.argv[1]
    if len(sys.argv) > 2:
        NUM_EPOCHS = string.atoi(sys.argv[2])
    if len(sys.argv) > 3:
        BATCH_SIZE = string.atoi(sys.argv[3])
    if len(sys.argv) > 4:
        WEIGHTS = sys.argv[4]
    if len(sys.argv) > 5:
        NUM_PARAMS = string.atoi(sys.argv[5])
    if len(sys.argv) > 6:
        CONTINUE = bool(string.atoi(sys.argv[6]))

    with h5py.File(DATA_DIR + '/imgs_inter_train.h5', 'r') as hf:
        X_train = hf['data'][:]

    with h5py.File(DATA_DIR + '/imgs_inter_test.h5', 'r') as hf:
        X_test = hf['data'][:]

    with h5py.File(DATA_DIR + '/results_inter_train.h5', 'r') as hf:
        y_train = hf['data'][:]

    with h5py.File(DATA_DIR + '/results_inter_test.h5', 'r') as hf:
        y_test = hf['data'][:]

    if NUM_PARAMS == 4:
        ids = np.array([0, 1, 5, 6])
    if NUM_PARAMS == 6:
        ids = np.array([0, 1, 2, 5, 6, 7])
    if NUM_PARAMS == 8:
        ids = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    if NUM_PARAMS == 10:
        ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    y_test = y_test[:, ids]
    y_train = y_train[:, ids]

    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    scaler.fit(y_train)
    # NUM_PARAMS = y_train.shape[1]
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)


    print "NUM params: ", NUM_PARAMS
    m = model()
    m.fit(X_train, y_train, validation_split=0.05, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    pred = m.predict(X_test)

    pred = scaler.inverse_transform(pred)
    y_test = scaler.inverse_transform(y_test)
    # print pred2.shape

    with h5py.File(DATA_DIR+'/predictions.h5', 'w') as hf:
        hf.create_dataset("data", data=pred)

    m.save_weights(DATA_DIR + "/interaction_weights_"+str(NUM_PARAMS)+".h5")

    model_json = m.to_json()
    with open(DATA_DIR + "/interaction_model_"+str(NUM_PARAMS)+".json", "w") as json_file:
        json_file.write(model_json)

    from sklearn.metrics import mean_squared_error
    print "MSE", mean_squared_error(y_test, pred)
    from sklearn.metrics import mean_absolute_error
    print "MAE", mean_absolute_error(y_test, pred)

    # print m.score(X_test, y_test)