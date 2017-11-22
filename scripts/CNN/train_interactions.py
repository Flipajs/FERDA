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
from keras.layers import Dense
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

    input_shape = Input(shape=(200, 200, 3))

    x = Conv2D(32, (3, 3))(input_shape)
    x = Conv2D(32, (15, 15), dilation_rate=(2, 2))(x)
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

    x = Flatten()(x)

    out = Dense(NUM_PARAMS, activation='linear')(x)

    model = Model(input_shape, out)

    if CONTINUE:
        model.load_weights(DATA_DIR + "/interaction_weights_"+str(NUM_PARAMS)+".h5")

    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    return model



# def model():
#     global NUM_PARAMS, DATA_DIR, CONTINUE
#     input_shape = Input(shape=(200, 200, 3))
#
#     # LOAD...
#     from keras.models import model_from_json
#
#     json_file = open(ROOT_DIR+'/vision_model_'+WEIGHTS+'.json', 'r')
#     vision_model_json = json_file.read()
#     json_file.close()
#     vision_model = model_from_json(vision_model_json)
#     # load weights into new model
#     vision_model.load_weights(ROOT_DIR+"/vision_"+WEIGHTS+".h5")
#     # vision_model.layers.pop()
#     # vision_model.layers.pop()
#
#     vision_model.summary()
#
#     # The vision model will be shared, weights and all
#     out_a = vision_model(input_shape)
#     # out_a = Flatten()(out_a)
#     #
#     # out_a = Dense(256, activation='relu')(out_a)
#     # out_a = Dense(128, activation='relu')(out_a)
#     # out_a = Dense(32, activation='relu')(out_a)
#     out_a = Dense(64, activation='relu')(out_a)
#     out_a = Dense(32, activation='relu')(out_a)
#
#     # out = Dense(128, activation='relu')(out_a)
#     # out = Dense(K, activation='softmax')(out_a)
#     previous = NUM_PARAMS - 2
#     if NUM_PARAMS == 4 or CONTINUE:
#         previous = NUM_PARAMS
#
#     out = Dense(previous, kernel_initializer='normal', activation='linear')(out_a)
#
#     model = Model(input_shape, out)
#
#     model.load_weights(DATA_DIR + "/interaction_weights_"+str(previous)+".h5")
#     # model.load_weights(DATA_DIR + "/interaction_weights.h5")
#
#     out = Dense(NUM_PARAMS, kernel_initializer='normal', activation='linear')(out_a)
#     model = Model(input_shape, out)
#     #
#     # model =
#
#     model.summary()
#     # 8. Compile model
#     # adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#     model.compile(loss='mean_squared_error',
#                   optimizer='adam')
#
#     # model.lr.set_value(0.05)
#
#     return model


if __name__ == '__main__':
    NUM_EPOCHS = 5
    USE_PREVIOUS_AS_INIT = 0
    K = 6
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