import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import sys
import string
import numpy as np
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, AveragePooling2D, Dropout, concatenate, merge, Conv2DTranspose, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, BatchNormalization, Activation, AveragePooling2D, ZeroPadding2D
from keras import backend as K
from keras.losses import MSE
from keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import math

ROOT_DIR = '/home/threedoid/cnn_descriptor/'
# ROOT_DIR = '/Users/flipajs/Documents/wd/FERDA/cnn_exp'
DATA_DIR = ROOT_DIR + '/data'
BATCH_SIZE = 32
TWO_TESTS = True

WEIGHTS = 'cam3_zebr_weights_vgg'
NUM_PARAMS = 6

from scipy.ndimage.interpolation import rotate

def rotate_pts(ox, oy, th_deg, x, y):
    th = np.deg2rad(th_deg)
    qx = ox + math.cos(th) * (x - ox) - math.sin(th) * (y - oy)
    qy = oy + math.sin(th) * (x - ox) + math.cos(th) * (y - oy)

    return qx, qy

def myGenerator(scaler):
    global DATA_DIR, BATCH_SIZE
    with h5py.File(DATA_DIR + '/imgs_inter_train.h5', 'r') as hf:
        X_train = hf['data'][:]

    with h5py.File(DATA_DIR + '/results_inter_train.h5', 'r') as hf:
        y_train = hf['data'][:]

    # X_train = X_train.astype('float32')
    # X_train /= 255

    # y_train = np_utils.to_categorical(y_train, K)
    ii = -1
    while 1:
        x_batch = []
        y_batch = []
        ii += 1

        thetas = np.linspace(1, 360, BATCH_SIZE, endpoint=False)
        for i in range(BATCH_SIZE):
            # ii = np.random.randint(y_train.shape[0])
            # th = np.random.randint(1, 359)
            X = X_train[ii, :, :, :]
            y = y_train[ii, :]

            th = thetas[i]
            new_y = np.copy(y)
            # print new_y
            x_batch.append(rotate(X, angle=th, reshape=False, mode='nearest'))
            new_y[4] = (new_y[4] + th) % 360
            new_y[9] = (new_y[9] + th) % 360

            oy, ox = X.shape[0]/2.0, X.shape[1]/2.0
            x1, y1 = new_y[0], new_y[1]
            x2, y2 = new_y[5], new_y[6]

            new_y[0], new_y[1] = rotate_pts(ox, oy, -th, x1, y1)
            new_y[5], new_y[6] = rotate_pts(ox, oy, -th, x2, y2)

            # test ordering:
            if (new_y[0]**2 + new_y[1]**2)**0.5 > (new_y[5]**2 + new_y[6]**2)**0.5:
                a1, a2 = new_y[5:], new_y[:5]
                new_y = np.concatenate([a2, a1])

            # print new_y
            y_batch.append(new_y)

        y_batch = np.array(y_batch)
        # y_batch = scaler.transform(y_batch)
        # print x_batch.shape, y_batch.shape
        yield np.array(x_batch), y_batch

# def mean_squared_error(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true), axis=-1)
#
# def w_categorical_crossentropy(y_true, y_pred, weights):
#     nb_cl = len(weights)
#     final_mask = K.zeros_like(y_pred[:, 0])
#     y_pred_max = K.max(y_pred, axis=1)
#     y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
#     y_pred_max_mat = K.equal(y_pred, y_pred_max)
#     for c_p, c_t in product(range(nb_cl), range(nb_cl)):
#         final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
#     return K.categorical_crossentropy(y_pred, y_true) * final_mask

def my_loss2(y_true, y_pred):
    # val1 = K.mean(1 - K.abs(K.cos(y_pred[:, 4]/57.29 - y_true[:, 4]/57.29)), axis=-1)
    val = K.abs(y_pred[:, 3:4] - y_true[:, 3:4]) % 180
    theta11 = K.square(K.minimum(val, 180-val))
    val = K.abs(y_pred[:, 8:9] - y_true[:, 8:9]) % 180
    theta22 = K.square(K.minimum(val, 180-val))

    pos11 = K.square(y_pred[:, :2] - y_true[:, :2])
    pos22 = K.square(y_pred[:, 5:7] - y_true[:, 5:7])

    return K.mean(K.concatenate([pos11, pos22, theta11, theta22]), axis=-1)

def my_loss(y_true, y_pred):
    # 2pi * y, because y is normed to <0, 1> range
    # lower_0 = K.mean(K.less(y_pred[:, 4], 0.0), axis=-1)
    # higher_360 = K.mean(K.greater(y_pred[:, 4], 360.0), axis=-1)



    # # val1 = K.mean(1 - K.abs(K.cos(y_pred[:, 4]/57.29 - y_true[:, 4]/57.29)), axis=-1)
    # val = K.abs(y_pred[:, 4] - y_true[:, 4])
    # # theta11 = val
    # theta11 = K.square(K.minimum(val, 360.0-val))
    #
    # # val = np.mod(y_pred[:, 4] - y_true[:, 9], 360.0)
    # val = K.abs(y_pred[:, 4] - y_true[:, 9])
    # # theta12 = val
    # theta12 = K.square(K.minimum(val, 360.0 - val))
    #
    # val = K.abs(y_pred[:, 9] - y_true[:, 4])
    # # theta21 = val
    # theta21 = K.square(K.minimum(val, 360.0 - val))
    #
    # val = K.abs(y_pred[:, 9] - y_true[:, 9])
    # # theta22 = val
    # theta22 = K.square(K.minimum(val, 360.0 - val))
    # val1 = K.mean(K.square(np.mod(y_pred[:, 4] - y_true[:, 4], 360.0)), axis=-1)
    # val1 = K.mean(K.square(y_pred[:, 4] - y_true[:, 4]), axis=-1)
    # val2 = K.mean(K.square(1 - K.abs(K.cos(3.14*y_pred[:, 9] - 3.14*y_true[:, 9]))), axis=-1)
    # val2 = K.mean(K.square((y_pred[:, 9]%1.0 - y_true[:, 9]%1.0) % 1.0), axis=-1)
    #
    pos11 = K.sum(K.square(y_pred[:, :2] - y_true[:, :2]), axis=-1)
    pos12 = K.sum(K.square(y_pred[:, :2] - y_true[:, 5:7]), axis=-1)

    pos22 = K.sum(K.square(y_pred[:, 5:7] - y_true[:, 5:7]), axis=-1)
    pos21 = K.sum(K.square(y_pred[:, 5:7] - y_true[:, :2]), axis=-1)

    # val = K.mean(K.minimum(theta1+pos1, theta2+pos2), axis=-1)



    # matching1 = theta11 + pos11 + theta22 + pos22
    # matching2 = theta12 + pos12 + theta21 + pos21

    # regularization = 25 - K.clip(K.sum(K.square(y_pred[:, :2] - y_pred[:, 5:7]), axis=-1), 0, 25)

    matching1 = pos11 + pos22
    matching2 = pos12 + pos21
    val = matching1
    # val = K.minimum(matching1, matching2) + regularization

    # return val1 + val2 + 4*val3 + 4*val4
    return val
    # return  4*val3 + 4*val4

    # theta_diff1 = K.mean(K.square(K.min(y_true[:, 4] - y_pred[:, 4], y_true[:, 4] - (1.0 + y_pred[:, 4]))))
    # return theta_diff1
    # theta_diff2 = K.square(K.min(y_true[:, 9] - y_pred[:, 9], y_true[:, 9] - (1 + y_pred[:, 9])))
    #
    # return K.mean(K.square(y_pred[:, [0, 1, 2, 3, 5, 6, 7, 8]] - y_true[:, [0, 1, 2, 3, 5, 6, 7, 8]]) + theta_diff1 + theta_diff2, axis=-1)
    #
    # pos = K.sum(y_true * y_pred, axis=-1)
    # neg = K.max((1. - y_true) * y_pred, axis=-1)
    # return K.maximum(0., neg - pos + 1.)


def model():
    global NUM_PARAMS, DATA_DIR, CONTINUE

    img_input = Input(shape=(200, 200, 1))

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(img_input)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    conv4 = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(conv4)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = MaxPooling2D((2, 2))(x)
    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = concatenate([x, Flatten()(conv4)])
    x = Dense(16, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(NUM_PARAMS)(x)
    model = Model(img_input, x)

    if CONTINUE:
        print "loading ", DATA_DIR + "/interaction_weights_"+str(NUM_PARAMS)+"_e0.h5"
        model.load_weights(DATA_DIR + "/interaction_weights_"+str(NUM_PARAMS)+"_e0.h5")

    model.summary()

    model.compile(loss=my_loss2,
                  optimizer='adam')

    return model

def model_not_working():
    global NUM_PARAMS, DATA_DIR, CONTINUE

    img_input = Input(shape=(200, 200, 3))

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(32, (15, 15), padding='same', activation='relu')(img_input)
    x = Conv2D(32, (15, 15), dilation_rate=(2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (9, 9), dilation_rate=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(32, (7, 7), padding='same', activation='relu')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), dilation_rate=(4, 4), padding='same', activation='relu')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2), padding='same', activation='relu')(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(2, 2), padding='same', activation='relu')(x)
    # x = AveragePooling2D((2, 2))(x)
    # x = Conv2D(64, (3, 3), activation='relu')(x)
    # x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    # x = Activation('relu')(x)

    # x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    # x = AveragePooling2D((2, 2))(x)
    # x = Conv2D(64, (3, 3), activation='relu')(x)
    # x = Conv2D(128, (3, 3), activation='relu')(x)
    # x = Conv2D(256, (3, 3), activation='relu')(x)
    # x = AveragePooling2D((2, 2))(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(2, 2), activation='relu')(x)
    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = MaxPooling2D((2, 2))(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(2, 2), activation='relu')(x)
    # x = AveragePooling2D((2, 2))(x)
    # x = AveragePooling2D((2, 2))(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(4, 4), activation='relu')(x)
    # x = AveragePooling2D((2, 2))(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(4, 4), activation='relu')(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(8, 8), activation='relu')(x)

    # x = ZeroPadding2D((1, 1))(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(8, 8), activation='relu')(x)
    # x = ZeroPadding2D((1, 1))(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(4, 4), padding='same', activation='relu')(x)
    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(NUM_PARAMS)(x)
    model = Model(img_input, x)

    if CONTINUE:
        print "loading ", DATA_DIR + "/interaction_weights_"+str(NUM_PARAMS)+"_e0.h5"
        model.load_weights(DATA_DIR + "/interaction_weights_"+str(NUM_PARAMS)+"_e0.h5")

    model.summary()

    model.compile(loss=my_loss,
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

    X_train = X_train[:, :, :, :1]
    X_test = X_test[:, :, :, :1]

    y_test = y_test[:, ids]
    y_train = y_train[:, ids]

    print X_train.shape, y_train.shape, y_test.shape

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    # scaler.fit(y_train)
    # NUM_PARAMS = y_train.shape[1]
    # y_train = scaler.transform(y_train)
    # y_test = scaler.transform(y_test)


    print "NUM params: ", NUM_PARAMS
    m = model()


    m.fit(X_train, y_train, validation_split=0.05, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # for e in range(NUM_EPOCHS):
    for e in range(1):
        print e
        # m.fit_generator(myGenerator(scaler), 500, epochs=1, verbose=1)
        # m.fit_generator(myGenerator(), 500, epochs=1, verbose=1)

        m.evaluate(X_test, y_test)
        # 10. Evaluate model on test data
        pred = m.predict(X_test)

        # pred = scaler.inverse_transform(pred)
        # y_test = scaler.inverse_transform(y_test)
        # print pred2.shape

        with h5py.File(DATA_DIR+'/predictions_e'+str(e)+'.h5', 'w') as hf:
            hf.create_dataset("data", data=pred)

        m.save_weights(DATA_DIR + "/interaction_weights_"+str(NUM_PARAMS)+"_e"+str(e)+".h5")

        model_json = m.to_json()
        with open(DATA_DIR + "/interaction_model_"+str(NUM_PARAMS)+".json", "w") as json_file:
            json_file.write(model_json)

        from sklearn.metrics import mean_squared_error
        print "MSE", mean_squared_error(y_test, pred)
        from sklearn.metrics import mean_absolute_error
        print "MAE", mean_absolute_error(y_test, pred)
