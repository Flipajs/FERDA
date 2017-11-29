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
from keras import backend as K


# ROOT_DIR = '/home/threedoid/cnn_descriptor/'
ROOT_DIR = '../../data/CNN_models/interactions'
# ROOT_DIR = '/home/threedoid/cnn_descriptor/'
# ROOT_DIR = '/Users/flipajs/Documents/wd/FERDA/cnn_exp'
# DATA_DIR = ROOT_DIR + '/data'
DATA_DIR = '/datagrid/personal/smidm1/ferda/iteractions/'
BATCH_SIZE = 32
TWO_TESTS = True

WEIGHTS = 'cam3_zebr_weights_vgg'
# NUM_PARAMS = 4
NUM_PARAMS = 10


def angle_absolute_error(y_true, y_pred, backend):
    val = backend.abs(y_pred[:, 4:5] - y_true[:, 4:5]) % 180
    theta11 = backend.minimum(val, 180 - val)
    val = backend.abs(y_pred[:, 9:] - y_true[:, 9:]) % 180
    theta22 = backend.minimum(val, 180 - val)
    return theta11, theta22


def xy_absolute_error(y_true, y_pred, backend):
    pos11 = backend.abs(y_pred[:, :2] - y_true[:, :2])
    pos22 = backend.abs(y_pred[:, 5:7] - y_true[:, 5:7])
    return pos11, pos22


def interaction_loss(y_true, y_pred):
    alpha = 0.5

    theta11, theta22 = angle_absolute_error(y_true, y_pred, K)
    pos11, pos22 = xy_absolute_error(y_true, y_pred, K)
    return K.mean(K.concatenate([K.square(pos11), K.square(pos22),
                                 K.square(theta11) * alpha, K.square(theta22) * alpha]), axis=-1)


def model():
    input_shape = Input(shape=(200, 200, 3))

    # LOAD...
    from keras.models import model_from_json

    json_file = open(ROOT_DIR+'/vision_model_'+WEIGHTS+'.json', 'r')
    vision_model_json = json_file.read()
    json_file.close()
    vision_model = model_from_json(vision_model_json)
    # load weights into new model
    vision_model.load_weights(ROOT_DIR+"/vision_"+WEIGHTS+".h5")
    vision_model.layers.pop()
    vision_model.layers.pop()

    vision_model.summary()

    # The vision model will be shared, weights and all
    out_a = vision_model(input_shape)
    # out_a = Flatten()(out_a)
    #
    # out_a = Dense(256, activation='relu')(out_a)
    # out_a = Dense(128, activation='relu')(out_a)
    # out_a = Dense(32, activation='relu')(out_a)
    out_a = Dense(64, activation='relu')(out_a)
    out_a = Dense(32, activation='relu')(out_a)

    # out = Dense(128, activation='relu')(out_a)
    # out = Dense(K, activation='softmax')(out_a)
    out = Dense(NUM_PARAMS, kernel_initializer='normal', activation='linear')(out_a)
    model = Model(input_shape, out)
    model.summary()
    # 8. Compile model
    # adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss=interaction_loss,
                  optimizer='adam')

    # model.lr.set_value(0.05)

    return model


if __name__ == '__main__':
    NUM_EPOCHS = 5
    # NUM_EPOCHS = 1
    USE_PREVIOUS_AS_INIT = 0
    # K = 6
    # WEIGHTS = 'best_weights'
    OUT_NAME = 'softmax'
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
        OUT_NAME = sys.argv[5]

    with h5py.File(DATA_DIR + '/imgs_inter_train.h5', 'r') as hf:
        X_train = hf['data'][:]

    with h5py.File(DATA_DIR + '/imgs_inter_test.h5', 'r') as hf:
        X_test = hf['data'][:]

    with h5py.File(DATA_DIR + '/results_inter_train.h5', 'r') as hf:
        y_train = hf['data'][:]

    with h5py.File(DATA_DIR + '/results_inter_test.h5', 'r') as hf:
        y_test = hf['data'][:]

    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(y_train)
    # y_train = scaler.transform(y_train)
    # y_test = scaler.transform(y_test)

    m = model()
    m.fit(X_train, y_train, validation_split=0.05, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # evaluate model with standardized dataset
    # estimator = KerasRegressor(build_fn=model, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    #
    # kfold = KFold(n_splits=10, random_state=seed)
    # results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    # estimator.fit(X_train, y_train, validation_split=0.05)

    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import StandardScaler
    #
    # estimators = []
    # estimators.append(('standardize', StandardScaler()))
    # estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2)))
    # pipeline = Pipeline(estimators)
    # # X_train, X_test, y_train, y_test = train_test_split(X, Y,
    # #                                                     train_size=0.75, test_size=0.25)
    # pipeline.fit(X_train, y_train, mlp__validation_split=0.05)

    pred = m.predict(X_test)
    # pred = scaler.inverse_transform(pred)
    # y_test = scaler.inverse_transform(y_test)

    with h5py.File(DATA_DIR+'/predictions.h5', 'w') as hf:
        hf.create_dataset("data", data=pred)

    print "xy MAE", np.linalg.norm(np.vstack(xy_absolute_error(y_test, pred, np)), 2, axis=1).mean()
    # K.mean(K.concatenate(xy_absolute_error(y_test, pred)), axis=-1)
    print "angle MAE", np.vstack(xy_absolute_error(y_test, pred, np)).mean()
    # K.mean(K.concatenate(angle_absolute_error(y_test, pred, np)), axis=-1)
    # print "xy MSE", K.mean(K.square(K.concatenate(xy_absolute_error(y_test, pred, np))), axis=-1)
    # print "angle MSE", K.mean(K.square(K.concatenate(angle_absolute_error(y_test, pred, np))), axis=-1)

    # print m.score(X_test, y_test)

    m.save_weights(DATA_DIR + "/weights.h5")

    # model_json = m.to_json()
    # with open(DATA_DIR + "/model.json", "w") as json_file:
    #     json_file.write(model_json)
