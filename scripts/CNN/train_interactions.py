import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import sys
import string
import numpy as np
import time
from os.path import join
from sklearn.preprocessing import StandardScaler
import pandas as pd
try:
    from keras.utils import np_utils
    from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from keras import backend as K
    from keras.callbacks import CSVLogger
except ImportError:
    print('Warning, no keras installed.')

# ROOT_DIR = '/home/threedoid/cnn_descriptor/'
ROOT_DIR = '../../data/CNN_models/interactions'
# ROOT_DIR = '/home/threedoid/cnn_descriptor/'
# ROOT_DIR = '/Users/flipajs/Documents/wd/FERDA/cnn_exp'
# DATA_DIR = ROOT_DIR + '/data'
DATA_DIR = '/datagrid/personal/smidm1/ferda/iteractions/'

BATCH_SIZE = 32
TWO_TESTS = True

# WEIGHTS = 'cam3_zebr_weights_vgg'
WEIGHTS = 'cam3_zebr_weights_vgg_dilated'
NUM_PARAMS = 10


def angle_absolute_error(y_true, y_pred, backend, scaler=None):
    if scaler is not None:
        # y_pred1 = scaler.inverse_transform(y_pred[:, 4:5])  # this doesn't work with Tensors
        # y_pred2 = scaler.inverse_transform(y_pred[:, 9:])
        y_pred1 = y_pred[:, 4:5] * scaler[1] + scaler[0]
        y_pred2 = y_pred[:, 9:] * scaler[1] + scaler[0]
    else:
        y_pred1 = y_pred[:, 4:5]
        y_pred2 = y_pred[:, 9:]
    val = backend.abs(y_pred1 - y_true[:, 4:5]) % 180
    theta11 = backend.minimum(val, 180 - val)
    val = backend.abs(y_pred2 - y_true[:, 9:]) % 180
    theta22 = backend.minimum(val, 180 - val)
    return theta11, theta22


def xy_absolute_error(y_true, y_pred, backend):
    pos11 = backend.abs(y_pred[:, :2] - y_true[:, :2])
    pos22 = backend.abs(y_pred[:, 5:7] - y_true[:, 5:7])
    return pos11, pos22


def interaction_loss(y_true, y_pred, angle_scaler=None, alpha=1.):
    theta11, theta22 = angle_absolute_error(y_true, y_pred, K, angle_scaler)
    pos11, pos22 = xy_absolute_error(y_true, y_pred, K)
    return K.mean(K.concatenate([K.square(pos11), K.square(pos22),
                                 K.square(theta11) * alpha, K.square(theta22) * alpha]), axis=-1)


def model():
    input_shape = Input(shape=(200, 200, 3))

    # LOAD...
    # from keras.models import model_from_json
    #
    # json_file = open(join(ROOT_DIR, 'vision_model_' + WEIGHTS + '.json'), 'r')
    # vision_model_json = json_file.read()
    # json_file.close()
    # vision_model = model_from_json(vision_model_json)
    # # load weights into new model
    # vision_model.load_weights(join(ROOT_DIR, 'vision_' + WEIGHTS + '.h5'))
    # vision_model.layers.pop()
    # vision_model.layers.pop()
    # vision_model.summary()

    # animal_input = Input(shape=X_train_a.shape[1:])
    x = Conv2D(32, (3, 3), padding='same')(input_shape)
    x = Conv2D(32, (3, 3), padding='same', dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    out_a = Conv2D(16, (3, 3))(x)


    # The vision model will be shared, weights and all
    # out_a = vision_model(input_shape)
    out_a = Flatten()(out_a)
    # # out_a = Dense(256, activation='relu')(out_a)
    # # out_a = Dense(128, activation='relu')(out_a)
    # # out_a = Dense(32, activation='relu')(out_a)
    out_a = Dense(64, activation='relu')(out_a)
    out_a = Dense(32, activation='relu')(out_a)

    # out = Dense(128, activation='relu')(out_a)
    # out = Dense(K, activation='softmax')(out_a)
    out = Dense(NUM_PARAMS, kernel_initializer='normal', activation='linear')(out_a)
    return Model(input_shape, out)


def train_and_evaluate(model, params):
    # adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss=lambda x, y: interaction_loss(x, y, (angle_scaler.mean_, angle_scaler.scale_), params['loss_alpha']),
                  optimizer='adam')
    # model.lr.set_value(0.05)
    with open(join(EXPERIMENT_DIR, 'model.txt'), 'w') as fw:
        model.summary(print_fn=lambda x: fw.write(x + '\n'))
    csv_logger = CSVLogger(join(EXPERIMENT_DIR, 'log.csv'), append=True, separator=';')
    m.fit(X_train, y_train, validation_split=0.05, epochs=params['epochs'], batch_size=BATCH_SIZE, verbose=1,
          callbacks=[csv_logger])

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
    pred[:, [0, 1, 5, 6]] = xy_scaler.inverse_transform(pred[:, [0, 1, 5, 6]])
    pred[:, [4, 9]] = angle_scaler.inverse_transform(pred[:, [4, 9]])

    with h5py.File(join(EXPERIMENT_DIR, 'predictions.h5'), 'w') as hf:
        hf.create_dataset("data", data=pred)

    xy_mae = np.linalg.norm(np.vstack(xy_absolute_error(y_test, pred, np)), 2, axis=1).mean()
    angle_mae = np.vstack(angle_absolute_error(y_test, pred, np)).mean()
    results = pd.DataFrame.from_items([('xy MAE', [xy_mae]), ('angle MAE', angle_mae)])
    print(results.to_string(index=False))
    results.to_csv(join(EXPERIMENT_DIR, 'results.csv'))

    # print m.score(X_test, y_test)
    m.save_weights(join(EXPERIMENT_DIR, 'weights.h5'))
    # model_json = m.to_json()
    # with open(DATA_DIR + "/model.json", "w") as json_file:
    #     json_file.write(model_json)
    return results


if __name__ == '__main__':
    NUM_EPOCHS = 10
    # NUM_EPOCHS = 1
    # USE_PREVIOUS_AS_INIT = 0
    # K = 6
    # WEIGHTS = 'best_weights'
    # OUT_NAME = 'softmax'
    # CONTINUE = False
    # SAMPLES = 2000
    NAMES = 'ant1_x, ant1_y, ant1_major, ant1_minor, ant1_angle, ' \
            'ant2_x, ant2_y, ant2_major, ant2_minor, ant2_angle'
    FORMATS = 10 * 'f,'

    if len(sys.argv) > 1:
        DATA_DIR = ROOT_DIR + '/' + sys.argv[1]
    if len(sys.argv) > 2:
        NUM_EPOCHS = string.atoi(sys.argv[2])
    if len(sys.argv) > 3:
        BATCH_SIZE = string.atoi(sys.argv[3])
    if len(sys.argv) > 4:
        WEIGHTS = sys.argv[4]
    # if len(sys.argv) > 5:
    #     OUT_NAME = sys.argv[5]

    with h5py.File(DATA_DIR + '/imgs_inter_train.h5', 'r') as hf:
        X_train = hf['data'][:]
    with h5py.File(DATA_DIR + '/imgs_inter_test.h5', 'r') as hf:
        X_test = hf['data'][:]
    with h5py.File(DATA_DIR + '/results_inter_train.h5', 'r') as hf:
        y_train = hf['data'][:]
        # y_train_np = np.core.records.fromarrays(hf['data'][:].transpose(), names=NAMES, formats=FORMATS)
    with h5py.File(DATA_DIR + '/results_inter_test.h5', 'r') as hf:
        y_test = hf['data'][:]
        # y_test_np = np.core.records.fromarrays(hf['data'][:].transpose(), names=NAMES, formats=FORMATS)
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

    parameters = {'epochs': NUM_EPOCHS}

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # rescale data
    xy_scaler = StandardScaler()
    xy_scaler.mean_ = 0  # 100
    xy_scaler.scale_ = 1  # 100
    # y_train = xy_scaler.transform(y_train)
    # y_test = xy_scaler.transform(y_test)
    y_train[:, [0, 1, 5, 6]] = xy_scaler.transform(y_train[:, [0, 1, 5, 6]])
    angle_scaler = StandardScaler()
    angle_scaler.mean_ = 0  # 180
    angle_scaler.scale_ = 1  # 180
    y_train[:, [4, 9]] = angle_scaler.transform(y_train[:, [4, 9]])

    m = model()

    results = pd.DataFrame()

    BASE_EXPERIMENT_DIR = '/datagrid/personal/smidm1/ferda/iteractions/experiments/' + \
                          time.strftime("%y%m%d_%H%M_batch", time.localtime())
    if not os.path.exists(BASE_EXPERIMENT_DIR):
        os.mkdir(BASE_EXPERIMENT_DIR)

    for alpha in np.linspace(0, 1, 30):
        EXPERIMENT_DIR = join(BASE_EXPERIMENT_DIR, str(alpha))
        if not os.path.exists(EXPERIMENT_DIR):
            os.mkdir(EXPERIMENT_DIR)

        parameters['loss_alpha'] = alpha
        results_ = train_and_evaluate(m, parameters)
        results_['loss_alpha'] = alpha
        results = results.append(results_, ignore_index=True)

    print(results.to_string(index=False))
    results.to_csv(join(BASE_EXPERIMENT_DIR, 'batch_results.csv'))
