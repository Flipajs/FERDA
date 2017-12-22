import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import h5py
import sys
import string
import numpy as np
import time
from os.path import join
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
import numbers
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
    from keras.callbacks import CSVLogger, TensorBoard
    import tensorflow as tf
except ImportError:
    print('Warning, no keras installed.')

# ROOT_DIR = '/home/threedoid/cnn_descriptor/'
ROOT_DIR = '../../data/CNN_models/interactions'
# ROOT_DIR = '/home/threedoid/cnn_descriptor/'
# ROOT_DIR = '/Users/flipajs/Documents/wd/FERDA/cnn_exp'
# DATA_DIR = ROOT_DIR + '/data'
DATA_DIR = '/datagrid/personal/smidm1/ferda/interactions/1712_36k_random'
ROOT_EXPERIMENT_DIR = '/datagrid/personal/smidm1/ferda/interactions/experiments/'
ROOT_TENSOR_BOARD_DIR = '/datagrid/personal/smidm1/ferda/interactions/tb_logs'

BATCH_SIZE = 32
TWO_TESTS = True

# WEIGHTS = 'cam3_zebr_weights_vgg'
WEIGHTS = 'cam3_zebr_weights_vgg_dilated'
NUM_PARAMS = 10

NAMES = ['ant1_x', 'ant1_y', 'ant1_major', 'ant1_minor', 'ant1_angle_deg',
         'ant2_x', 'ant2_y', 'ant2_major', 'ant2_minor', 'ant2_angle_deg']


def angle_absolute_error(y_true, y_pred, backend, scaler=None):
    if scaler is not None:
        # y_pred_ = scaler.inverse_transform(y_pred[:, 4:5])  # this doesn't work with Tensors
        y_pred_ = y_pred[:, 4:5] * scaler[1] + scaler[0]
    else:
        y_pred_ = y_pred[:, 4:5]
    val = backend.abs(y_pred_ - y_true[:, 4:5]) % 180
    return backend.minimum(val, 180 - val)


def xy_absolute_error(y_true, y_pred, backend):
    return backend.abs(y_pred[:, :2] - y_true[:, :2])


def absolute_errors(y_true, y_pred, backend, angle_scaler):
    theta = angle_absolute_error(y_true, y_pred, backend, angle_scaler)
    pos = xy_absolute_error(y_true, y_pred, backend)
    return pos, theta


def interaction_loss(y_true, y_pred, angle_scaler=None, alpha=0.5):
    assert 0 <= alpha <= 1
    sum_errors_xy, sum_errors_angle, indices = match_pred_to_gt(y_true, y_pred, K, angle_scaler)

    return K.mean(tf.gather_nd(sum_errors_xy, indices) * (1 - alpha) +
                  tf.gather_nd(sum_errors_angle, indices) * alpha)


def match_pred_to_gt(y_true, y_pred, backend, angle_scaler=None):
    """
    Return mean absolute errors for individual samples for xy and theta
    in two possible combinations of prediction and ground truth.
    """
    xy11, theta11 = absolute_errors(y_true[:, :5], y_pred[:, :5], backend, angle_scaler)
    xy22, theta22 = absolute_errors(y_true[:, 5:], y_pred[:, 5:], backend, angle_scaler)
    xy12, theta12 = absolute_errors(y_true[:, :5], y_pred[:, 5:], backend, angle_scaler)
    xy21, theta21 = absolute_errors(y_true[:, 5:], y_pred[:, :5], backend, angle_scaler)
    if backend == np:
        norm = np.linalg.norm
        int64 = np.int64
        shape = lambda x, n: x.shape[n]
    else:
        norm = tf.linalg.norm
        int64 = tf.int64
        shape = lambda x, n: backend.cast(backend.shape(x)[n], int64)
    mean_errors_xy = backend.stack((backend.mean(backend.stack((norm(xy11, axis=1), norm(xy22, axis=1))), axis=0),
                                   backend.mean(backend.stack((norm(xy12, axis=1), norm(xy21, axis=1))), axis=0)))  # shape=(2, n)
    mean_errors_angle = backend.stack((backend.mean(backend.concatenate((theta11, theta22), axis=1), axis=1),
                                      backend.mean(backend.concatenate((theta12, theta21), axis=1), axis=1)))  # shape=(2, n)
    swap_idx = backend.argmin(mean_errors_xy, axis=0)  # shape = (n,)
    indices = backend.transpose(
        backend.stack((swap_idx, backend.arange(0, shape(mean_errors_xy, 1)))))  # shape=(n, 2)
    return mean_errors_xy, mean_errors_angle, indices


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
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_shape)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
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


def train(model, train_generator, params):
    # adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss=lambda x, y: interaction_loss(x, y, (angle_scaler.mean_, angle_scaler.scale_), params['loss_alpha']),
                  optimizer='adam')
    # model.lr.set_value(0.05)
    with open(join(EXPERIMENT_DIR, 'model.txt'), 'w') as fw:
        model.summary(print_fn=lambda x: fw.write(x + '\n'))
    csv_logger = CSVLogger(join(EXPERIMENT_DIR, 'log.csv'), append=True, separator=';')
    tb = TensorBoard(log_dir=TENSOR_BOARD_DIR, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                     write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)
    model.fit_generator(train_generator, steps_per_epoch=params['steps_per_epoch'], epochs=params['epochs'],
                    verbose=1, callbacks=[csv_logger, tb])  # , validation_data=

    model.save_weights(join(EXPERIMENT_DIR, 'weights.h5'))
    return model


def evaluate(model, test_generator, y_test):
    pred = model.predict_generator(test_generator, int(len(y_test) / BATCH_SIZE))
    # pred[:, [0, 1, 5, 6]] = xy_scaler.inverse_transform(pred[:, [0, 1, 5, 6]])
    # pred[:, [4, 9]] = angle_scaler.inverse_transform(pred[:, [4, 9]])

    with h5py.File(join(EXPERIMENT_DIR, 'predictions.h5'), 'w') as hf:
        hf.create_dataset("data", data=pred)
    xy, angle, indices = match_pred_to_gt(pred, y_test.values, np)
    xy_mae = (xy[indices[:, 0], indices[:, 1]]).mean()
    angle_mae = (angle[indices[:, 0], indices[:, 1]]).mean()
    results = pd.DataFrame.from_items([('xy MAE', [xy_mae]), ('angle MAE', angle_mae)])
    print(results.to_string(index=False))
    results.to_csv(join(EXPERIMENT_DIR, 'results.csv'))
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

    hf = h5py.File(join(DATA_DIR, 'images.h5'), 'r')
    X_train = hf['train']
    X_test = hf['test']
    y_train_df = pd.read_csv(join(DATA_DIR, 'train.csv'))
    y_train = y_train_df[NAMES]
    y_test_df = pd.read_csv(join(DATA_DIR, 'test.csv'))
    y_test = y_test_df[NAMES]
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

    parameters = {'epochs': NUM_EPOCHS,
                  'loss_alpha': np.linspace(0, 1, 15),
                  # 'loss_alpha': 0.62,
                  # 'loss_alpha': 0.344827586207,
                  # 'loss_alpha': 0.66,
                  'steps_per_epoch': int(len(X_train) / BATCH_SIZE)
                  }

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # # rescale data
    xy_scaler = StandardScaler()
    xy_scaler.mean_ = 0  # 100
    xy_scaler.scale_ = 1  # 100
    # # y_train = xy_scaler.transform(y_train)
    # # y_test = xy_scaler.transform(y_test)
    # y_train[:, [0, 1, 5, 6]] = xy_scaler.transform(y_train[:, [0, 1, 5, 6]])
    angle_scaler = StandardScaler()
    angle_scaler.mean_ = 0  # 180
    angle_scaler.scale_ = 1  # 180
    # y_train[:, [4, 9]] = angle_scaler.transform(y_train[:, [4, 9]])

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow(X_train, y_train)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow(X_test, shuffle=False)

    base_experiment_name = time.strftime("%y%m%d_%H%M", time.localtime())
    # base_experiment_name = '171208_2152_batch_augmented_1k'
    BASE_EXPERIMENT_DIR = ROOT_EXPERIMENT_DIR + base_experiment_name
    BASE_TENSOR_BOARD_DIR = join(ROOT_TENSOR_BOARD_DIR, base_experiment_name)

    if not os.path.exists(BASE_EXPERIMENT_DIR):
        os.mkdir(BASE_EXPERIMENT_DIR)

    results = pd.DataFrame()

    if not isinstance(parameters['loss_alpha'], numbers.Number):
        for alpha in parameters['loss_alpha']:
            m = model()
            print('loss_alpha %f' % alpha)
            EXPERIMENT_DIR = join(BASE_EXPERIMENT_DIR, str(alpha))
            if not os.path.exists(EXPERIMENT_DIR):
                os.mkdir(EXPERIMENT_DIR)
            TENSOR_BOARD_DIR = join(BASE_TENSOR_BOARD_DIR, str(alpha))

            parameters['loss_alpha'] = alpha
            m = train(m, train_generator, parameters)
            # m.load_weights(join(EXPERIMENT_DIR, 'weights.h5'))
            results_ = evaluate(m, test_generator, y_test)
            results_['loss_alpha'] = alpha
            results = results.append(results_, ignore_index=True)

        print(results.to_string(index=False))
        results.to_csv(join(BASE_EXPERIMENT_DIR, 'results.csv'))
    else:
        m = model()
        EXPERIMENT_DIR = BASE_EXPERIMENT_DIR
        TENSOR_BOARD_DIR = BASE_TENSOR_BOARD_DIR
        m = train(m, train_generator, parameters)
        # m.load_weights(join(EXPERIMENT_DIR, 'weights.h5'))
        results = evaluate(m, test_generator, y_test)

    hf.close()
