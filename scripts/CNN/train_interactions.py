import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import h5py
import sys
import string
import numpy as np
import time
from os.path import join
# from sklearn.preprocessing import StandardScaler
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
import skimage.transform
import fire
from core.region.transformableregion import TransformableRegion
from keras.models import model_from_yaml, model_from_json
import warnings


ROOT_EXPERIMENT_DIR = '/datagrid/personal/smidm1/ferda/interactions/experiments/'
ROOT_TENSOR_BOARD_DIR = '/datagrid/personal/smidm1/ferda/interactions/tb_logs'
BATCH_SIZE = 32
COLUMNS = ['x', 'y', 'major', 'minor', 'angle_deg']  # , 'dx', 'dy']


class TrainInteractions:
    def __init__(self, num_objects=None):
        self.num_objects = num_objects

    def columns(self):
        names = []
        for i in range(self.num_objects):
            names.extend(['%d_%s' % (i, c) for c in COLUMNS])
            # names.extend(['ant%d_%s' % (i + 1, c) for c in COLUMNS])
        return names

    def col2idx(self, obj, col):
        return obj * len(COLUMNS) + COLUMNS.index(col)

    # col2idx = {j: {key: i + (j * len(COLUMNS)) for i, key in enumerate(COLUMNS)}
    #            for j in range(NUM_OBJECTS)}

    def col2idx_(self, obj, col):
        return slice(self.col2idx(obj, col), self.col2idx(obj, col) + 1)
    # COL2IDX_ = {j: {key: slice(i + (j * len(COLUMNS)),
    #                            i + (j * len(COLUMNS)) + 1)
    #                 for i, key in enumerate(COLUMNS)}
    #             for j in range(NUM_OBJECTS)}

    @staticmethod
    def toarray(struct_array):
        types = [x[1] for x in struct_array.dtype.descr]
        all(x == types[0] for x in types)
        return struct_array.view(types[0]).reshape(struct_array.shape + (-1,))

    def tostruct(self, ndarray):
        formats = len(self.columns()) * 'f,'
        return np.core.records.fromarrays(ndarray.transpose(), names=', '.join(self.columns()), formats=formats)

    # def angle_absolute_error(y_true, y_pred, i, j, backend, scaler=None):
    #     if scaler is not None:
    #         # y_pred_ = scaler.inverse_transform(y_pred[:, 4:5])  # this doesn't work with Tensors
    #         y_pred_ = y_pred[:, NAME2COL[i]['angle_deg']] * scaler[1] + scaler[0]
    #     else:
    #         y_pred_ = y_pred[:, NAME2COL[i]['angle_deg']]
    #     val = backend.abs(y_pred_ - y_true[:, NAME2COL[j]['angle_deg']]) % 180
    #     return backend.minimum(val, 180 - val)

    def angle_absolute_error_direction_agnostic(self, angles_pred, angles_true, backend, scaler=None):
        if scaler is not None:
            # y_pred_ = scaler.inverse_transform(y_pred[:, 4:5])  # this doesn't work with Tensors
            angles_pred_ = angles_pred * scaler[1] + scaler[0]
        else:
            angles_pred_ = angles_pred
        val = backend.abs(angles_pred_ - angles_true) % 180
        return backend.minimum(val, 180 - val)


    def angle_absolute_error(self, angles_pred, angles_true, backend, scaler=None):
        if scaler is not None:
            # y_pred_ = scaler.inverse_transform(y_pred[:, 4:5])  # this doesn't work with Tensors
            angles_pred_ = angles_pred * scaler[1] + scaler[0]
        else:
            angles_pred_ = angles_pred
        angles_pred_ %= 360
        angles_true %= 360
        return backend.minimum(
            backend.abs(angles_pred_ - angles_true),
            180 - backend.abs(angles_pred_ % 180 - angles_true % 180)
        )

    def xy_absolute_error(self, y_true, y_pred, i, j, backend):
        return backend.abs(backend.concatenate(
            (y_pred[:, self.col2idx_(i, 'x')] - y_true[:, self.col2idx_(j, 'x')],
             y_pred[:, self.col2idx_(i, 'y')] - y_true[:, self.col2idx_(j, 'y')]),
            axis=1))

    def delta_error(self, y_true, y_pred, i, j, backend):
        """
        absolute difference between dx and dy in y_pred and y_true

        :param y_true:
        :param y_pred:
        :param i: y_true object index
        :param j: y_pred object index
        :param backend:
        :return: shape=(n, 2)
        """
        dx = y_true[:, self.col2idx_(i, 'dx')] - y_pred[:, self.col2idx_(j, 'dx')]
        dy = y_true[:, self.col2idx_(i, 'dy')] - y_pred[:, self.col2idx_(j, 'dy')]
        return backend.concatenate((backend.abs(dx), backend.abs(dy)), axis=1)

    def interaction_loss_angle(self, y_true, y_pred, angle_scaler=None, alpha=0.5):
        assert 0 <= alpha <= 1
        mean_errors_xy, mean_errors_angle, indices = self.match_pred_to_gt(y_true, y_pred, K, angle_scaler)
        if self.num_objects == 2:
            errors_xy = tf.gather_nd(mean_errors_xy, indices)
            errors_angle = tf.gather_nd(mean_errors_angle, indices)
        elif self.num_objects == 1:
            errors_xy = mean_errors_xy
            errors_angle = mean_errors_angle
        else:
            assert False, 'not implemented'
        return K.mean(errors_xy * (1 - alpha) + errors_angle * alpha)

    def interaction_loss_dxdy(self, y_true, y_pred, angle_scaler=None, alpha=0.5):
        assert 0 <= alpha <= 1
        mean_errors_xy, mean_errors_delta, indices = self.match_pred_to_gt_dxdy(y_true, y_pred, K, angle_scaler)

        return K.mean(tf.gather_nd(mean_errors_xy, indices) * (1 - alpha) +
                      tf.gather_nd(mean_errors_delta, indices) * alpha)

    def match_pred_to_gt_dxdy(self, y_true, y_pred, backend):
        """
        Return mean absolute errors for individual samples for xy and theta
        in two possible combinations of prediction and ground truth.
        """
        bk = backend
        xy = {}
        delta = {}
        for i, j in ((0, 0), (1, 1), (0, 1), (1, 0)):
            xy[(i, j)] = self.xy_absolute_error(y_true, y_pred, i, j,
                                           bk)  # shape=(n, 2) [[x_abs_err, y_abs_err], [x_abs_err, y_abs_err], ...]
            delta[(i, j)] = self.delta_error(y_true, y_pred, i, j, bk)

        if backend == np:
            norm = np.linalg.norm
            int64 = np.int64
            shape = lambda x, n: x.shape[n]
        else:
            norm = tf.linalg.norm
            int64 = tf.int64
            shape = lambda x, n: bk.cast(bk.shape(x)[n], int64)

        mean_errors_xy = bk.stack((
            bk.mean(bk.stack((norm(xy[0, 0], axis=1),
                              norm(xy[1, 1], axis=1))), axis=0),
            bk.mean(bk.stack((norm(xy[0, 1], axis=1),
                              norm(xy[1, 0], axis=1))), axis=0)))  # shape=(2, n)
        mean_errors_delta = bk.stack((
            bk.mean(bk.stack((bk.sum(delta[0, 0], axis=1),
                              bk.sum(delta[1, 1], axis=1))), axis=0),
            bk.mean(bk.stack((bk.sum(delta[0, 1], axis=1),
                              bk.sum(delta[1, 0], axis=1))), axis=0)))  # shape=(2, n)

        swap_idx = bk.argmin(mean_errors_xy, axis=0)  # shape = (n,)

        indices = backend.transpose(
            backend.stack((swap_idx, backend.arange(0, shape(mean_errors_xy, 1)))))  # shape=(n, 2)
        return mean_errors_xy, mean_errors_delta, indices

    def match_pred_to_gt(self, y_true, y_pred, backend, angle_scaler=None):
        """
        Return mean absolute errors for individual samples for xy and theta
        in two possible combinations of prediction and ground truth.
        """
        bk = backend
        if backend == np:
            norm = np.linalg.norm
            int64 = np.int64
            shape = lambda x, n: x.shape[n]
        else:
            norm = tf.linalg.norm
            int64 = tf.int64
            shape = lambda x, n: bk.cast(bk.shape(x)[n], int64)

        if self.num_objects == 1:
            xy = self.xy_absolute_error(y_true, y_pred, 0, 0, backend)  # shape=(n, 2) [[x_abs_err, y_abs_err], [x_abs_err, y_abs_err], ...]
            angle = self.angle_absolute_error_direction_agnostic(
                y_pred[:, self.col2idx(0, 'angle_deg')],
                y_true[:, self.col2idx(0, 'angle_deg')],
                backend, angle_scaler)  # shape=(n, 1)
            mean_errors_xy = norm(xy, axis=1)  # shape=(n,)
            mean_errors_angle = angle  # shape=(n,)
            indices = backend.arange(0, shape(mean_errors_xy, 0))
        elif self.num_objects == 2:
            xy = {}
            angle = {}
            for i, j in ((0, 0), (1, 1), (0, 1), (1, 0)):
                xy[(i, j)] = self.xy_absolute_error(y_true, y_pred, i, j,
                                               bk)  # shape=(n, 2) [[x_abs_err, y_abs_err], [x_abs_err, y_abs_err], ...]
                angle[(i, j)] = self.angle_absolute_error_direction_agnostic(
                    y_pred[:, self.col2idx(i, 'angle_deg')],  # shape=(n,)
                    y_true[:, self.col2idx(j, 'angle_deg')],  # shape=(n,)
                    bk, angle_scaler)  # shape=(n,)
            mean_errors_xy = bk.stack((
                bk.mean(bk.stack((norm(xy[0, 0], axis=1),
                                  norm(xy[1, 1], axis=1))), axis=0),
                bk.mean(bk.stack((norm(xy[0, 1], axis=1),
                                  norm(xy[1, 0], axis=1))), axis=0)))  # shape=(2, n)
            mean_errors_angle = bk.stack((
                bk.mean(bk.stack((angle[0, 0], angle[1, 1]), axis=0), axis=0),
                bk.mean(bk.stack((angle[0, 1], angle[1, 0]), axis=0), axis=0)
            ), axis=0)  # shape=(2, n)

            swap_idx = bk.argmin(mean_errors_xy, axis=0)  # shape = (n,)

            indices = backend.transpose(
                backend.stack((swap_idx, backend.arange(0, shape(mean_errors_xy, 1)))))  # shape=(n, 2)
        else:
            assert False, 'not implemented'

        return mean_errors_xy, mean_errors_angle, indices

    def model(self):
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
        out = Dense(len(COLUMNS) * self.num_objects, kernel_initializer='normal', activation='linear')(out_a)
        return Model(input_shape, out)

    def train(self, model, train_generator, params):
        # adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=lambda x, y: self.interaction_loss_angle(x, y, alpha=params['loss_alpha']), # , (angle_scaler.mean_, angle_scaler.scale_)
                      optimizer='adam')
        # model.lr.set_value(0.05)
        with open(join(params['experiment_dir'], 'model.txt'), 'w') as fw:
            model.summary(print_fn=lambda x: fw.write(x + '\n'))
        csv_logger = CSVLogger(join(params['experiment_dir'], 'log.csv'), append=True, separator=';')
        tb = TensorBoard(log_dir=params['tensorboard_dir'], histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                         write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                         embeddings_metadata=None)
        model.fit_generator(train_generator, steps_per_epoch=params['steps_per_epoch'], epochs=params['epochs'],
                            verbose=1, callbacks=[csv_logger, tb])  # , validation_data=

        model.save_weights(join(params['experiment_dir'], 'weights.h5'))
        return model

    def evaluate(self, model, test_generator, y_test, params):
        pred = model.predict_generator(test_generator, int(len(y_test) / BATCH_SIZE))
        # pred[:, [0, 1, 5, 6]] = xy_scaler.inverse_transform(pred[:, [0, 1, 5, 6]])
        # pred[:, [4, 9]] = angle_scaler.inverse_transform(pred[:, [4, 9]])

        with h5py.File(join(params['experiment_dir'], 'predictions.h5'), 'w') as hf:
            hf.create_dataset("data", data=pred)
        # xy, _, indices = match_pred_to_gt_dxdy(pred, y_test.values, np)
        xy, angle, indices = self.match_pred_to_gt(pred, y_test.values, np)

        if self.num_objects == 1:
            xy_mae = np.take(xy, indices).mean()
            angle_mae = np.take(angle, indices).mean()
        else:
            xy_mae = (xy[indices[:, 0], indices[:, 1]]).mean()
            angle_mae = (angle[indices[:, 0], indices[:, 1]]).mean()

        # # compute angle errors
        # angle = {}
        # for i, j in ((0, 0), (1, 1), (0, 1), (1, 0)):
        #     angle[(i, j)] = angle_absolute_error(
        #         y_test.values[:, NAME2COL[i]['angle_deg']],
        #         np.degrees(np.arctan((pred[:, NAME2COL[j]['dy']] / pred[:, NAME2COL[j]['dx']]))),
        #         np)
        # mean_errors_angle = np.stack((
        #     np.mean(np.stack((angle[0, 0], angle[1, 1]), axis=1), axis=1),
        #     np.mean(np.stack((angle[0, 1], angle[1, 0]), axis=1), axis=1)))  # shape=(2, n)
        # angle_mae = (mean_errors_angle[indices[:, 0], indices[:, 1]]).mean()

        results = pd.DataFrame.from_items([('xy MAE', [xy_mae]), ('angle MAE', angle_mae)])
        print(results.to_string(index=False))
        results.to_csv(join(params['experiment_dir'], 'results.csv'))
        return results

    def evaluate_model(self, data_dir, model_dir, n_objects=2):
        self.num_objects = n_objects
        hf = h5py.File(join(data_dir, 'images.h5'), 'r')
        X_test = hf['test']

        y_test_df = pd.read_csv(join(data_dir, 'test.csv'))
        # convert to counter-clockwise
        # for i in range(self.num_objects):
        #     y_test_df.loc[:, '%d_angle_deg' % i] *= -1
        y_test = y_test_df[self.columns()]

        # size = 200
        # x, y = np.mgrid[0:size, 0:size]
        # mask = np.expand_dims(np.exp(- 0.0002 * ((x - size / 2) ** 2 + (y - size / 2) ** 2)), 2)
        #
        # def image_dim(img):
        #     return img * mask

        test_datagen = ImageDataGenerator(rescale=1./255) # , preprocessing_function=rotate90)
        test_generator = test_datagen.flow(X_test, shuffle=False)

        base_experiment_name = time.strftime("%y%m%d_%H%M", time.localtime())
        base_experiment_dir = ROOT_EXPERIMENT_DIR + base_experiment_name
        base_tensor_board_dir = join(ROOT_TENSOR_BOARD_DIR, base_experiment_name)

        if not os.path.exists(base_experiment_dir):
            os.mkdir(base_experiment_dir)

        with file(join(base_experiment_dir, 'parameters.txt'), 'w') as fw:
            fw.writelines('\n'.join(sys.argv))

        if os.path.exists(join(model_dir, 'model.yaml')):
            with open(join(model_dir, 'model.yaml'), 'r') as fr:
                m = model_from_yaml(fr.read())
        elif os.path.exists(join(model_dir, 'model.json')):
            with open(join(model_dir, 'model.json'), 'r') as fr:
                m = model_from_json(fr.read())
        else:
            m = self.model()
            warnings.warn('Stored model not found, initializing model using model().')
        m.load_weights(join(model_dir, 'weights.h5'))

        parameters = {}
        parameters['experiment_dir'] = base_experiment_dir
        parameters['tensorboard_dir'] = base_tensor_board_dir
        self.evaluate(m, test_generator, y_test, parameters)
        hf.close()

    def train_and_evaluate(self, data_dir, loss_alpha, n_epochs=10, n_objects=2):
        self.num_objects = n_objects
        hf = h5py.File(join(data_dir, 'images.h5'), 'r')
        X_train = hf['train']
        X_test = hf['test']
        y_train_df = pd.read_csv(join(data_dir, 'train.csv'))
        y_test_df = pd.read_csv(join(data_dir, 'test.csv'))

        # convert to counter-clockwise
        for i in range(self.num_objects):
            y_train_df.loc[:, '%d_angle_deg' % i] *= -1
            y_test_df.loc[:, '%d_angle_deg' % i] *= -1

        # input image and gt rotation
        tregion = TransformableRegion(X_test[0])
        tregion.rotate(90, np.array(tregion.img.shape[:2]) / 2)
        for i in range(ti.num_objects):
            y_train_df[['%d_x' % i, '%d_y' % i]] = tregion.get_transformed_coords(
                y_train_df[['%d_x' % i, '%d_y' % i]].values.T).T
            y_train_df['%d_angle_deg' % i] = tregion.get_transformed_angle(y_train_df['%d_angle_deg' % i])

            y_test_df[['%d_x' % i, '%d_y' % i]] = tregion.get_transformed_coords(
                y_test_df[['%d_x' % i, '%d_y' % i]].values.T).T
            y_test_df['%d_angle_deg' % i] = tregion.get_transformed_angle(y_test_df['%d_angle_deg' % i])

        # dx and dy columns
        # for i in range(self.num_objects):
        #     # y_train_df.loc[:, '%d_angle_deg' % i] += 90
        #     # y_train_df.loc[:, '%d_angle_deg' % i] %= 360
        #
        #     angle_rad = np.radians(y_train_df['%d_angle_deg' % i])
        #     y_train_df.loc[:, '%d_dx' % i] = y_train_df['%d_major' % i] * np.cos(angle_rad)
        #     y_train_df.loc[:, '%d_dy' % i] = y_train_df['%d_major' % i] * np.sin(angle_rad)
        #
        #     # y_test_df.loc[:, '%d_angle_deg' % i] += 90
        #     # y_test_df.loc[:, '%d_angle_deg' % i] %= 360
        #
        #     angle_rad = np.radians(y_test_df['%d_angle_deg' % i])
        #     y_test_df.loc[:, '%d_dx' % i] = y_test_df['%d_major' % i] * np.cos(angle_rad)
        #     y_test_df.loc[:, '%d_dy' % i] = y_test_df['%d_major' % i] * np.sin(angle_rad)

        y_train = y_train_df[self.columns()]
        y_test = y_test_df[self.columns()]

        print X_train.shape, X_test.shape, y_train.shape, y_test.shape

        parameters = {'epochs': n_epochs,
                      'steps_per_epoch': int(len(X_train) / BATCH_SIZE)
                      }
        if isinstance(loss_alpha, str) and loss_alpha == 'batch':
            parameters['loss_alpha'] = np.linspace(0, 1, 15)
        else:
            parameters['loss_alpha'] = float(loss_alpha)

        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # # # rescale data
        # xy_scaler = StandardScaler()
        # xy_scaler.mean_ = 0  # 100
        # xy_scaler.scale_ = 1  # 100
        # # # y_train = xy_scaler.transform(y_train)
        # # # y_test = xy_scaler.transform(y_test)
        # # y_train[:, [0, 1, 5, 6]] = xy_scaler.transform(y_train[:, [0, 1, 5, 6]])
        # angle_scaler = StandardScaler()
        # angle_scaler.mean_ = 0  # 180
        # angle_scaler.scale_ = 1  # 180
        # # y_train[:, [4, 9]] = angle_scaler.transform(y_train[:, [4, 9]])

        def rotate90(img):
            tregion.set_img(img)
            return tregion.get_img()
            # out_img = skimage.transform.rotate(img, 90, preserve_range=True)
            # return out_img

        size = 200
        x, y = np.mgrid[0:size, 0:size]
        mask = np.expand_dims(np.exp(- 0.0002 * ((x - size / 2) ** 2 + (y - size / 2) ** 2)), 2)

        def image_dim(img):
            return img * mask

        train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=rotate90)
        train_generator = train_datagen.flow(X_train, y_train)
        test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=rotate90)
        test_generator = test_datagen.flow(X_test, shuffle=False)

        base_experiment_name = time.strftime("%y%m%d_%H%M", time.localtime())
        base_experiment_dir = ROOT_EXPERIMENT_DIR + base_experiment_name
        base_tensor_board_dir = join(ROOT_TENSOR_BOARD_DIR, base_experiment_name)

        if not os.path.exists(base_experiment_dir):
            os.mkdir(base_experiment_dir)

        with file(join(base_experiment_dir, 'parameters.txt'), 'w') as fw:
            fw.writelines('\n'.join(sys.argv))

        results = pd.DataFrame()

        if not isinstance(parameters['loss_alpha'], numbers.Number):
            for alpha in parameters['loss_alpha']:
                m = self.model()
                print('loss_alpha %f' % alpha)
                experiment_dir = join(base_experiment_dir, str(alpha))
                if not os.path.exists(experiment_dir):
                    os.mkdir(experiment_dir)

                parameters['loss_alpha'] = alpha
                parameters['experiment_dir'] = experiment_dir
                parameters['tensorboard_dir'] = join(base_tensor_board_dir, str(alpha))
                m = self.train(m, train_generator, parameters)
                with open(join(experiment_dir, 'model.yaml'), 'w') as fw:
                    fw.write(m.to_yaml())
                results_ = self.evaluate(m, test_generator, y_test, parameters)
                results_['loss_alpha'] = alpha
                results = results.append(results_, ignore_index=True)

            print(results.to_string(index=False))
            results.to_csv(join(base_experiment_dir, 'results.csv'))
        else:
            m = self.model()
            parameters['experiment_dir'] = base_experiment_dir
            parameters['tensorboard_dir'] = base_tensor_board_dir
            m = self.train(m, train_generator, parameters)
            # m.load_weights(join(experiment_dir, 'weights.h5'))
            results = self.evaluate(m, test_generator, y_test, parameters)

        hf.close()


if __name__ == '__main__':
    ti = TrainInteractions()
    fire.Fire({
      'train': ti.train_and_evaluate,
      'evaluate': ti.evaluate_model,
    })
