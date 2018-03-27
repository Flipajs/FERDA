from __future__ import print_function
import os
import h5py
import sys
import numpy as np
import time
from os.path import join
import numbers
import pandas as pd
try:
    from keras.utils import np_utils
    from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, GlobalAveragePooling2D
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from keras import backend as K
    from keras.callbacks import CSVLogger, TensorBoard, Callback, ModelCheckpoint
    import tensorflow as tf
    import keras
    from keras.models import model_from_yaml, model_from_json
    from keras.preprocessing.image import ImageDataGenerator
except ImportError as e:
    print('Warning, no keras installed: {}'.format(e))
import fire
from core.region.transformableregion import TransformableRegion
from utils.angles import angle_absolute_error, angle_absolute_error_direction_agnostic
import warnings
import yaml
import re
from utils.objectsarray import ObjectsArray
# import skimage.transform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


ROOT_EXPERIMENT_DIR = '/datagrid/personal/smidm1/ferda/interactions/experiments/'
ROOT_TENSOR_BOARD_DIR = '/datagrid/personal/smidm1/ferda/interactions/tb_logs'
BATCH_SIZE = 32


class ValidationCallback(Callback):
    def __init__(self, test_generator, evaluate_fun):
        self.test_generator = test_generator
        self.evaluate_fun = evaluate_fun

    def on_epoch_end(self, epoch, logs={}):
        self.evaluate_fun(self.model, self.test_generator)


class TrainInteractions:
    def __init__(self, num_objects=None):
        self.models = {
            '6conv_3dense': self.model_6conv_3dense,  # former default
            '6conv_2dense': self.model_6conv_2dense,
            'mobilenet': self.model_mobilenet,
        }
        self.PREDICTED_PROPERTIES = ['x', 'y', 'angle_deg']
        self.DETECTOR_INPUT_SIZE_PX = 200
        self.IMAGE_LAYERS = 1
        self.num_objects = None
        self.array = None  # provide column name to index mapping using ObjectsArray
        if num_objects is not None:
            self.set_num_objects(num_objects)  # init num_objects and map

    def set_num_objects(self, n):
        self.num_objects = n
        self.array = ObjectsArray(self.PREDICTED_PROPERTIES, n)

    @staticmethod
    def toarray(struct_array):
        types = [x[1] for x in struct_array.dtype.descr]
        all(x == types[0] for x in types)
        return struct_array.view(types[0]).reshape(struct_array.shape + (-1,))

    def read_gt(self, filename):
        regexp = re.compile('(\d*)_(\w*)')
        df = pd.read_csv(filename)
        ids = set()
        properties = []
        for col in df.columns:
            match = regexp.match(col)
            if match is not None:
                ids.add(int(match.group(1)))
                properties.append(match.group(2))
        n = len(ids)
        assert min(ids) == 0 and max(ids) == n - 1, 'object describing columns have to be prefixed with numbers starting with 0'
        assert len(properties) % n == 0
        properties = properties[:(len(properties) / n) - 1]  # only properties for object 0
        return n, properties, df

    def xy_absolute_error(self, y_true, y_pred, i, j, backend):
        return backend.abs(backend.concatenate(
            (y_pred[:, self.array.prop2idx_(i, 'x')] - y_true[:, self.array.prop2idx_(j, 'x')],
             y_pred[:, self.array.prop2idx_(i, 'y')] - y_true[:, self.array.prop2idx_(j, 'y')]),
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
        dx = y_true[:, self.array.prop2idx_(i, 'dx')] - y_pred[:, self.array.prop2idx_(j, 'dx')]
        dy = y_true[:, self.array.prop2idx_(i, 'dy')] - y_pred[:, self.array.prop2idx_(j, 'dy')]
        return backend.concatenate((backend.abs(dx), backend.abs(dy)), axis=1)

    def interaction_loss_angle(self, y_true, y_pred, alpha=0.5):
        assert 0 <= alpha <= 1
        # y_pred = (y_pred - 0.5) * 2  # softmax -> tanh range
        # following expects tanh output (-1; 1)
        tensor_columns = []
        for i in range(self.num_objects):
            for col in self.array.properties:
                if col != 'angle_deg':
                    tensor_columns.append((y_pred[:, self.array.prop2idx(i, col)] + 1) * self.DETECTOR_INPUT_SIZE_PX / 2)
                else:
                    tensor_columns.append((y_pred[:, self.array.prop2idx(i, 'angle_deg')] * np.pi / 2) / np.pi * 180)
        y_pred = K.stack(tensor_columns, axis=1)

        mean_errors_xy, mean_errors_angle, indices = self.match_pred_to_gt(y_true, y_pred, K)
        if self.num_objects == 2:
            errors_xy = tf.gather_nd(mean_errors_xy, indices)
            errors_angle = tf.gather_nd(mean_errors_angle, indices)
        elif self.num_objects == 1:
            errors_xy = mean_errors_xy
            errors_angle = mean_errors_angle
        else:
            assert False, 'not implemented'
        return K.mean(errors_xy * (1 - alpha) + errors_angle * alpha)

    def interaction_loss_dxdy(self, y_true, y_pred, alpha=0.5):
        assert 0 <= alpha <= 1
        mean_errors_xy, mean_errors_delta, indices = self.match_pred_to_gt_dxdy(y_true, y_pred, K)

        return K.mean(tf.gather_nd(mean_errors_xy, indices) * (1 - alpha) +
                      tf.gather_nd(mean_errors_delta, indices) * alpha)

    def match_pred_to_gt_dxdy(self, y_true, y_pred, backend):
        """
        Return mean absolute errors for individual samples for xy and theta
        in two possible combinations of prediction and ground truth.
        """
        assert False, 'outdated'
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

    def match_pred_to_gt(self, y_true, y_pred, backend):
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
            angle = angle_absolute_error_direction_agnostic(
                y_pred[:, self.array.prop2idx(0, 'angle_deg')],
                y_true[:, self.array.prop2idx(0, 'angle_deg')],
                backend)  # shape=(n, 1)
            mean_errors_xy = norm(xy, axis=1)  # shape=(n,)
            mean_errors_angle = angle  # shape=(n,)
            indices = backend.arange(0, shape(mean_errors_xy, 0))
        elif self.num_objects == 2:
            xy = {}
            angle = {}
            for i, j in ((0, 0), (1, 1), (0, 1), (1, 0)):
                xy[(i, j)] = self.xy_absolute_error(y_true, y_pred, i, j,
                                               bk)  # shape=(n, 2) [[x_abs_err, y_abs_err], [x_abs_err, y_abs_err], ...]
                angle[(i, j)] = angle_absolute_error_direction_agnostic(
                    y_pred[:, self.array.prop2idx(i, 'angle_deg')],  # shape=(n,)
                    y_true[:, self.array.prop2idx(j, 'angle_deg')],  # shape=(n,)
                    bk)  # shape=(n,)
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

    def model_6conv_3dense(self):
        input_shape = Input(shape=(self.DETECTOR_INPUT_SIZE_PX, self.DETECTOR_INPUT_SIZE_PX, self.IMAGE_LAYERS))
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_shape)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        x = Conv2D(32, (3, 3))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(16, (3, 3))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(self.array.num_columns(), kernel_initializer='normal', activation='tanh')(x)
        return Model(input_shape, out)

    def model_6conv_3dense_legacy(self):
        input_shape = Input(shape=(200, 200, 3))
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_shape)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        x = Conv2D(32, (3, 3))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(16, (3, 3))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(10, kernel_initializer='normal', activation='tanh')(x)
        return Model(input_shape, out)

    def model_6conv_2dense(self):
        input_shape = Input(shape=(self.DETECTOR_INPUT_SIZE_PX, self.DETECTOR_INPUT_SIZE_PX, self.IMAGE_LAYERS))
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_shape)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        x = Conv2D(32, (3, 3))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(16, (3, 3))(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        out = Dense(self.array.num_columns(), kernel_initializer='normal', activation='tanh')(x)
        return Model(input_shape, out)

    def model_mobilenet(self):
        base_model = keras.applications.mobilenet.MobileNet(
            (self.DETECTOR_INPUT_SIZE_PX, self.DETECTOR_INPUT_SIZE_PX, self.IMAGE_LAYERS),
            include_top=False, weights=None)  # weights='imagenet'
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.array.num_columns(), activation='tanh')(x)  # softmax
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        # for layer in base_model.layers:
        #     layer.trainable = False
        return model

    def train(self, model, train_generator, test_generator, params, callbacks=[]):
        # adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=lambda x, y: self.interaction_loss_angle(x, y, alpha=params['loss_alpha']),
                      optimizer='adam')
        # model.lr.set_value(0.05)
        with open(join(params['experiment_dir'], 'model.txt'), 'w') as fw:
            model.summary(print_fn=lambda x: fw.write(x + '\n'))
        csv_logger = CSVLogger(join(params['experiment_dir'], 'log.csv'), append=True, separator=';')
        tb = TensorBoard(log_dir=params['tensorboard_dir'], histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                         write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                         embeddings_metadata=None)
        checkpoint = ModelCheckpoint(join(params['experiment_dir'], 'weights.h5'), save_best_only=True)
        model.fit_generator(train_generator, steps_per_epoch=params['steps_per_epoch'], epochs=params['epochs'],
                            verbose=1, callbacks=[csv_logger, tb, checkpoint] + callbacks,
                            validation_data=test_generator, validation_steps=params['n_test'])
        # model.save_weights(join(params['experiment_dir'], 'weights.h5'))
        return model

    def evaluate(self, model, test_generator, params, y_test=None, csv_filename=None):
        pred = model.predict_generator(test_generator, int(params['n_test'] / BATCH_SIZE))
        assert pred is not None and pred is not []

        # with h5py.File(join(params['experiment_dir'], 'predictions.h5'), 'w') as hf:
        #     hf.create_dataset("data", data=pred)

        # pred = (pred - 0.5) * 2  # softmax -> tanh range
        # following expects tanh output (-1; 1)
        pred = self.postprocess_predictions(pred)

        pred_df = self.array.array_to_dataframe(pred)
        pred_df.to_csv(join(params['experiment_dir'], 'predictions.csv'), index=False)
        self.save_model_properties(join(params['experiment_dir'], 'config.yaml'))

        if y_test is not None:
            # xy, _, indices = match_pred_to_gt_dxdy(y_test.values, pred, np)
            xy, angle, indices = self.match_pred_to_gt(y_test[:len(pred)], pred, np)  # trim y_test to be modulo BATCH_SIZE

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
            if csv_filename is not None:
                results.to_csv(csv_filename)
            return results

    def postprocess_predictions(self, pred):
        for i in range(self.num_objects):
            pred[:, self.array.prop2idx(i, 'angle_deg')] = \
                np.degrees(pred[:, self.array.prop2idx(i, 'angle_deg')] * np.pi / 2)
            pred[:, self.array.prop2idx(i, 'x')] = \
                (pred[:, self.array.prop2idx(i, 'x')] + 1) * self.DETECTOR_INPUT_SIZE_PX / 2
            pred[:, self.array.prop2idx(i, 'y')] = \
                (pred[:, self.array.prop2idx(i, 'y')] + 1) * self.DETECTOR_INPUT_SIZE_PX / 2
        return pred

    def save_model_properties(self, out_yaml):
        with open(out_yaml, 'w') as fw:
            yaml.dump({
                'num_objects': self.num_objects,
                'properties': self.PREDICTED_PROPERTIES,
                'input_size_px': self.DETECTOR_INPUT_SIZE_PX,
            }, fw)

    def evaluate_model(self, data_dir, model_dir):
        # load model
        if os.path.exists(join(model_dir, 'model.yaml')):
            with open(join(model_dir, 'model.yaml'), 'r') as fr:
                m = model_from_yaml(fr.read())
        elif os.path.exists(join(model_dir, 'model.json')):
            with open(join(model_dir, 'model.json'), 'r') as fr:
                m = model_from_json(fr.read())
        else:
            m = self.model_6conv_3dense()
            warnings.warn('Stored model not found, initializing model using model_6conv_3dense().')
        m.load_weights(join(model_dir, 'weights.h5'))
        with open(join(model_dir, 'config.yaml'), 'r') as fr:
            model_metadata = yaml.load(fr)
        self.PREDICTED_PROPERTIES = model_metadata['properties']
        self.DETECTOR_INPUT_SIZE_PX = model_metadata['input_size_px']
        self.array = ObjectsArray(self.PREDICTED_PROPERTIES, model_metadata['num_objects'])

        # load images
        hf = h5py.File(join(data_dir, 'images.h5'), 'r')
        X_test = hf['test']

        # load gt
        gt_filename = join(data_dir, 'test.csv')
        if os.path.exists(gt_filename):
            gt_n, gt_columns, y_test_df = self.read_gt(gt_filename)
            # convert to counter-clockwise
            for i in range(gt_n):
                y_test_df.loc[:, '%d_angle_deg' % i] *= -1
            assert model_metadata['num_objects'] == gt_n, 'number of predicted objects and number of objects in gt has to agree'
            y_test = self.array.dataframe_to_array(y_test_df)
        else:
            warnings.warn('Ground truth file test.csv not found. Generating predictions without evaluation.')
            y_test = None

        self.set_num_objects(self.array.n)
        # size = 200
        # x, y = np.mgrid[0:size, 0:size]
        # mask = np.expand_dims(np.exp(- 0.0002 * ((x - size / 2) ** 2 + (y - size / 2) ** 2)), 2)
        #
        # def image_dim(img):
        #     return img * mask

        test_datagen = ImageDataGenerator(rescale=1./255)  # , preprocessing_function=rotate90)
        test_generator = test_datagen.flow(X_test, shuffle=False)

        base_experiment_name = time.strftime("%y%m%d_%H%M", time.localtime())
        base_experiment_dir = ROOT_EXPERIMENT_DIR + base_experiment_name
        base_tensor_board_dir = join(ROOT_TENSOR_BOARD_DIR, base_experiment_name)

        if not os.path.exists(base_experiment_dir):
            os.mkdir(base_experiment_dir)
        self._write_argv(base_experiment_dir)

        parameters = {'experiment_dir': base_experiment_dir,
                      'tensorboard_dir': base_tensor_board_dir,
                      'n_test': len(X_test)
                      }
        results = self.evaluate(m, test_generator, parameters, y_test)
        print(results.to_string(index=False))
        hf.close()

    def resize_images(self, img_batch, shape):
        img_shape = img_batch[0].shape
        assert img_shape[0] <= shape[0] and img_shape[1] <= shape[1]
        out = np.zeros(shape=((len(img_batch), ) + shape), dtype=img_batch[0].dtype)
        for i, img in enumerate(img_batch):
            out[i, :img_shape[0], :img_shape[1]] = img  # np.expand_dims(img, 2)
        return out

    def train_and_evaluate(self, data_dir, loss_alpha, n_epochs=10, rotate=False, exp_name='',
                           model='6conv_3dense', dataset_names=None):
        # example:
        # local: train /home/matej/prace/ferda/data/interactions/1712_1k_36rot/ 0.5 100 --exp-name=two_mobilenet_scratch
        # remote: train /mnt/home.stud/smidm/datagrid/ferda/interactions/1712_1k_36rot_fixed/ 0.5 100 --exp-name=two_mobilenet_scratch
        if dataset_names is None:
            dataset_names = {'train': 'train', 'test': 'test'}
        # load images
        hf = h5py.File(join(data_dir, 'images.h5'), 'r')
        X_train = hf[dataset_names['train']]
        X_test = hf[dataset_names['test']]
        # if model == 'mobilenet':
        #     X_train = self.resize_images(hf[dataset_names['train']], (224, 224, 1))
        #     X_test = self.resize_images(hf[dataset_names['test']], (224, 224, 1))

        # load gt
        n_train, columns_train, y_train_df = self.read_gt(join(data_dir, 'train.csv'))
        n_test, columns_test, y_test_df = self.read_gt(join(data_dir, 'test.csv'))
        assert n_train == n_test
        assert columns_train == columns_test
        self.set_num_objects(n_train)

        # convert to counter-clockwise
        for i in range(self.num_objects):
            y_train_df.loc[:, '%d_angle_deg' % i] *= -1
            y_test_df.loc[:, '%d_angle_deg' % i] *= -1

        # input image and gt rotation
        if rotate:
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

        y_train = self.array.dataframe_to_array(y_train_df)
        y_test = self.array.dataframe_to_array(y_test_df)

        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        assert model in self.models, 'model {} doesn\'t exist'.format(model)

        parameters = {'epochs': n_epochs,
                      'steps_per_epoch': int(len(X_train) / BATCH_SIZE),  # 1
                      'n_test': len(X_test),
        }
        if isinstance(loss_alpha, str) and loss_alpha == 'batch':
            parameters['loss_alpha'] = np.linspace(0, 1, 8)
        else:
            parameters['loss_alpha'] = float(loss_alpha)

        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        def rotate90(img):
            tregion.set_img(img)
            return tregion.get_img()
            # out_img = skimage.transform.rotate(img, 90, preserve_range=True)
            # return out_img

        size = self.DETECTOR_INPUT_SIZE_PX
        x, y = np.mgrid[0:size, 0:size]
        mask = np.expand_dims(np.exp(- 0.0002 * ((x - size / 2) ** 2 + (y - size / 2) ** 2)), 2)

        def image_dim(img):
            return img * mask

        if rotate:
            preprocessing = rotate90
        else:
            preprocessing = None
        if self.IMAGE_LAYERS == 1:
            X_train = np.expand_dims(X_train, 3)
            X_test = np.expand_dims(X_test, 3)
        train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocessing)
        train_generator = train_datagen.flow(X_train, y_train)
        test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocessing)
        test_generator = test_datagen.flow(X_test, y_test, shuffle=False)

        def eval_and_print(_m, _t):
            results = self.evaluate(_m, _t, parameters, y_test)
            print('\n' + results.to_string(index=False) + '\n')

        val_callback = ValidationCallback(test_generator, eval_and_print)

        base_experiment_name = time.strftime("%y%m%d_%H%M", time.localtime()) + '_' + exp_name
        base_experiment_dir = join(ROOT_EXPERIMENT_DIR, base_experiment_name)
        base_tensor_board_dir = join(ROOT_TENSOR_BOARD_DIR, base_experiment_name)

        if not os.path.exists(base_experiment_dir):
            os.mkdir(base_experiment_dir)

        self._write_argv(base_experiment_dir)

        results = pd.DataFrame()

        if not isinstance(parameters['loss_alpha'], numbers.Number):
            for alpha in parameters['loss_alpha']:
                m = self.models[model]()
                print('loss_alpha %f' % alpha)
                experiment_dir = join(base_experiment_dir, str(alpha))
                if not os.path.exists(experiment_dir):
                    os.mkdir(experiment_dir)

                parameters['loss_alpha'] = alpha
                parameters['experiment_dir'] = experiment_dir
                parameters['tensorboard_dir'] = join(base_tensor_board_dir, str(alpha))
                m = self.train(m, train_generator, test_generator, parameters, callbacks=[val_callback])
                with open(join(experiment_dir, 'model.yaml'), 'w') as fw:
                    fw.write(m.to_yaml())
                results_ = self.evaluate(m, test_generator, parameters, y_test,
                                         csv_filename=join(parameters['experiment_dir'], 'results.csv'))
                results_['loss_alpha'] = alpha
                results = results.append(results_, ignore_index=True)

            print(results.to_string(index=False))
            results.to_csv(join(base_experiment_dir, 'results.csv'))
        else:
            m = self.models[model]()
            parameters['experiment_dir'] = base_experiment_dir
            parameters['tensorboard_dir'] = base_tensor_board_dir
            m = self.train(m, train_generator, test_generator, parameters, callbacks=[val_callback])
            with open(join(parameters['experiment_dir'], 'model.yaml'), 'w') as fw:
                fw.write(m.to_yaml())
            results = self.evaluate(m, test_generator, parameters, y_test,
                          csv_filename=join(parameters['experiment_dir'], 'results.csv'))

        hf.close()

    def _write_argv(self, out_dir):
        with open(join(out_dir, 'parameters.txt'), 'w') as fw:
            fw.writelines('\n'.join(sys.argv))


if __name__ == '__main__':
    ti = TrainInteractions()
    fire.Fire({
      'train': ti.train_and_evaluate,
      'predict': ti.evaluate_model,
    })
