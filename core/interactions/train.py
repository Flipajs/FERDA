from __future__ import print_function
import os
import h5py
import sys
import numpy as np
import time
from os.path import join
import numbers
import pandas as pd
from collections import OrderedDict
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
    from keras.applications.mobilenet import mobilenet
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
    def __init__(self, num_objects=None, num_input_layers=3, predicted_properties=None, error_functions=None,
                 detector_input_size_px=200):
        assert (predicted_properties is None and error_functions is None) or \
               len(predicted_properties) == len(error_functions)
        self.models = {
            '6conv_3dense': self.model_6conv_3dense,  # former default
            '6conv_2dense': self.model_6conv_2dense,
            'mobilenet': self.model_mobilenet,
            'single_mobilenet': self.model_single_mobilenet,
        }
        if predicted_properties is None:
            self.PREDICTED_PROPERTIES = ['x', 'y', 'angle_deg']  # , 'major', 'minor']
        else:
            self.PREDICTED_PROPERTIES = predicted_properties
        if error_functions is None:
            self.ERROR_FUNCTIONS = ['abs', 'abs', 'angle_180']  # , 'abs', 'abs']
        else:
            self.ERROR_FUNCTIONS = error_functions
        self.detector_input_size_px = detector_input_size_px
        self.num_input_layers = num_input_layers
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

    def xy_absolute_error(self, y_true, y_pred, i, j, backend):
        return backend.abs(backend.concatenate(
            (y_pred[:, self.array.prop2idx_(i, 'x')] - y_true[:, self.array.prop2idx_(j, 'x')],
             y_pred[:, self.array.prop2idx_(i, 'y')] - y_true[:, self.array.prop2idx_(j, 'y')]),
            axis=1))

    def axes_absolute_error(self, y_true, y_pred, i, j, backend):
        return backend.abs(backend.concatenate(
            (y_pred[:, self.array.prop2idx_(i, 'major')] - y_true[:, self.array.prop2idx_(j, 'major')],
             y_pred[:, self.array.prop2idx_(i, 'minor')] - y_true[:, self.array.prop2idx_(j, 'minor')]),
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
        """
        Compute prediction loss with respect to true xy and angle of all detected objects.

        :param y_true: ground truth for all objects; shape=(n_samples, n_objects * len(PREDICTED_PROPERTIES))
        :param y_pred: predictions for all objects; shape=(n_samples, n_objects * len(PREDICTED_PROPERTIES))
                       all predictions in tanh range (-1; 1)
        :param alpha: float; xy error and angle error balancing weight, 0 means only xy error is considered,
                      1 only angle error is considered
        :return: float; scalar loss
        """
        assert 0 <= alpha <= 1
        # y_pred = (y_pred - 0.5) * 2  # softmax -> tanh range
        # following expects tanh output (-1; 1)
        # TODO: replace with generalized postprocess_predictions()
        tensor_columns = []
        for i in range(self.num_objects):
            for col in self.array.properties:
                if col != 'angle_deg':
                    # (-1; 1) -> (0; detector_input_size_px)
                    tensor_columns.append((y_pred[:, self.array.prop2idx(i, col)] + 1) * self.detector_input_size_px / 2)
                else:
                    # (-1; 1) -> (-90; 90)
                    tensor_columns.append(y_pred[:, self.array.prop2idx(i, 'angle_deg')] * 90)
        y_pred = K.stack(tensor_columns, axis=1)

        errors, errors_xy, _ = self.match_pred_to_gt(y_true, y_pred)

        loss = 0
        for i in range(self.num_objects):
            error_angle = errors[:, self.array.prop2idx(i, 'angle_deg')]  # shape=(n,)
            error_xy = errors_xy[:, i]  # shape=(n,)
            loss += K.sum(error_xy * (1 - alpha) + error_angle * alpha)

        return loss

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

    def errors_ij(self, y_true, y_pred, j_true, i_pred, backend=np):
        """
        Compute error of ith predicted object with respect to jth ground truth object.

        Uses self.ERROR_FUNCTIONS corresponding to self.PREDICTED_PROPERTIES.

        :param y_true: ground truth for all objects; shape=(n_samples, len(PREDICTED_PROPERTIES))
        :param y_pred: predictions for all objects; shape=(n_samples, len(PREDICTED_PROPERTIES))
        :param j_true: ground truth object index
        :param i_pred: predicted object index
        :param backend: backend to use (numpy or keras backend)
        :return: tensor or array, shape=(n_samples, len(PREDICTED_PROPERTIES))
        """
        errors_columns = []
        for i, (prop, error_fun) in enumerate(zip(self.PREDICTED_PROPERTIES, self.ERROR_FUNCTIONS)):
            true = y_true[:, self.array.prop2idx(j_true, prop)]  # shape=(n,)
            pred = y_pred[:, self.array.prop2idx(i_pred, prop)]  # shape=(n,)
            if error_fun == 'abs':
                errors_columns.append(backend.abs(pred - true))
            elif error_fun == 'angle_180':
                errors_columns.append(angle_absolute_error_direction_agnostic(pred, true, backend))
            elif error_fun == 'angle_360':
                errors_columns.append(angle_absolute_error(pred, true, backend))
            else:
                assert False, 'error function ' + error_fun + ' not implemented'
        return backend.stack(errors_columns, axis=1)

    def match_pred_to_gt_numpy(self, y_true, y_pred):
        """
        Return errors for individual samples and indices to matching predicted and ground truth values.
        """
        if self.num_objects == 1:
            possible_matchings = (((0, 0),),
                                  )
        elif self.num_objects == 2:
            possible_matchings = (((0, 0), (1, 1)),
                                  ((0, 1), (1, 0)))
        else:
            assert False, 'not implemented'

        sum_xy_errors = []
        all_matching_errors = []
        all_xy_euclidean_errors = []
        for matching in possible_matchings:
            matching_errors = [self.errors_ij(y_true, y_pred, i, j, np) for i, j in matching]
            matching_errors_array = np.concatenate(matching_errors, axis=1)  # shape=(n, num_objects * len(self.PREDICTED_PROPERTIES))
            all_matching_errors.append(matching_errors_array)

            xy_euclidean_errors_ = [np.linalg.norm(matching_errors_array[:, self.array.prop2idx(i, ['x', 'y'])],
                                                   axis=1, keepdims=True)
                                    for i in range(self.num_objects)]
            xy_euclidean_errors = np.concatenate(xy_euclidean_errors_, axis=1)  # shape=(n, num_objects)
            sum_xy_errors.append(np.sum(xy_euclidean_errors, axis=1, keepdims=True))  # shape=(n, 1)
            all_xy_euclidean_errors.append(xy_euclidean_errors)  # list of array like, shape=(n, num_objects)

        swap_indices = np.argmin(np.concatenate(sum_xy_errors, axis=1), axis=1)  # shape = (n,)
        errors = np.array([all_matching_errors[i_swap][i_row] for i_row, i_swap in enumerate(swap_indices)])
        errors_xy = np.array([all_xy_euclidean_errors[i_swap][i_row] for i_row, i_swap in enumerate(swap_indices)])  # shape=(n, num_objects)
        return errors, errors_xy, swap_indices

    def match_pred_to_gt(self, y_true, y_pred):
        """
        Return errors of predicted properties and indices to matching predicted and ground truth values.

        :param y_true: shape=(n_samples, n_dims)
        :param y_pred: shape=(n_samples, n_dims)
        :return errors: errors of individual predicted properties, shape=(n, n_objects * len(PREDICTED_PROPERTIES))
                errors_xy: distance errors, shape=(n,)
                swap_indices: index of shape = (n,)
        """
        norm = tf.linalg.norm
        if self.num_objects == 1:
            possible_matchings = (((0, 0),),
                                  )
        elif self.num_objects == 2:
            possible_matchings = (((0, 0), (1, 1)),
                                  ((0, 1), (1, 0)))
        else:
            assert False, 'not implemented'

        sum_xy_errors = []
        all_matching_errors = []
        all_xy_euclidean_errors = []
        for matching in possible_matchings:
            matching_errors = [self.errors_ij(y_true, y_pred, i, j, K) for i, j in matching]
            matching_errors_array = K.concatenate(matching_errors, axis=1)  # shape=(n, n_objects * len(PREDICTED_PROPERTIES))
            all_matching_errors.append(matching_errors_array)

            # xy_euclidean_errors_ = [norm(matching_errors_array[:, self.array.prop2idx(i, ['x', 'y'])],
            #                              axis=1, keepdims=True)
            #                         for i in range(self.num_objects)]
            xy_euclidean_errors_ = []
            for i in range(self.num_objects):
                xy = K.concatenate([K.expand_dims(matching_errors_array[:, j]) for j in
                                    self.array.prop2idx(i, ['x', 'y'])], axis=1)  # shape=(n, 2)
                xy_euclidean_errors_.append(K.expand_dims(norm(xy, axis=1)))  # shape=(n, 1)
            xy_euclidean_errors = K.concatenate(xy_euclidean_errors_, axis=1)  # shape=(n, num_objects)
            sum_xy_errors.append(K.sum(xy_euclidean_errors, axis=1, keepdims=True))  # shape=(n, 1)
            all_xy_euclidean_errors.append(xy_euclidean_errors)  # list of array like, shape=(n, num_objects)

        swap_indices = K.argmin(K.concatenate(sum_xy_errors, axis=1), axis=1)  # shape = (n,)
        n = K.shape(swap_indices)[0]
        indices_gather = K.transpose(K.stack((swap_indices, K.arange(0, n, dtype=K.dtype(swap_indices)))))  # shape=(n, 2)
        errors = tf.gather_nd(K.stack(all_matching_errors), indices_gather)  # shape=(n, n_objects * len(PREDICTED_PROPERTIES))
        errors_xy = tf.gather_nd(K.stack(all_xy_euclidean_errors), indices_gather)  # shape=(n, num_objects)

        # errors = K.stack([all_matching_errors[i_swap][i_row]
        #                   for i_row, i_swap in enumerate(K.eval(swap_indices))], axis=0)
        # errors_xy = K.stack([all_xy_euclidean_errors[i_swap][i_row]
        #                   for i_row, i_swap in enumerate(K.eval(swap_indices))], axis=0)

        return errors, errors_xy, swap_indices

    def model_6conv_3dense(self):
        input_shape = Input(shape=(self.detector_input_size_px, self.detector_input_size_px, self.num_input_layers))
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
        input_shape = Input(shape=(self.detector_input_size_px, self.detector_input_size_px, self.num_input_layers))
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
            (self.detector_input_size_px, self.detector_input_size_px, self.num_input_layers),
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

    def model_single_mobilenet(self):
        base_model = keras.applications.mobilenet.MobileNet(
            (self.detector_input_size_px, self.detector_input_size_px, self.num_input_layers),
            include_top=False, weights=None)  # weights='imagenet'
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(3, activation='tanh')(x)  # softmax
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
        checkpoint = ModelCheckpoint(join(params['experiment_dir'], 'weights.h5'), save_best_only=True)
        callbacks.extend([csv_logger, checkpoint])
        if 'tensorboard_dir' in params and params['tensorboard_dir'] is not None:
            tb = TensorBoard(log_dir=params['tensorboard_dir'], histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                             write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                             embeddings_metadata=None)
            callbacks.append(tb)
        model.fit_generator(train_generator, steps_per_epoch=params['steps_per_epoch'], epochs=params['epochs'],
                            verbose=1, callbacks=callbacks,
                            validation_data=test_generator, validation_steps=params['n_test'])
        # model.save_weights(join(params['experiment_dir'], 'weights.h5'))
        return model

    def evaluate(self, model, test_generator, params, y_test=None, csv_filename=None):
        pred = model.predict_generator(test_generator)  # , int(params['n_test'] / BATCH_SIZE))
        assert pred is not None and pred is not []

        # with h5py.File(join(params['experiment_dir'], 'predictions.h5'), 'w') as hf:
        #     hf.create_dataset("data", data=pred)

        # pred = (pred - 0.5) * 2  # softmax -> tanh range
        # following expects tanh output (-1; 1)
        pred = self.postprocess_predictions(pred)

        if 'experiment_dir' in params:
            pred_df = self.array.array_to_dataframe(pred)
            pred_df.to_csv(join(params['experiment_dir'], 'predictions.csv'), index=False)
            self.save_model_properties(join(params['experiment_dir'], 'config.yaml'))

        if y_test is not None:
            # xy, _, indices = match_pred_to_gt_dxdy(y_test.values, pred, np)
            errors, errors_xy, _ = self.match_pred_to_gt(y_test[:len(pred)], pred)  # trim y_test to be modulo BATCH_SIZE

            errors_angle = K.concatenate([errors[:, self.array.prop2idx_(i, 'angle_deg')] for i in range(self.num_objects)])

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

            results = pd.DataFrame.from_dict(OrderedDict([
                ('xy MAE', [K.eval(K.mean(errors_xy))]),
                ('angle MAE', [K.eval(K.mean(errors_angle))]),
#                ('axes MAE', [axes_mae]),
                ]))
            if csv_filename is not None:
                results.to_csv(csv_filename)
            return results

    def postprocess_predictions(self, pred):
        for i in range(self.num_objects):
            pred[:, self.array.prop2idx(i, 'angle_deg')] = pred[:, self.array.prop2idx(i, 'angle_deg')] * 90
            pred[:, self.array.prop2idx(i, 'x')] = \
                (pred[:, self.array.prop2idx(i, 'x')] + 1) * self.detector_input_size_px / 2
            pred[:, self.array.prop2idx(i, 'y')] = \
                (pred[:, self.array.prop2idx(i, 'y')] + 1) * self.detector_input_size_px / 2
        return pred

    def save_model_properties(self, out_yaml):
        with open(out_yaml, 'w') as fw:
            yaml.dump({
                'num_objects': self.num_objects,
                'properties': self.PREDICTED_PROPERTIES,
                'input_size_px': self.detector_input_size_px,
            }, fw)

    def evaluate_model(self, data_dir, model_dir, image_store='images.h5:test'):
        """
        Evaluates a model.

        :param data_dir: test dateset images and ground truth (images.h5, test.csv)
        :param model_dir: model directory (model.yaml, weights.h5, config.yaml)
        :param image_store: filename of hdf5 image store and image database path
        """
        # load model
        if os.path.exists(join(model_dir, 'model.yaml')):
            with open(join(model_dir, 'model.yaml'), 'r') as fr:
                m = model_from_yaml(fr.read(), custom_objects={
                       'relu6': mobilenet.relu6})
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
        self.detector_input_size_px = model_metadata['input_size_px']
        self.array = ObjectsArray(self.PREDICTED_PROPERTIES, model_metadata['num_objects'])

        # load images
        hf = h5py.File(join(data_dir, image_store.split(':')[0]), 'r')
        X_test = hf[image_store.split(':')[1]]

        # load gt
        gt_filename = join(data_dir, 'test.csv')
        if os.path.exists(gt_filename):
            gt_n, gt_columns, y_test_df = read_gt(gt_filename)
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

        try:
            os.makedirs(base_experiment_dir)
        except OSError:
            pass
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

    def train_and_evaluate(self, data_dir, loss_alpha, train_img='images.h5:train', test_img='images.h5:test', n_epochs=10, rotate=False, exp_name='',
                           model='6conv_3dense', input_layers=3, experiment_dir=None, input_size_px=200):
        """
        example:
        local: train /home/matej/prace/ferda/data/interactions/1712_1k_36rot/ 0.5 100 --exp-name=two_mobilenet_scratch
               train . 0.5 train.h5:train/img1 test.h5:test/img1 --input-size-px=150 --model=single_mobilenet
        remote: train /mnt/home.stud/smidm/datagrid/ferda/interactions/1712_1k_36rot_fixed/ 0.5 100 --exp-name=two_mobilenet_scratch
        """
        self.num_input_layers = input_layers
        self.detector_input_size_px = input_size_px
        # load images
        hf_train = h5py.File(join(data_dir, train_img.split(':')[0]), 'r')
        X_train = hf_train[train_img.split(':')[1]]
        hf_test = h5py.File(join(data_dir, test_img.split(':')[0]), 'r')
        X_test = hf_test[test_img.split(':')[1]]
        # if model == 'mobilenet':
        #     X_train = self.resize_images(hf[dataset_names['train']], (224, 224, 1))
        #     X_test = self.resize_images(hf[dataset_names['test']], (224, 224, 1))

        # load gt
        n_train, columns_train, y_train_df = read_gt(join(data_dir, 'train.csv'))
        n_test, columns_test, y_test_df = read_gt(join(data_dir, 'test.csv'))
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

        assert model in self.models, 'model {} doesn\'t exist'.format(model)

        parameters = {'epochs': n_epochs,
                      'steps_per_epoch': int(len(X_train) / BATCH_SIZE),  # 1
                      'n_test': len(X_test),
        }
        if isinstance(loss_alpha, str) and loss_alpha == 'batch':
            parameters['loss_alpha'] = np.linspace(0, 1, 8)
        elif isinstance(loss_alpha, numbers.Number):
            parameters['loss_alpha'] = float(loss_alpha)
        else:
            assert False, 'invalid loss_alpha specified'

        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        def rotate90(img):
            tregion.set_img(img)
            return tregion.get_img()
            # out_img = skimage.transform.rotate(img, 90, preserve_range=True)
            # return out_img

        size = self.detector_input_size_px
        x, y = np.mgrid[0:size, 0:size]
        mask = np.expand_dims(np.exp(- 0.0002 * ((x - size / 2) ** 2 + (y - size / 2) ** 2)), 2)

        def image_dim(img):
            return img * mask

        if rotate:
            preprocessing = rotate90
        else:
            preprocessing = None
        if self.num_input_layers == 1:
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

        if experiment_dir is None:
            base_experiment_name = time.strftime("%y%m%d_%H%M", time.localtime()) + '_' + exp_name
            base_experiment_dir = join(ROOT_EXPERIMENT_DIR, base_experiment_name)
            base_tensor_board_dir = join(ROOT_TENSOR_BOARD_DIR, base_experiment_name)
            try:
                os.makedirs(base_experiment_dir)
            except OSError:
                pass
        else:
            base_experiment_dir = experiment_dir
            base_tensor_board_dir = None

        print('argv -{}-'.format(sys.argv))
        self._write_argv(base_experiment_dir)

        results = pd.DataFrame()

        if isinstance(parameters['loss_alpha'], np.ndarray):
            for alpha in parameters['loss_alpha']:
                m = self.models[model]()
                print('loss_alpha %f' % alpha)
                experiment_dir = join(base_experiment_dir, str(alpha))
                try:
                    os.makedirs(experiment_dir)
                except OSError:
                    pass

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

        hf_train.close()
        hf_test.close()

    def _write_argv(self, out_dir):
        with open(join(out_dir, 'parameters.txt'), 'w') as fw:
            fw.writelines('\n'.join(sys.argv))


def read_gt(filename):
    regexp = re.compile('(\d*)_(\w*)')
    df = pd.read_csv(filename)
    ids = set()
    properties = []
    properties_no_prefix = []
    for col in df.columns:
        match = regexp.match(col)
        if match is not None:
            ids.add(int(match.group(1)))
            properties.append(match.group(2))
        else:
            properties_no_prefix.append(col)
    if not properties:
        # not prefixed columns, assuming single object
        n = 1
        properties = properties_no_prefix
        ids.add(0)
        df.columns = ['0_{}'.format(col) for col in df.columns]
    else:
        n = len(ids)
    assert min(ids) == 0 and max(ids) == n - 1, 'object describing columns have to be prefixed with numbers starting with 0'
    assert len(properties) % n == 0
    properties = properties[:(len(properties) / n)]  # only properties for object 0
    return n, properties, df


if __name__ == '__main__':
    ti = TrainInteractions()
    fire.Fire({
      'train': ti.train_and_evaluate,
      'predict': ti.evaluate_model,
    })
