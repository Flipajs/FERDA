from __future__ import print_function
import os
from collections import OrderedDict
from os.path import join
import h5py
import numpy as np
import pandas as pd

from core.interactions.io import read_gt
from utils.experiment import Experiment

try:
    from keras.utils import np_utils
    from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, GlobalAveragePooling2D, Concatenate, concatenate
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
    from keras.utils import Sequence
    from core.interactions.keras_utils import GetBest
except ImportError as e:
    print('Warning, no keras installed: {}'.format(e))
import fire
from core.region.transformableregion import TransformableRegion
from utils.angles import angle_absolute_error, angle_absolute_error_direction_agnostic
import warnings
import yaml
from utils.objectsarray import ObjectsArray
from core.interactions.visualization import visualize_results
# import skimage.transform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ValidationCallback(Callback):
    def __init__(self, test_dataset, evaluate_fun):
        self.test_dataset = test_dataset
        self.evaluate_fun = evaluate_fun

    def on_epoch_end(self, epoch, logs={}):
        self.evaluate_fun(self.model, self.test_dataset)


class Hdf5CsvSequence(Sequence):

    def __init__(self, hdf5_filename, dataset_name, csv_filename, batch_size, array):
        self.batch_size = batch_size
        self.h5file = h5py.File(hdf5_filename, 'r')
        self.x = self.h5file[dataset_name]
        n_train, columns_train, df = read_gt(csv_filename)
        self.y = array.dataframe_to_array(df)
#        warnings.warn('Ground truth file test.csv not found. Generating predictions without evaluation.')

        # if self.num_input_layers == 1:
        #     dataset = np.expand_dims(dataset, 3)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
        # return 1

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x / 255., batch_y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __del__(self):
        self.h5file.close()


class TwoImageSequence(Hdf5CsvSequence):
    def __init__(self, hdf5_filename, dataset_basename, csv_filename, batch_size, array):
        super(TwoImageSequence, self).__init__(hdf5_filename, dataset_basename + '1', csv_filename, batch_size, array)
        self.x0 = self.h5file[dataset_basename + '0']

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        x1, y = super(TwoImageSequence, self).__getitem__(idx)
        x0 = self.x0[idx * self.batch_size:(idx + 1) * self.batch_size] / 255.
        return [x0, x1], y


class StackedTwoImageSequence(TwoImageSequence):
    def __getitem__(self, idx):
        [x0, x1], y = super(StackedTwoImageSequence, self).__getitem__(idx)
        return np.concatenate((x0, x1), axis=3), y


class TrainInteractions:
    def __init__(self, predicted_properties=None, error_functions=None):
        assert (predicted_properties is None and error_functions is None) or \
               len(predicted_properties) == len(error_functions)
        self.models = {
            '6conv_3dense': self.model_6conv_3dense,  # former default
            '6conv_3dense_two_inputs': self.model_6conv_3dense_two_inputs,
            '6conv_2dense': self.model_6conv_2dense,
            'mobilenet': self.model_mobilenet,
            'single_mobilenet': self.model_single_mobilenet,
            'mobilenet_siamese': self.model_mobilenet_siamese,
            'single_concat_mp': self.model_single_concat_mp,
            'single_concat_conv3': self.model_single_concat_conv3,
            'single_concat_conv3_2inputs': self.model_single_concat_conv3_2inputs,
            'single_concat_conv3_dense32': self.model_single_concat_conv3_dense32,
            'mobilenet2_siamese': self.model_mobilenet2_siamese,
            'resnet_siamese': self.model_resnet_siamese,
        }
        if predicted_properties is None:
            self.PREDICTED_PROPERTIES = ['x', 'y', 'angle_deg_cw']  # , 'major', 'minor']
        else:
            self.PREDICTED_PROPERTIES = predicted_properties
        if error_functions is None:
            self.ERROR_FUNCTIONS = ['abs', 'abs', 'angle_360']  # , 'abs', 'abs']
        else:
            self.ERROR_FUNCTIONS = error_functions
        self.num_objects = 1
        self.array = ObjectsArray(self.PREDICTED_PROPERTIES, self.num_objects)  # provide column name to index mapping using ObjectsArray
        self.parameters = {}
        self.input_shape = None

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

    def loss(self, y_true, y_pred, alpha=0.5):
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
                if col != 'angle_deg_cw':
                    # (-1; 1) -> (0; detector_input_size_px)
                    tensor_columns.append((y_pred[:, self.array.prop2idx(i, col)] + 1) *
                                          self.parameters['input_size_px'] / 2)
                else:
                    # (-1; 1) -> (-90; 90)
                    tensor_columns.append(y_pred[:, self.array.prop2idx(i, 'angle_deg_cw')] * 90)
        y_pred = K.stack(tensor_columns, axis=1)

        errors, errors_xy, _ = self.match_pred_to_gt(y_true, y_pred)

        loss = 0
        for i in range(self.num_objects):
            error_angle = errors[:, self.array.prop2idx(i, 'angle_deg_cw')]  # shape=(n,)
            error_xy = errors_xy[:, i]  # shape=(n,)
            loss += K.sum(error_xy * (1 - alpha) + error_angle * alpha)

        return loss

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

    def model_6conv_3dense_convolutions(self):
        input_shape = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_shape)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        x = Conv2D(32, (3, 3))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(16, (3, 3))(x)
        x = Flatten()(x)
        return input_shape, x

    def model_6conv_3dense(self):
        input_shape, x = self.model_6conv_3dense_convolutions()
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(self.array.num_columns(), kernel_initializer='normal', activation='tanh')(x)
        return Model(input_shape, out)

    def model_6conv_3dense_two_inputs(self):
        input1, x1 = self.model_6conv_3dense_convolutions()
        input2, x2 = self.model_6conv_3dense_convolutions()
        x = Concatenate()([x1, x2])
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(self.array.num_columns(), kernel_initializer='normal', activation='tanh')(x)
        return Model([input1, input2], out)

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
        input_shape = Input(shape=self.input_shape)
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
        base_model = keras.applications.mobilenet.MobileNet(self.input_shape,
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
        base_model = keras.applications.mobilenet.MobileNet(self.input_shape,
                                                            self.parameters['mobilenet_alpha'],
                                                            self.parameters['mobilenet_depth_multiplier'],
                                                            include_top=False, weights=None)  # weights='imagenet'
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(32, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(3, activation='tanh')(x)  # softmax
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        # for layer in base_model.layers:
        #     layer.trainable = False
        return model

    def model_mobilenet_siamese(self):
        from keras.applications.mobilenet import mobilenet
        base_model = keras.applications.mobilenet.MobileNet(self.input_shape,
                                                            self.parameters['mobilenet_alpha'],
                                                            self.parameters['mobilenet_depth_multiplier'],
                                                            include_top=False, weights=None,
                                                            pooling='avg')  # weights='imagenet'
        input0 = keras.layers.Input(shape=self.input_shape)
        x0 = base_model(input0)
        input1 = keras.layers.Input(shape=self.input_shape)
        x1 = base_model(input1)
        x = keras.layers.concatenate([x0, x1], axis=-1)
        # x = Dense(1024, activation='relu')(x)
        # predictions = Dense(3, activation='tanh')(x)  # softmax

        if K.image_data_format() == 'channels_first':
            shape = (int(2 * 1024 * self.parameters['mobilenet_alpha']), 1, 1)
        else:
            shape = (1, 1, int(2 * 1024 * self.parameters['mobilenet_alpha']))

        x = keras.layers.Reshape(shape, name='reshape_1')(x)
        x = keras.layers.Dropout(1e-3, name='dropout')(x)
        x = keras.layers.Conv2D(3, (1, 1),
                          padding='same',
                          name='conv_preds')(x)
        x = keras.layers.Activation('tanh', name='act_tanh')(x)
        x = keras.layers.Reshape((3,), name='reshape_2')(x)

        model = Model(inputs=[input0, input1], outputs=x)
        return model

    def model_mobilenet2_siamese(self):
        from keras.applications.mobilenet_v2 import MobileNetV2
        base_model = MobileNetV2(self.input_shape, self.parameters['mobilenet_alpha'],
                                 self.parameters['mobilenet_depth_multiplier'],
                                 include_top=False, weights='imagenet',  # 'imagenet',
                                 pooling='avg')
        input0 = keras.layers.Input(shape=self.input_shape)
        x0 = base_model(input0)
        input1 = keras.layers.Input(shape=self.input_shape)
        x1 = base_model(input1)
        x = keras.layers.concatenate([x0, x1], axis=-1)
        x = keras.layers.Dense(3, activation='tanh', use_bias=True, name='3xtanh')(x)

        model = Model(inputs=[input0, input1], outputs=x)
        return model

    def model_resnet_siamese(self):
        from keras.applications.resnet50 import ResNet50
        base_model = ResNet50(input_shape=self.input_shape, include_top=False, pooling='avg')  # weights='imagenet',
        input0 = keras.layers.Input(shape=self.input_shape)
        x0 = base_model(input0)
        input1 = keras.layers.Input(shape=self.input_shape)
        x1 = base_model(input1)
        x = keras.layers.concatenate([x0, x1], axis=-1)
        x = keras.Flatten()(x)
        x = keras.Dense(3, activation='tanh', name='3xtanh')(x)
        model = Model(inputs=[input0, input1], outputs=x)
        return model

    def model_single_concat_mp(self):
        input_shape = Input(shape=self.input_shape)
        x = Conv2D(1, (1, 1), padding='same', activation='relu')(input_shape)
        conv2 = Conv2D(32, (7, 7), padding='same', activation='relu')(x)
        conv3 = Conv2D(32, (5, 5), padding='same', activation='relu')(conv2)
        mp1 = MaxPooling2D((2, 2))(conv3)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Flatten()(x)

        x = concatenate([x, Flatten()(mp1)])

        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(self.array.num_columns(), activation='tanh')(x)
        return Model(input_shape, out)

    def _model_single_concat_conv3_convolutions(self):
        input_shape = Input(shape=self.input_shape)
        x = Conv2D(1, (1, 1), padding='same', activation='relu')(input_shape)
        conv2 = Conv2D(32, (7, 7), padding='same', activation='relu')(x)
        conv3 = Conv2D(32, (5, 5), padding='same', activation='relu')(conv2)
        mp1 = MaxPooling2D((2, 2))(conv3)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Flatten()(x)

        x = concatenate([x, Flatten()(conv3)])
        return input_shape, x

    def model_single_concat_conv3(self):
        input_shape, x = self._model_single_concat_conv3_convolutions()
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(self.array.num_columns(), activation='tanh')(x)
        return Model(input_shape, out)

    def model_single_concat_conv3_dense32(self):
        input_shape, x = self._model_single_concat_conv3_convolutions()
        x = Dense(32, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(self.array.num_columns(), activation='tanh')(x)
        return Model(input_shape, out)

    def model_single_concat_conv3_2inputs(self):
        input1, x1 = self._model_single_concat_conv3_convolutions()
        input2, x2 = self._model_single_concat_conv3_convolutions()
        x = Concatenate()([x1, x2])
        x = Dense(64, activation='relu')(x)
        # x = Dense(32, activation='relu')(x)
        out = Dense(self.array.num_columns(), activation='tanh')(x)
        return Model([input1, input2], out)

    def train(self, model, train_dataset, experiment, test_dataset=None, callbacks=None):
        if callbacks is None:
            callbacks = []
        # adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=lambda x, y: self.loss(x, y, alpha=experiment.params['loss_alpha']),
                      optimizer='adam')
        # model.lr.set_value(0.05)
        with open(join(experiment.dir, 'model.txt'), 'w') as fw:
            model.summary(print_fn=lambda x: fw.write(x + '\n'))
        callbacks.extend([
            CSVLogger(join(experiment.dir, 'log.csv'), append=True, separator=';'),
            GetBest(monitor='val_loss', verbose=1),
            ModelCheckpoint(join(experiment.dir, 'weights_tmp.h5'), save_best_only=True, save_weights_only=True)
        ])
        if experiment.tensor_board_dir is not None:
            callbacks.append(
                TensorBoard(log_dir=experiment.tensor_board_dir, histogram_freq=0,
                            batch_size=experiment.params['batch_size'], write_graph=True,
                            write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                            embeddings_metadata=None))
        model.fit_generator(train_dataset, epochs=experiment.params['epochs'], verbose=1, callbacks=callbacks,
                            validation_data=test_dataset)
        model.save_weights(join(experiment.dir, 'weights.h5'))
        os.remove(join(experiment.dir, 'weights_tmp.h5'))
        return model

    def evaluate(self, model, dataset, experiment=None, evaluation_csv_filename=None, prediction_csv_filename=None):
        pred = model.predict_generator(dataset, verbose=0)
        assert pred is not None and pred is not []

        # following expects tanh output (-1; 1)
        pred = self.postprocess_predictions(pred)

        if experiment is not None:
            if prediction_csv_filename is None:
                prediction_csv_filename = join(experiment.dir, 'predictions.csv')
            pred_df = self.array.array_to_dataframe(pred)
            pred_df.to_csv(prediction_csv_filename, index=False)
            self.save_model_properties(join(experiment.dir, 'config.yaml'))

        if dataset.y is not None:
            errors, errors_xy, _ = self.match_pred_to_gt(np.vstack([item[1] for item in dataset]), pred)
            errors_angle = K.concatenate([errors[:, self.array.prop2idx_(i, 'angle_deg_cw')] for i in range(self.num_objects)])

            results = pd.DataFrame.from_dict(OrderedDict([
                ('xy MAE', [K.eval(K.mean(errors_xy))]),
                ('angle MAE', [K.eval(K.mean(errors_angle))]),
                ]))
            if evaluation_csv_filename is not None:
                results.to_csv(evaluation_csv_filename)
            else:
                print(results)
            return results
        else:
            return None

    def postprocess_predictions(self, pred):
        for i in range(self.num_objects):
            pred[:, self.array.prop2idx(i, 'angle_deg_cw')] = pred[:, self.array.prop2idx(i, 'angle_deg_cw')] * 90
            pred[:, self.array.prop2idx(i, 'x')] = \
                (pred[:, self.array.prop2idx(i, 'x')] + 1) * self.parameters['input_size_px'] / 2
            pred[:, self.array.prop2idx(i, 'y')] = \
                (pred[:, self.array.prop2idx(i, 'y')] + 1) * self.parameters['input_size_px'] / 2
        return pred

    def save_model_properties(self, out_yaml):
        with open(out_yaml, 'w') as fw:
            yaml.dump({
                'num_objects': self.num_objects,
                'properties': self.PREDICTED_PROPERTIES,
            }, fw)

    def evaluate_model(self, data_dir, model_dir, image_store='images.h5:test', target_csv_file='test.csv'):
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
        model_metadata = yaml.load(open(join(model_dir, 'config.yaml')))
        self.PREDICTED_PROPERTIES = model_metadata['properties']
        self.array = ObjectsArray(self.PREDICTED_PROPERTIES, model_metadata['num_objects'])

        parameters = {'batch_size': 32,
                      }

        # load images
        test_dataset = Hdf5CsvSequence(os.path.join(data_dir, image_store.split(':')[0]),
                                       image_store.split(':')[1],
                                       join(data_dir, target_csv_file),
                                       parameters['batch_size'],
                                       self.array)

        # size = 200
        # x, y = np.mgrid[0:size, 0:size]
        # mask = np.expand_dims(np.exp(- 0.0002 * ((x - size / 2) ** 2 + (y - size / 2) ** 2)), 2)
        #
        # def image_dim(img):
        #     return img * mask

        results = self.evaluate(m, test_dataset, Experiment('evaluation'))

    def resize_images(self, img_batch, shape):
        img_shape = img_batch[0].shape
        assert img_shape[0] <= shape[0] and img_shape[1] <= shape[1]
        out = np.zeros(shape=((len(img_batch), ) + shape), dtype=img_batch[0].dtype)
        for i, img in enumerate(img_batch):
            out[i, :img_shape[0], :img_shape[1]] = img  # np.expand_dims(img, 2)
        return out

    def load_datasets(self, datasets_str, data_dir='.'):
        datasets = []
        h5files = []
        for dataset_str in datasets_str.split(';'):
            filename, dataset_name = dataset_str.split(':')
            h5file = h5py.File(join(data_dir, filename), 'r')
            dataset = h5file[dataset_name]
            if self.parameters['input_channels'] == 1:
                dataset = np.expand_dims(dataset, 3)
            datasets.append(dataset)
            h5files.append(h5file)
        return datasets, h5files

    def train_and_evaluate(self, data_dir, train_img='images.h5:train', test_img='images.h5:test', **parameters):
        """
        example usage:
        train /datagrid/ferda/datasets/interactions/181116_cam1_5k_rot/ images.h5:train/img1 images.h5:test/img1
        --model=mobilenet --name=mobilenet_last --batch_size=32 --epochs=150 --loss_alpha=1
        """
        if not ('epochs' in parameters and 'loss_alpha' in parameters and 'batch_size' in parameters):
            print("""missing experimental parameters, minimal example:
parameters = {'epochs': 150,
              'loss_alpha': 0.1,
              'batch_size': 32,
              }""")
            return

        train_dataset = TwoImageSequence(os.path.join(data_dir, train_img.split(':')[0]),  # TwoImageSequence Hdf5CsvSequence
                                                train_img.split(':')[1],
                                                join(data_dir, 'train.csv'),
                                                parameters['batch_size'],
                                                self.array)
        test_dataset = TwoImageSequence(os.path.join(data_dir, test_img.split(':')[0]),
                                               test_img.split(':')[1],
                                               join(data_dir, 'test.csv'),
                                               parameters['batch_size'],
                                               self.array)

        assert parameters['model'] in self.models, 'model {} doesn\'t exist'.format(parameters['model'])
        batch_x, _ = train_dataset[0]
        if isinstance(batch_x, list):
            batch_x = batch_x[0]  # inspect only the first input
        parameters['input_size_px'] = batch_x.shape[1]  # (batch size, y, x, channels)
        parameters['input_channels'] = batch_x.shape[3]
        self.input_shape = batch_x.shape[1:]
        self.parameters = parameters

        # fix random seed for reproducibility
        # seed = 7
        # np.random.seed(seed)

        # size = self.detector_input_size_px
        # x, y = np.mgrid[0:size, 0:size]
        # mask = np.expand_dims(np.exp(- 0.0002 * ((x - size / 2) ** 2 + (y - size / 2) ** 2)), 2)
        #
        # def image_dim(img):
        #     return img * mask

        # def get_sample_weight(df, detector_input_size_px, max_reweighted_distance_px=20):
        #     distance = np.linalg.norm(df[['0_x', '0_y']] - detector_input_size_px / 2, axis=1)
        #     kde = stats.gaussian_kde(distance)
        #     weights = 1. / kde(distance)
        #     max_weight = weights[distance < max_reweighted_distance_px].max()
        #     weights[distance >= max_reweighted_distance_px] = max_weight
        #     return weights / weights.min()

        # def eval(_m, _t):
        #     results = self.evaluate(_m, test_dataset, parameters)
        # callbacks = [ValidationCallback(test_dataset, eval), ]
        callbacks = None

        root_experiment = Experiment.create(parameters.get('name', None), params=parameters,
                                            config=yaml.load(open('interactions_config.yaml')),
                                            tensorboard=True)
        root_experiment.save_params()
        for experiment in root_experiment:
            print(experiment.basename)
            experiment.save_argv()
            experiment.save_params()
            m = self.models[parameters['model']]()
            open(join(experiment.dir, 'model.yaml'), 'w').write(m.to_yaml())
            m = self.train(m, train_dataset, experiment, test_dataset, callbacks=callbacks)
            results = self.evaluate(m, test_dataset, experiment,
                                    evaluation_csv_filename=join(experiment.dir, 'results.csv'),
                                    prediction_csv_filename=join(experiment.dir, 'predictions.csv'),
                                    )
            print('test dataset')
            print(results.to_string(index=False))
            results = self.evaluate(m, train_dataset, experiment,
                                    evaluation_csv_filename=join(experiment.dir, 'results_train.csv'),
                                    prediction_csv_filename=join(experiment.dir, 'predictions_train.csv'),
                                    )
            print('train dataset')
            print(results.to_string(index=False))
            visualize_results(experiment.dir, data_dir, 'images.h5:test/img1')


if __name__ == '__main__':
    ti = TrainInteractions()
    fire.Fire({
      'train': ti.train_and_evaluate,
      'predict': ti.evaluate_model,
    })
