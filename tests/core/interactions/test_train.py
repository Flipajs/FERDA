import unittest

import h5py
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from numpy.testing import assert_array_equal
from os.path import join

import core.interactions.train as train_interactions
import shapes.transformableregion as tr
from core.interactions.train import TrainInteractions
from core.interactions.io import read_gt


class LossFunctionsTestCase(unittest.TestCase):
    def setUp(self):
        # ant1_x ant1_y ant1_major ant1_minor ant1_angle_deg ant1_dx ant1_dy; ant2_x ant2_y ant2_major ant2_minor ant2_angle_deg ant2_dx ant2_dy
        self.y_a = np.array([[10., 10, 25, 5, 20,   100, 100, 25, 5, 30],
                             [200, 200, 25, 5, 30,   10., 10, 25, 5, 20]])

        self.y_b = np.array([[20., 20, 20, 4, -5,   150, 170, 25, 5, 0],
                             [30., 60, 30, 3, 30,   170, 120, 25, 5, 5]])

        self.y_a1 = np.array([[100, 100, 25, 5, 30],
                             [10., 10, 25, 5, 20]])

        self.y_b1 = np.array([[150, 170, 25, 5, 0],
                             [170, 120, 25, 5, 5]])

        self.ti = TrainInteractions(predicted_properties=['x', 'y', 'angle_deg', 'major', 'minor'],
                                    error_functions=['abs', 'abs', 'angle_180', 'abs', 'abs'])

    def test_errors_ij(self):
        errors00 = self.ti.errors_ij(self.y_a, self.y_b, 0, 0)
        assert_array_equal(errors00, [[10, 10, 5, 1, 25],
                                      [170, 140, 5, 2, 0]])
        errors01 = self.ti.errors_ij(self.y_a, self.y_b, 0, 1)
        assert_array_equal(errors01, [[140, 160, 0, 0, 20],
                                      [30, 80, 0, 0, 25]])

    def test_interaction_loss_angle(self):
        data_dir = 'test/interactions_dataset'
        n, properties, y_test_df = read_gt(join(data_dir, 'test.csv'))
        self.ti.array = train_interactions.ObjectsArray(self.ti.PREDICTED_PROPERTIES, n)
        self.ti.set_num_objects(n)
        y_test = self.ti.array.dataframe_to_array(y_test_df)
        pred = y_test.copy()
        # pred += 1
        pred[:] = 10
        # xy, angle, indices = self.ti.match_pred_to_gt(pred[:5], y_test[:5])
        errors, errors_xy, indices = self.ti.match_pred_to_gt(pred, y_test)

        print((K.eval(errors)))
        print((K.eval(errors_xy)))

        # pred['0_angle_deg'] = 1. / np.tan(np.radians(5.))
        # pred['1_angle_deg'] = 1. / np.tan(np.radians(45.))
        print((train_interactions.K.eval(self.ti.loss(y_test[:3], pred[:3]))))

    def test_match_pred_to_gt(self):
        self.ti.set_num_objects(2)
        errors, errors_xy, indices = self.ti.match_pred_to_gt(self.y_a, self.y_b)
        print((K.eval(errors)))
        print((K.eval(errors_xy)))
        print((K.eval(indices)))

        self.ti.set_num_objects(1)
        errors, errors_xy, indices = self.ti.match_pred_to_gt(self.y_a1, self.y_b1)
        print((K.eval(errors)))
        print((K.eval(errors_xy)))
        print((K.eval(indices)))

        # test tf.gather_nd (used in match_pred_to_gt)
        err1 = K.variable(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))  # shape=(n, n_objects * len(PREDICTED_PROPERTIES))
        err2 = K.variable(np.array([[10, 11, 12, 13], [14, 15, 16, 17]]))  # shape=(n, n_objects * len(PREDICTED_PROPERTIES))
        errs = K.stack([err1, err2])
        indices = K.variable([1, 0], dtype='int32')
        n = K.shape(indices)[0]
        indices_gather = K.transpose(K.stack((indices, K.arange(0, n))))  # shape=(n, 2)
        assert_array_equal(K.eval(tf.gather_nd(errs, indices_gather)),
                           np.array([[10, 11, 12, 13], [5, 6, 7, 8]]))

        # with patch.object(self.ti, 'xy_absolute_error',
        #                   return_value=np.array([[10, 10], [20, 20], [30, 30]])) as mock_method1:
        #     with patch.object(angles, 'angle_absolute_error_direction_agnostic',
        #                       return_value=np.array([10, 20, 30])) as mock_method2:
        #         mean_errors_xy, axes, mean_errors_angle, indices = self.ti.match_pred_to_gt(self.y_a, self.y_b, np)
        # print(mean_errors_xy)
        # print(mean_errors_angle)



    # def test_interaction_loss_angle(self):
    #     assert train_interactions.NUM_OBJECTS == 2
    #     loss = train_interactions.interaction_loss_angle(self.y_a, self.y_b)
    #     print train_interactions.K.eval(loss)

    @staticmethod
    def run_generator_with_preprocessing():
        import skimage.io as io
        size = 200
        x, y = np.mgrid[0:size, 0:size]
        mask = np.exp(- 0.0002 * ((x - size / 2) ** 2 + (y - size / 2) ** 2))
        io.imshow(np.uint8(mask * 255))


        X_train = LossFunctionsTestCase.load_images()

        tregion = tr.TransformableRegion(X_train[0])
        tregion.rotate(20, np.array(tregion.img.shape[:2]) / 2)

        def rotate(img):
            tregion.set_img(img)
            return tregion.get_img()

        def image_dim(img):
            return img * np.expand_dims(mask, 2)

        from keras.preprocessing.image import ImageDataGenerator
        train_datagen = ImageDataGenerator(preprocessing_function=rotate, rescale=1./255)
        train_generator = train_datagen.flow(X_train[:10], shuffle=False, batch_size=1)

        io.imshow(X_train[0])
        plt.figure()
        io.imshow(next(train_generator)[0])  # .astype(np.uint8)
        io.show()

    @staticmethod
    def load_images():
        import h5py
        from os.path import join
        data_dir = 'test/interactions_dataset'
        hf = h5py.File(join(data_dir, 'images.h5'), 'r')
        images = hf['train']
        return images


DATA_DIR = 'test/interactions_dataset'


class TrainInteractionsTestCase(unittest.TestCase):
    def setUp(self):
        self.n_images = 3
        self.ti = train_interactions.TrainInteractions(1)
        self.input_shape = (self.ti.detector_input_size_px, self.ti.detector_input_size_px, self.ti.num_input_layers)
        self.hf = h5py.File(join(DATA_DIR, 'images.h5'), 'r')
        self.X_train = self.hf['train/img1']
        self.X_test = self.hf['test/img1']
        if self.input_shape[2] == 1:
            self.X_train_ = np.mean(self.X_train[:self.n_images], axis=3, keepdims=True)
            self.X_test_ = np.mean(self.X_test, axis=3, keepdims=True)
        else:
            self.X_train_ = self.X_train[:self.n_images]
            self.X_test_ = self.X_test
        self.X_train_ = self.ti.resize_images(self.X_train_, self.input_shape)
        self.X_test_ = self.ti.resize_images(self.X_test_, self.input_shape)
        n, properties, self.y_test_df = read_gt(join(DATA_DIR, 'test.csv'))
        n, properties, self.y_train_df = read_gt(join(DATA_DIR, 'train.csv'))
        self.y_test = self.ti.array.dataframe_to_array(self.y_test_df)
        self.y_train = self.ti.array.dataframe_to_array(self.y_train_df)

    def tearDown(self):
        self.hf.close()

    def test_init_model_mobilenet(self):
        m = self.ti.model_mobilenet()
        out = m.predict(self.X_train_)
        self.assertEqual(out.shape[0], self.n_images)
        self.assertEqual(out.shape[1], self.ti.array.num_columns())
        self.assertTrue(np.all((out > -1) & (out < 1)))
        # pd.DataFrame(out).to_csv('model_out.csv')
        # print out

    def test_init_model_6conv_3dense(self):
        m = self.ti.model_6conv_3dense()
        out = m.predict(self.X_train_)
        self.assertEqual(out.shape[0], self.n_images)
        self.assertEqual(out.shape[1], self.ti.array.num_columns())
        self.assertTrue(np.all((out > -1) & (out < 1)))
        # pd.DataFrame(out).to_csv('model_out.csv')
        # print out

    def test_loss(self):
        m = self.ti.model_mobilenet()
        pred = m.predict(self.X_train_)
        loss = K.eval(self.ti.loss(self.y_train[:self.n_images], pred))
        self.assertTrue(np.isscalar(loss))
        self.assertTrue(loss > 0)

    def test_train(self):
        n_images = 3
        m = self.ti.model_mobilenet()
        m.compile(loss=lambda x, y: self.ti.loss(x, y, alpha=0.5), optimizer='adam')
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow(self.X_train_, self.y_train[:self.n_images], batch_size=self.n_images)

        w1 = m.get_weights()
        m.fit_generator(train_generator, steps_per_epoch=1, epochs=1, verbose=1)
        w2 = m.get_weights()

        # # output differences
        # for i, (w_pre, w_post) in enumerate(zip(w1, w2)):
        #     assert_array_equal(w_pre, w_post, 'layer {}'.format(i))

    def test_eval(self):
        n_images = 3
        m = self.ti.model_mobilenet()
        m.compile(loss=lambda x, y: self.ti.loss(x, y, alpha=0.5), optimizer='adam')
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow(self.X_test_, self.y_test, batch_size=self.n_images)
        results = self.ti.evaluate(m, test_generator, {'n_test': len(self.y_test)}, self.y_test)


if __name__ == '__main__':
    unittest.main()
