import unittest
from scripts.CNN.train_interactions import TrainInteractions
import scripts.CNN.train_interactions as train_interactions
import numpy as np
from mock import patch
import pandas as pd
from os.path import join
import core.region.transformableregion as tr
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


class LossFunctionsTestCase(unittest.TestCase):
    def setUp(self):
        # ant1_x ant1_y ant1_major ant1_minor ant1_angle_deg ant1_dx ant1_dy; ant2_x ant2_y ant2_major ant2_minor ant2_angle_deg ant2_dx ant2_dy
        self.y_a = np.array([[10., 10, 25, 5, 100, -1, -1, 100, 100, 25, 5, 30, -1, -1],
                             [100., 100, 25, 5, 30, -1, -1, 20, 20, 25, 5, 20, -1, -1],
                             [10., 10, 25, 5, 20, -1, -1, 200, 200, 25, 5, 30, -1, -1]])
        self.y_b = np.array([[20., 20, 25, 5, 30, -1, -1, 150, 170, 25, 5, 0, -1, -1],
                             [30., 30, 25, 5, 30, -1, -1, 170, 150, 25, 5, 5, -1, -1],
                             [30., 60, 25, 5, 30, -1, -1, 170, 120, 25, 5, 5, -1, -1]])
        self.ti = TrainInteractions()

    def run_interaction_loss_angle(self):
        data_dir = '/home/matej/prace/ferda/data/interactions/1712_36k_random'
        n, properties, y_test_df = self.ti.read_gt(join(data_dir, 'test.csv'))
        self.ti.array = train_interactions.ObjectsArray(self.ti.PREDICTED_PROPERTIES, n)
        self.ti.set_num_objects(n)
        y_test = self.ti.array.dataframe_to_array(y_test_df)
        pred = y_test.copy()
        # pred += 1
        pred = 10
        xy, angle, indices = self.ti.match_pred_to_gt(pred[:5], y_test[:5], np)

        # xy_mae = (xy[indices[:, 0], indices[:, 1]]).mean()
        # angle_mae = (angle[indices[:, 0], indices[:, 1]]).mean()
        xy_mae = np.take(xy, indices).mean()
        angle_mae = np.take(angle, indices).mean()
        print(xy_mae)
        print(angle_mae)

        # pred['0_angle_deg'] = 1. / np.tan(np.radians(5.))
        # pred['1_angle_deg'] = 1. / np.tan(np.radians(45.))
        print(train_interactions.K.eval(self.ti.interaction_loss_angle(y_test[:3], pred[:3])))

    def run_match_pred_to_gt(self):
        self.ti.set_num_objects(2)
        with patch.object(self.ti, 'xy_absolute_error',
                          return_value=np.array([[10, 10], [20, 20], [30, 30]])) as mock_method1:
            with patch.object(self.ti, 'angle_absolute_error_direction_agnostic',
                              return_value=np.array([10, 20, 30])) as mock_method2:
                mean_errors_xy, mean_errors_angle, indices = self.ti.match_pred_to_gt(self.y_a, self.y_b, np)
        print(mean_errors_xy)
        print(mean_errors_angle)
        print(indices)


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
        data_dir = '/home/matej/prace/ferda/data/interactions/1712_1k_36rot'
        hf = h5py.File(join(data_dir, 'images.h5'), 'r')
        images = hf['train']
        return images


DATA_DIR = '/home/matej/prace/ferda/data/interactions/1712_1k_36rot/'


class TrainInteractionsTestCase(unittest.TestCase):
    def setUp(self):
        self.n_images = 3
        self.ti = train_interactions.TrainInteractions(2)
        self.input_shape = (self.ti.DETECTOR_INPUT_SIZE_PX, self.ti.DETECTOR_INPUT_SIZE_PX, self.ti.num_input_layers)
        self.hf = h5py.File(join(DATA_DIR, 'images.h5'), 'r')
        self.X_train = self.hf['train']
        if self.input_shape[2] == 1:
            self.X_train_ = np.mean(self.X_train[:self.n_images], axis=3, keepdims=True)
        else:
            self.X_train_ = self.X_train[:self.n_images]
        self.X_train_ = self.ti.resize_images(self.X_train_, self.input_shape)

        self.X_test = self.hf['test']
        n, properties, self.y_test_df = self.ti.read_gt(join(DATA_DIR, 'test.csv'))
        n, properties, self.y_train_df = self.ti.read_gt(join(DATA_DIR, 'test.csv'))
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
        loss = K.eval(self.ti.interaction_loss_angle(self.y_train[:self.n_images], pred))
        self.assertTrue(np.isscalar(loss))
        self.assertTrue(loss > 0)

    def test_train(self):
        n_images = 3
        m = self.ti.model_mobilenet()
        m.compile(loss=lambda x, y: self.ti.interaction_loss_angle(x, y, alpha=0.5), optimizer='adam')
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow(self.X_train_, self.y_train[:self.n_images], batch_size=self.n_images)

        w1 = m.get_weights()
        m.fit_generator(train_generator, steps_per_epoch=1, epochs=1, verbose=1)
        w2 = m.get_weights()

        # # output differences
        # for i, (w_pre, w_post) in enumerate(zip(w1, w2)):
        #     assert_array_equal(w_pre, w_post, 'layer {}'.format(i))


if __name__ == '__main__':
    unittest.main()
