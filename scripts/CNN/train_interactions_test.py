import unittest
import numpy as np
from scripts.CNN.train_interactions import angle_absolute_error, angle_absolute_error_direction_agnostic
import scripts.CNN.train_interactions as train_interactions
import matplotlib.pylab as plt
import numpy as np
from mock import patch
from numpy.testing import assert_array_equal
import pandas as pd
from os.path import join


class LossFunctionsTestCase(unittest.TestCase):
    def setUp(self):
        # ant1_x ant1_y ant1_major ant1_minor ant1_angle_deg ant1_dx ant1_dy; ant2_x ant2_y ant2_major ant2_minor ant2_angle_deg ant2_dx ant2_dy
        self.y_a = np.array([[10., 10, 25, 5, 100, -1, -1, 100, 100, 25, 5, 30, -1, -1],
                             [100., 100, 25, 5, 30, -1, -1, 20, 20, 25, 5, 20, -1, -1],
                             [10., 10, 25, 5, 20, -1, -1, 200, 200, 25, 5, 30, -1, -1]])
        self.y_b = np.array([[20., 20, 25, 5, 30, -1, -1, 150, 170, 25, 5, 0, -1, -1],
                             [30., 30, 25, 5, 30, -1, -1, 170, 150, 25, 5, 5, -1, -1],
                             [30., 60, 25, 5, 30, -1, -1, 170, 120, 25, 5, 5, -1, -1]])

    def run_interaction_loss_angle(self):
        data_dir = '/home/matej/prace/ferda/data/interactions/1712_36k_random'
        y_test_df = pd.read_csv(join(data_dir, 'test.csv'))
        y_test = y_test_df[train_interactions.columns(train_interactions.NUM_OBJECTS)]
        pred = y_test.copy()
        # pred += 1
        pred.iloc[:] = 1
        xy, angle, indices = train_interactions.match_pred_to_gt(pred.values[:5], y_test.values[:5], np)

        # xy_mae = (xy[indices[:, 0], indices[:, 1]]).mean()
        # angle_mae = (angle[indices[:, 0], indices[:, 1]]).mean()
        xy_mae = np.take(xy, indices).mean()
        angle_mae = np.take(angle, indices).mean()
        print(xy_mae)
        print(angle_mae)

        print train_interactions.K.eval(train_interactions.interaction_loss_angle(y_test.values[:5], pred.values[:5]))

    def run_match_pred_to_gt(self):
        assert train_interactions.NUM_OBJECTS == 2
        with patch.object(train_interactions, 'xy_absolute_error',
                          return_value=np.array([[10, 10], [20, 20], [30, 30]])) as mock_method1:
            with patch.object(train_interactions, 'angle_absolute_error_direction_agnostic',
                              return_value=np.array([10, 20, 30])) as mock_method2:
                mean_errors_xy, mean_errors_angle, indices = train_interactions.match_pred_to_gt(self.y_a, self.y_b, np)
        print(mean_errors_xy)
        print(mean_errors_angle)
        print(indices)

    def test_angle_absolute_error(self):
        self.assertEqual(angle_absolute_error(10, 20, np), 10)
        self.assertEqual(angle_absolute_error(10, 180, np), 170)
        self.assertEqual(angle_absolute_error(0, 180, np), 180)
        self.assertEqual(angle_absolute_error(0, 190, np), 170)
        self.assertEqual(angle_absolute_error(0, 270, np), 90)
        self.assertEqual(angle_absolute_error(90, 270, np), 180)
        self.assertEqual(angle_absolute_error(-10, 10, np), 20)
        self.assertEqual(angle_absolute_error(80, 280, np), 160)
        self.assertEqual(angle_absolute_error(0, -10, np), 10)
        self.assertEqual(angle_absolute_error(0, 360, np), 0)
        self.assertEqual(angle_absolute_error(10, 300, np), 70)

    def test_angle_absolute_error_direction_agnostic(self):
        self.assertEqual(angle_absolute_error_direction_agnostic(10, 20, np), 10)
        self.assertEqual(angle_absolute_error_direction_agnostic(10, 180, np), 10)
        self.assertEqual(angle_absolute_error_direction_agnostic(0, 180, np), 0)
        self.assertEqual(angle_absolute_error_direction_agnostic(0, 190, np), 10)
        self.assertEqual(angle_absolute_error_direction_agnostic(0, 270, np), 90)
        self.assertEqual(angle_absolute_error_direction_agnostic(90, 270, np), 0)
        self.assertEqual(angle_absolute_error_direction_agnostic(-10, 10, np), 20)
        self.assertEqual(angle_absolute_error_direction_agnostic(80, 280, np), 20)
        self.assertEqual(angle_absolute_error_direction_agnostic(0, -10, np), 10)
        self.assertEqual(angle_absolute_error_direction_agnostic(0, 360, np), 0)
        self.assertEqual(angle_absolute_error_direction_agnostic(10, 300, np), 70)
        assert_array_equal(angle_absolute_error_direction_agnostic(np.array([10, 0, 0]),
                                                                   np.array([300, 360, -10]), np),
                           np.array([70, 0, 10]))

    # def test_interaction_loss_angle(self):
    #     assert train_interactions.NUM_OBJECTS == 2
    #     loss = train_interactions.interaction_loss_angle(self.y_a, self.y_b)
    #     print train_interactions.K.eval(loss)

    @staticmethod
    def run_generator_with_preprocessing():
        import skimage.io as io
        from skimage.transform import rotate
        size = 200
        x, y = np.mgrid[0:size, 0:size]
        mask = np.exp(- 0.0002 * ((x - size / 2) ** 2 + (y - size / 2) ** 2))
        io.imshow(np.uint8(mask * 255))

        def image_dim(img):
            return img * np.expand_dims(mask, 2)

        def rotate90(img):
            return rotate(img, 90, preserve_range=True)

        X_train = LossFunctionsTestCase.load_images()

        from keras.preprocessing.image import ImageDataGenerator
        train_datagen = ImageDataGenerator(preprocessing_function=image_dim, rescale=1./255)
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


if __name__ == '__main__':
    # LossFunctionsTestCase.run_generator_with_preprocessing()
    unittest.main()
