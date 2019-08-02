import unittest
from utils.img import safe_crop
import numpy as np
from numpy.testing import assert_array_equal


class safe_crop_TestCase(unittest.TestCase):
    def setUp(self):
        # self.img = np.ones((10, 10), dtype=np.uint8)
        self.img = np.atleast_2d(np.arange(10)).T.dot(np.atleast_2d(np.arange(10))) + 1

    def assert_row_zero(self, arr, i):
        assert_array_equal(arr[i], np.zeros(arr.shape[1]))

    def assert_col_zero(self, arr, i):
        assert_array_equal(arr[:, i], np.zeros(arr.shape[0]))

    def test_safe_crop(self):
        img_crop, delta_yx = safe_crop(self.img, (5, 5), 10)
        self.assertEqual(img_crop.shape, (10, 10))
        assert_array_equal(img_crop, self.img)
        assert_array_equal(delta_yx, (0, 0))

        img_crop, delta_yx = safe_crop(self.img, (5.5, 5.5), 10)
        self.assertEqual(img_crop.shape, (10, 10))
        assert_array_equal(img_crop[:-1, :-1], self.img[1:, 1:])
        self.assert_col_zero(img_crop, -1)
        self.assert_row_zero(img_crop, -1)
        assert_array_equal(delta_yx, (1, 1))

        img_crop, delta_yx = safe_crop(self.img, (4.5, 4.5), 10)
        self.assertEqual(img_crop.shape, (10, 10))
        assert_array_equal(img_crop, self.img)
        assert_array_equal(delta_yx, (0, 0))

        img_crop, delta_yx = safe_crop(self.img, (6, 6), 10)
        self.assertEqual(img_crop.shape, (10, 10))
        assert_array_equal(img_crop[:-1, :-1], self.img[1:, 1:])
        self.assert_col_zero(img_crop, -1)
        self.assert_row_zero(img_crop, -1)
        assert_array_equal(delta_yx, (1, 1))

        img_crop, delta_yx = safe_crop(self.img, (5, 5), 5)
        self.assertEqual(img_crop.shape, (5, 5))
        assert_array_equal(img_crop, self.img[3:8, 3:8])
        assert_array_equal(delta_yx, (3, 3))


if __name__ == '__main__':
    unittest.main()
