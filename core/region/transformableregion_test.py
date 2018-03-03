import unittest
import core.region.transformableregion as tr
import numpy as np
import matplotlib.pylab as plt
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd


class TransformableRegionTestCase(unittest.TestCase):
    def test_p2e(self):
        proj2d = np.array([[1., 2., 3.]]).T
        eucl2d = tr.p2e(proj2d)
        self.assertEqual(eucl2d.size, 2)
        self.assertEqual(eucl2d[0], 1. / 3)

        eucl2d_2x = tr.p2e(np.hstack((proj2d, proj2d)))
        self.assertEqual(eucl2d_2x.shape, (2, 2))

        proj3d = np.array([[1., 2., 3., 4.]]).T
        eucl3d = tr.p2e(proj3d)
        self.assertEqual(eucl3d.size, 3)
        self.assertEqual(eucl3d[0], 1. / 4)

        eucl3d_2x = tr.p2e(np.hstack((proj3d, proj3d)))
        self.assertEqual(eucl3d_2x.shape, (3, 2))

        assert_array_equal(tr.p2e(np.array([2, 2, 1])), np.array([2, 2]))
        assert_array_equal(tr.p2e(np.array([2, 2, 2])), np.array([1, 1]))

    def test_e2p(self):
        euclid_2d = np.array([[1., 2.]]).T
        proj_2d = tr.e2p(euclid_2d)
        self.assertEqual(proj_2d.shape, (3, 1))
        assert_array_equal(proj_2d, np.vstack((euclid_2d, 1)))

        euclid_3d = np.array([[1., 2., 3.]]).T
        proj_3d = tr.e2p(euclid_3d)
        self.assertEqual(proj_3d.shape, (4, 1))
        assert_array_equal(proj_3d, np.vstack((euclid_3d, 1)))

        assert_array_equal(tr.e2p(np.array([2, 2])), np.array([2, 2, 1]))


if __name__ == '__main__':
    unittest.main()