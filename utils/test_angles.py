from __future__ import unicode_literals
import unittest
from numpy.testing import assert_array_equal
import numpy as np
from utils.angles import angle_absolute_error, angle_absolute_error_direction_agnostic


class AnglesTestCase(unittest.TestCase):
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
        self.assertEqual(angle_absolute_error_direction_agnostic(-30, 300, np), 30)
        assert_array_equal(angle_absolute_error_direction_agnostic(np.array([10, 0, 0]),
                                                                           np.array([300, 360, -10]), np),
                           np.array([70, 0, 10]))


if __name__ == '__main__':
    unittest.main()
