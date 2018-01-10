import unittest
import numpy as np
from scripts.CNN.train_interactions import angle_absolute_error


class LossFunctionsTestCase(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()