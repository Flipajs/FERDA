import unittest
# import matplotlib.pylab as plt
from shapes.polygon import Polygon
from numpy.testing import assert_array_almost_equal


class PolygonTestCase(unittest.TestCase):
    def test_area(self):
        poly = Polygon([(0, 0), (10, 0), (0, 10)])
        self.assertEqual(poly.area, 50)

    def test_xy(self):
        triangle = Polygon([(0, 0), (10, 0), (10, 10)])
        assert_array_almost_equal(triangle.xy, triangle.to_poly().mean(axis=0))

        rectangle = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        assert_array_almost_equal(rectangle.xy, (5, 5))

    def test_is_intersecting(self):
        triangle = Polygon([(0, 0), (10, 0), (10, 10)])
        rectangle = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        self.assertTrue(triangle.is_intersecting(rectangle))
        self.assertTrue(rectangle.is_intersecting(triangle))

        triangle2 = Polygon([(20, 20), (30, 20), (30, 30)])
        self.assertFalse(triangle.is_intersecting(triangle2))
        self.assertFalse(triangle2.is_intersecting(triangle))
        self.assertFalse(rectangle.is_intersecting(triangle2))
        self.assertFalse(triangle2.is_intersecting(rectangle))

        triangle3 = Polygon([(0, 0), (10, 10), (0, 10)])
        self.assertFalse(triangle.is_intersecting(triangle3))
        self.assertFalse(triangle3.is_intersecting(triangle))

        line = Polygon([(1, 1), (9, 9)])
        self.assertTrue(rectangle.is_intersecting(line))




