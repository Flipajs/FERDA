import unittest
import matplotlib.pylab as plt
from shapes.bbox import BBox
from numpy.testing import assert_array_equal


class BBoxTestCase(unittest.TestCase):
    def test_bounds(self):
        bbox = BBox(10, 10, 20, 20)
        self.assertTrue(bbox.is_inside_bounds(0, 0, 30, 30))
        self.assertFalse(bbox.is_inside_bounds(30, 30, 40, 40))
        self.assertTrue(bbox.is_partially_outside_bounds(11, 11, 30, 30))
        self.assertTrue(bbox.is_partially_outside_bounds(12, 0, 13, 15))
        self.assertFalse(bbox.is_partially_outside_bounds(30, 30, 40, 40))  # strictly outside
        self.assertFalse(bbox.is_partially_outside_bounds(0, 0, 30, 30))  # inside
        self.assertTrue(bbox.is_strictly_outside_bounds(30, 30, 40, 40))
        self.assertFalse(bbox.is_strictly_outside_bounds(15, 15, 30, 30))
        
    def test_bounds_bbox(self):
        bbox = BBox(10, 10, 20, 20)
        self.assertTrue(bbox.is_inside_bbox(BBox(0, 0, 30, 30)))
        self.assertFalse(bbox.is_inside_bbox(BBox(30, 30, 40, 40)))
        self.assertTrue(bbox.is_partially_outside_bbox(BBox(11, 11, 30, 30)))
        self.assertTrue(bbox.is_partially_outside_bbox(BBox(12, 0, 13, 15)))
        self.assertFalse(bbox.is_partially_outside_bbox(BBox(30, 30, 40, 40)))  # strictly outside
        self.assertFalse(bbox.is_partially_outside_bbox(BBox(0, 0, 30, 30)))  # inside
        self.assertTrue(bbox.is_strictly_outside_bbox(BBox(30, 30, 40, 40)))
        self.assertFalse(bbox.is_strictly_outside_bbox(BBox(15, 15, 30, 30)))

    def test_intersection(self):
        intersection = BBox(0, 0, 20, 20).intersection(BBox(10, 10, 30, 30))  # standard intersection
        assert_array_equal(intersection.to_array()[:4], [10, 10, 20, 20])

        intersection = BBox(0, 0, 20, 20).intersection(BBox(-5, -5, 10, 10))  # standard intersection
        assert_array_equal(intersection.to_array()[:4], [0, 0, 10, 10])

        intersection = BBox(0, 0, 20, 20).intersection(BBox(5, 5, 10, 10))  # one inside other
        assert_array_equal(intersection.to_array()[:4], [5, 5, 10, 10])

        intersection = BBox(5, 5, 10, 10).intersection(BBox(0, 0, 20, 20))  # one inside other
        assert_array_equal(intersection.to_array()[:4], [5, 5, 10, 10])

        intersection = BBox(5, 5, 10, 10).intersection(BBox(20, 20, 30, 30))  # no intersection
        self.assertIsNone(intersection)

    def test_cut(self):
        viewport = BBox(0, 0, 10, 10)

        cut = BBox(1, 1, 2, 2).cut(viewport)  # completely inside viewport
        assert_array_equal(cut.to_array()[:4], [1, 1, 2, 2])

        cut = BBox(15, 15, 20, 20).cut(viewport)  # completely outside viewport
        self.assertIsNone(cut)

        cut = BBox(5, 5, 15, 15).cut(viewport)  # cut both sides
        assert_array_equal(cut.to_array()[:4], [5, 5, 10, 10])

        cut = BBox(5, 5, 15, 7).cut(viewport)  # cut one side
        assert_array_equal(cut.to_array()[:4], [5, 5, 10, 7])
