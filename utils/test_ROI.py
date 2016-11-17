from unittest import TestCase
from utils.roi import ROI

class TestROI(TestCase):
    def test_expand(self):
        r = ROI(5, 8, 11, 13)
        r2 = r.expand(2)

        self.assertEqual(r2.y(), 3)
        self.assertEqual(r2.x(), 6)
        self.assertEqual(r2.height(), 15)
        self.assertEqual(r2.width(), 17)

    def test_is_intersecting(self):
        roi1 = ROI(0, 0, 10, 10)
        roi2 = ROI(5.0, 5, 1, 1)
        roi3 = ROI(10, 10, 1, 1)

        # all should be TRUE
        self.assertEqual(roi1.is_intersecting(roi2), True)
        self.assertEqual(roi3.is_intersecting(roi1), True)
        self.assertEqual(roi3.is_intersecting(roi1), True)
        self.assertEqual(roi1.is_intersecting(roi3), True)

        # SHOULD BE FALSE
        roi2 = ROI(5, 5, 1, 1)
        roi3 = ROI(10, 10, 1, 1)
        self.assertEqual(roi2.is_intersecting(roi3), False)
        self.assertEqual(roi3.is_intersecting(roi2), False)