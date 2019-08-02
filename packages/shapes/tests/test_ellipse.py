import unittest

import matplotlib.pylab as plt
from shapes.ellipse import Ellipse


class EllipseTestCase(unittest.TestCase):

    def test_get_vertices(self):
        el = Ellipse(10, 10, angle_deg=20, major=10, minor=3, frame=0)
        el.draw()
        v1, v2 = el.get_vertices()
        plt.plot(v1[0], v1[1], 'r*')
        plt.plot(v2[0], v2[1], 'b*')
        plt.gca().invert_yaxis()
        # plt.show()

    def test_get_point(self):
        el = Ellipse(100, 100, angle_deg=20, major=50, minor=10, frame=0)
        el.draw()
        xy = el.get_point(180)
        plt.plot(*xy, color='g', marker='*')
        xy = el.get_point(-20)
        plt.plot(*xy, color='b', marker='*')
        plt.gca().invert_yaxis()
        # plt.show()


if __name__ == '__main__':
    unittest.main()
