from core.region.shape import Shape
import numpy as np


class Point(Shape):
    def __init__(self, x=None, y=None, frame=None):
        super(Point, self).__init__(frame)
        self.x = x
        self.y = y

    @property
    def xy(self):
        return np.array((self.x, self.y))

    @xy.setter
    def xy(self, xy):
        self.x = xy[0]
        self.y = xy[1]

    def is_outside_bounds(self, x1, y1, x2, y2):
        return self.x < x1 or self.y < y1 or self.x > x2 or self.y > y2

    def move(self, delta_xy):
        self.xy += delta_xy
        return self

    def draw(self, ax=None, label=None, color=None):
        import matplotlib.pylab as plt
        if color is None:
            color = 'r'
        plt.scatter(self.x, self.y, c=color, label=label)

