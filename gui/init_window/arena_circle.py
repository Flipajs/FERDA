__author__ = 'filip@naiser.cz'

from PyQt4.QtGui import *
import math


class ArenaCircle(QGraphicsEllipseItem):
    def __init__(self):
        super(ArenaCircle, self).__init__(0, 0, 5, 5)
        self.c = None
        self.a = None
        self.is_ready = False

    def add_points(self, c_center, radius):
        self.c = c_center
        self.a = radius
        self.is_ready = True

    def update_geometry(self):
        cx = self.c.pos().x()
        cy = self.c.pos().y()
        a = math.sqrt((cx-self.a.pos().x())**2 + (cy-self.a.pos().y())**2)

        self.setRect(cx-a+10, cy-a+10, 2*a, 2*a)
