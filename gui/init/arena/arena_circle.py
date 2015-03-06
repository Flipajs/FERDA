__author__ = 'filip@naiser.cz'

from PyQt4.QtGui import *
import math
from PyQt4 import QtCore


class ArenaCircle(QGraphicsEllipseItem):
    double_clicked = QtCore.pyqtSignal("PyQt_PyObject")

    def __init__(self):
        super(ArenaCircle, self).__init__()
        self.c = None
        self.a = None
        self.is_ready = False

    def add_points(self, c_center, radius):
        self.c = c_center
        self.a = radius
        self.is_ready = True

    def update_geometry(self):
        a = self.radius()
        if a <= 0:
            a = 1

        self.setRect(self.c.x()-a, self.c.y()-a, 2*a, 2*a)

    def radius(self):
        return math.sqrt((self.c.x()-self.a.x())**2 + (self.c.y()-self.a.y())**2)

    def mouseReleaseEvent(self, event):
        print "TEST"

    def mouseDoubleClickEvent(self, event):
        super(ArenaCircle, self)

        print "DOUBLE CLICKED"
        self.double_clicked.emit(id)
        pass
