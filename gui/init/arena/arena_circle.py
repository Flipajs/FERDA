__author__ = 'filip@naiser.cz'

from PyQt4.QtGui import *
import math
from PyQt4 import QtGui, QtCore


class ArenaCircle(QGraphicsEllipseItem):
    double_clicked = QtCore.pyqtSignal("int")

    def __init__(self, id=-1):
        super(ArenaCircle, self).__init__()
        self.c = None
        self.a = None
        self.is_ready = False
        self.click_num = 0
        self.id = id


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
        self.click_num += 1
        QtCore.QTimer.singleShot(QtGui.QApplication.instance().doubleClickInterval(), self.test_double_click)

    def test_double_click(self):
        if self.click_num >= 2:
            print "DOUBLE CLICKED"
            self.double_clicked.emit(self.id)

        self.click_num = 0