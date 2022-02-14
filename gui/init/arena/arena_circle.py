__author__ = 'filip@naiser.cz'

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import math
from PyQt5 import QtCore

# class QGraphicsItemEmitter(QtCore.QObject):
#     clicked = QtCore.pyQtSignal(int)
#     double_clicked = QtCore.pyQtSignal(int)
#
#     def __init__(self):
#         super(QGraphicsItemEmitter, self).__init__()


class ArenaCircle(QGraphicsEllipseItem):
    def __init__(self, id=-1):
        super(ArenaCircle, self).__init__()
        # QtGui.QGraphicsObject.__init__(self)
        self.c = None
        self.a = None
        self.is_ready = False
        self.click_num = 0
        self.id = id

        # self.emitter = QGraphicsItemEmitter()

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
    #
    # def mousePressEvent(self, event):
    #     print "PRESS"
    #
    # def mouseDoubleClickEvent(self, event):
    #     # QtGui.QGraphicsItem.mouseDoubleClickEvent(event)
    #     print "DOUBLE"

    # def mousePressEvent(self, event):
    #     super(ArenaCircle, self).mousePressEvent(event)
    #     # self.click_num += 1
    #     # QtCore.QTimer.singleShot(QtGui.QApplication.instance().doubleClickInterval(), self.test_double_click)
    #
    # def mouseReleaseEvent(self, event):
    #     super(ArenaCircle, self).mouseReleaseEvent(event)
    #
    # def mouseDoubleClickEvent(self, event):
    #     super(ArenaCircle, self).mouseDoubleClickEvent(event)
    #     print "DOUBLE CLICK"
    #     self.emitter.double_clicked.emit()
    #     # self.double_clicked.emit()
    #
    # def test_double_click(self):
    #     if self.click_num >= 2:
    #         print "DOUBLE CLICKED"
    #         self.double_clicked.emit()
    #
    #     self.click_num = 0
