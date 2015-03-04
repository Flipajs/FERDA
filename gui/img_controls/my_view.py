from PyQt4 import Qt

__author__ = 'filip@naiser.cz'

#from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
import PyQt4.Qt
from PyQt4 import QtGui
from PyQt4.QtCore import *


class MyView(QGraphicsView, object):
    def __init__(self, parent=None):
        super(MyView, self).__init__(parent)
        self.setMouseTracking(True)
        self.scale(1, 1)
        self.start_pos = None
        self.mouse_pressed = False
        self.start_position = (0, 0)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self._isPanning = False
        self._mousePressed = False
        self._drag_pos = None

    def mousePressEvent(self,  event):
        if event.button() == Qt.RightButton:
            self.setCursor(Qt.ClosedHandCursor)
            self._drag_pos = event.pos()
            self._isPanning = True
            event.accept()
        else:
            super(MyView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._isPanning:
            newPos = event.pos()
            diff = newPos - self._drag_pos
            self._drag_pos = newPos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - diff.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - diff.y())
            event.accept()
        else:
            super(MyView, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.setCursor(Qt.ArrowCursor)
            self._isPanning = False
        super(MyView, self).mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        pass

    def wheelEvent(self, event):
        scale_factor = 1.12

        self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)

        m11 = self.transform().m11()
        m22 = self.transform().m22()

        if event.delta() > 0:
            # max zoom-out restriction
            if m11 > 10 or m22 > 10:
                return

            self.scale(scale_factor, scale_factor)
        else:
            # max zoom-in restriction
            if m11 < 0.1 or m22 < 0.1:
                return

            self.scale(1.0 / scale_factor, 1.0 / scale_factor)

    def zoom(self, factor, center_point):
        self.scale(factor, factor)
        self.centerOn(center_point)

    def zoom_into(self, x1, y1, x2, y2):
        center_ = QPointF(float(x2 + x1) / 2, float(y2 + y1)/2)
        max_zoom = 20
        scale = min(self.width() / float(x2 - x1), self.height() / float(y2 - y1))
        self.zoom(min(scale, max_zoom), center_)