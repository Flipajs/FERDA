from PyQt4 import Qt

__author__ = 'flipajs'

#from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
import PyQt4.Qt
from PyQt4.QtCore import *

class MyView(QGraphicsView):
    def __init__(self, parent=None):
        super(MyView, self).__init__(parent)
        self.setMouseTracking(True)
        self.scale(1,1)
        self.startPos = None
        self.mouse_pressed = False
        self.start_position = (0, 0)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self._isPanning = False
        self._mousePressed = False


    def mousePressEvent(self,  event):
        if event.button() == Qt.LeftButton:
            self._mousePressed = True
            if self._isPanning:
                self.setCursor(Qt.ClosedHandCursor)
                self._dragPos = event.pos()
                event.accept()
            else:
                super(MyView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._mousePressed and self._isPanning:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - diff.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - diff.y())
            event.accept()
        else:
            super(MyView, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self._isPanning = False
                self.setCursor(Qt.ArrowCursor)
            self._mousePressed = False
        super(MyView, self).mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event): pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control and not self._mousePressed:
            self._isPanning = True
            self.setCursor(Qt.OpenHandCursor)
        else:
            super(MyView, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            if not self._mousePressed:
                self._isPanning = False
                self.setCursor(Qt.ArrowCursor)
        else:
            super(MyView, self).keyPressEvent(event)


    def wheelEvent(self, event):
        scale_factor = 1.15
        numSteps = event.delta() / 15 / 8

        if numSteps == 0:
            event.ignore()
            return

        sc = 1.25**numSteps

        if sc < 0.5:
            return

        self.zoom(sc, self.mapToScene(event.pos()))
    #
    #def mousePressEvent(self, e):
    #    if not self.mouse_pressed:
    #        self.mouse_pressed = True
    #        self.start_position = (e.pos().x(), e.pos().y())
    #        print self.start_position
    #
    #def mouseReleaseEvent(self, e):
    #    self.mouse_pressed = False
    #    print self.mouse_pressed
    #
    #def mouseMoveEvent(self, e):
    #    self.setTransformationAnchor( QtGui.QGraphicsView.NoAnchor )
    #    if self.mouse_pressed:
    #        ex = e.pos().x()
    #        ey = e.pos().y()
            #x = ex - self.start_position[0]
            #y = ey - self.start_position[1]
            #self.start_position = (ex, ey)
            #self.translate(ex, ey)

    def zoom(self, factor, center_point):
        self.scale(factor, factor)
        self.centerOn(center_point)
