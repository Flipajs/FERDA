from PyQt4 import Qt

__author__ = 'filip@naiser.cz'

from PyQt4 import QtGui, QtCore


class MyView(QtGui.QGraphicsView, object):
    areaSelected = QtCore.pyqtSignal("PyQt_PyObject", "PyQt_PyObject")
    clicked = QtCore.pyqtSignal("PyQt_PyObject")
    double_clicked = QtCore.pyqtSignal("PyQt_PyObject")
    mouse_moved = QtCore.pyqtSignal("PyQt_PyObject")

    def __init__(self, parent=None):
        super(MyView, self).__init__(parent)
        self.setMouseTracking(True)
        self.scale(1, 1)
        self.start_pos = None
        self.mouse_pressed = False
        self.start_position = (0, 0)
        self.setDragMode(QtGui.QGraphicsView.RubberBandDrag)
        self._isPanning = False
        self._mousePressed = False
        self._drag_pos = None
        self.selection_point_one = None
        self.selection_point_two = None
        self.last_ = ""
        self.event_click_pos = None

    def mousePressEvent(self,  event):
        self.last_ = "Click"
        self.event_click_pos = event.pos()
        if event.button() == QtCore.Qt.RightButton:
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self._drag_pos = event.pos()
            self._isPanning = True
            self.selection_point_one = event.pos()
            event.accept()
        else:
            self.selection_point_one = event.pos()
            super(MyView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(event.pos())
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
        if event.button() == QtCore.Qt.RightButton:
            self.setCursor(QtCore.Qt.ArrowCursor)
            self._isPanning = False
        else:
            if self.last_ == "Click":
                QtCore.QTimer.singleShot(QtGui.QApplication.instance().doubleClickInterval(), self.perform_single_click_action)

        super(MyView, self).mouseReleaseEvent(event)

        if self.selection_point_one is not None:
            self.selection_point_two = event.pos()
            if self.selection_point_one.x() > self.selection_point_two.x():
                tmp = self.selection_point_one.x()
                self.selection_point_one.setX(self.selection_point_two.x())
                self.selection_point_two.setX(tmp)
            if self.selection_point_one.y() > self.selection_point_two.y():
                tmp = self.selection_point_one.y()
                self.selection_point_one.setY(self.selection_point_two.y())
                self.selection_point_two.setY(tmp)
            self.areaSelected.emit(self.selection_point_one, self.selection_point_two)

    def perform_single_click_action(self):
        if self.last_ == "Click":
            self.clicked.emit(self.event_click_pos)

    def mouseDoubleClickEvent(self, event):
        super(MyView, self).mouseDoubleClickEvent(event)
        self.last_ = "Double Click"
        self.double_clicked.emit(event.pos())
        event.accept()

    def wheelEvent(self, event):
        modifiers = QtGui.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            scale_factor = 1.06

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
        center_ = QtCore.QPointF(float(x2 + x1) / 2, float(y2 + y1)/2)
        max_zoom = 20
        scale = min(self.width() / float(x2 - x1), self.height() / float(y2 - y1))
        self.zoom(min(scale, max_zoom), center_)