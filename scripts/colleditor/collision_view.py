__author__ = 'filip@naiser.cz'

from gui.img_controls.my_view import MyView
from PyQt6.QtCore import *
from PyQt6 import QtCore, QtGui, QtWidgets


class CollisionView(MyView):

    areaSelected = pyqtSignal("PyQt_PyObject", "PyQt_PyObject")

    def __init__(self, parent=None):
        super(CollisionView, self).__init__(parent)
        self.setDragMode(self.NoDrag)
        self.point_one = None
        self.point_two = None

        self.mouse_move_last_pos = None

        self.drawing = False
        self.continuous_move = False

        self.pts = None
        self.drawing_handler = None

    def set_drawing_mode(self, drawing_handler):
        self.drawing = not self.drawing

        if self.drawing:
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(Qt.CursorShape.CrossCursor))
            self.drawing_handler = drawing_handler
        else:
            QtWidgets.QApplication.restoreOverrideCursor()

    def mouseMoveEvent(self, event):
        super(CollisionView, self).mouseMoveEvent(event)

        p = self.mapToScene(event.pos())
        pos = [p.x(), p.y()]

        if self.mouse_move_last_pos != pos:
            if self.drawing:
                if self.continuous_move:
                    pts = bresenham(pos, self.mouse_move_last_pos)
                    self.draw(event, pts)

            self.mouse_move_last_pos = pos

    def draw(self, event, pts):
        modifiers = QtWidgets.QApplication.keyboardModifiers()

        if event.buttons() == Qt.MouseButton.LeftButton:
            if modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
                self.drawing_handler(pts, True)
            else:
                self.drawing_handler(pts)

    def mousePressEvent(self,  event):
        super(CollisionView, self).mousePressEvent(event)
        self.continuous_move = True

        if self.drawing:
            p = self.mapToScene(event.pos())
            pos = [int(p.x()), int(p.y())]
            self.draw(event, [pos])

    def mouseReleaseEvent(self, event):
        super(CollisionView, self).mouseReleaseEvent(event)
        self.continuous_move = False


def bresenham(pt1, pt2):
    x = int(pt1[0])
    y = int(pt1[1])
    x2 = int(pt2[0])
    y2 = int(pt2[1])
    """Brensenham line algorithm"""
    steep = 0
    coords = []
    dx = abs(x2 - x)
    if (x2 - x) > 0: sx = 1
    else: sx = -1
    dy = abs(y2 - y)
    if (y2 - y) > 0: sy = 1
    else: sy = -1
    if dy > dx:
        steep = 1
        x,y = y,x
        dx,dy = dy,dx
        sx,sy = sy,sx
    d = (2 * dy) - dx
    for i in range(0,dx):
        if steep: coords.append((y,x))
        else: coords.append((x,y))
        while d >= 0:
            y = y + sy
            d = d - (2 * dx)
        x = x + sx
        d = d + (2 * dy)
    return coords
