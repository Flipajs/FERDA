__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore
from gui.init.arena.arena_circle import ArenaCircle

class MyScene(QtGui.QGraphicsScene):
    clicked = QtCore.pyqtSignal("PyQt_PyObject")
    double_clicked = QtCore.pyqtSignal("PyQt_PyObject", "PyQt_PyObject")
    mouse_moved = QtCore.pyqtSignal("PyQt_PyObject")

    def __init__ (self, parent=None):
        super(MyScene, self).__init__ (parent)

    def mouseReleaseEvent(self, event):
        super(MyScene, self).mouseReleaseEvent(event)
        self.clicked.emit(event.scenePos())

    def mouseMoveEvent(self, event):
        super(MyScene, self).mouseMoveEvent(event)
        self.mouse_moved.emit(event.scenePos())

    def mouseDoubleClickEvent(self, event):
        super(MyScene, self).mouseDoubleClickEvent(event)
        items = self.items(event.pos())

        self.double_clicked.emit(event.pos(), items)