__author__ = 'filip@naiser.cz'
from PyQt4 import QtGui, QtCore

class MyView(QtGui.QGraphicsView):
    def __init__(self, update_callback_move=None, update_callback_release=None):
        super(MyView, self).__init__()
        self.setMouseTracking(True)
        self.update_callback_move = update_callback_move
        self.update_callback_release = update_callback_release

    def mouseMoveEvent(self, e):
        super(MyView, self).mouseMoveEvent(e)

        if self.update_callback_move:
            if (e.buttons() & QtCore.Qt.LeftButton):
                self.update_callback_move(e)

    #def mouseReleaseEvent(self, e):