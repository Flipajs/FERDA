from PyQt4.QtGui import QMatrix

__author__ = 'filip@naiser.cz'
from PyQt4 import QtGui, QtCore

class MyView(QtGui.QGraphicsView):
    def __init__(self, update_callback_move=None, update_callback_release=None):
        super(MyView, self).__init__()
        self.setMouseTracking(True)
        self.update_callback_move = update_callback_move
        self.update_callback_release = update_callback_release
        self.scale = 1
        self.scale_step = 0

        self.matrix = QtGui.QMatrix()
        self.setMatrix(self.matrix)

    def mouseMoveEvent(self, e):
        super(MyView, self).mouseMoveEvent(e)

        if self.update_callback_move:
            if (e.buttons() & QtCore.Qt.LeftButton):
                self.update_callback_move(e)


    def wheelEvent(self, event):
        if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            if event.delta() > 0:
                self.scale_step += 1
            else:
                self.scale_step -= 1

            self.scale = self.get_scale()
            print self.scale_step, self.scale

            matrix = QtGui.QMatrix()
            matrix.scale(self.scale, self.scale)
            self.setMatrix(matrix)
        else:
            super(MyView, self).wheelEvent(event)

    def get_scale(self):
        if self.scale_step >= 10:
            self.scale_step = 10
        elif self.scale_step <= -10:
            self.scale_step = -10

        if self.scale_step == 0:
            return 1

        elif self.scale_step < 0:
            # return (self.scale_step ** 2) / 4.0
            return -2 / (self.scale_step-2 + 0.0)

        else:
            return 2 ** (self.scale_step)



    #def mouseReleaseEvent(self, e):