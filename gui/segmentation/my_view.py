from PyQt4.QtGui import QMatrix

__author__ = 'filip@naiser.cz'
from PyQt4 import QtGui, QtCore

class MyView(QtGui.QGraphicsView):
    def __init__(self, update_callback_move=None, update_callback_press=None):
        super(MyView, self).__init__()
        self.setMouseTracking(True)
        self.update_callback_move = update_callback_move
        self.update_callback_press = update_callback_press
        self.scale = 1
        self.scale_step = 0
        self.scene = None

        self.matrix = QtGui.QMatrix()
        self.setMatrix(self.matrix)

    def mouseMoveEvent(self, e):
        super(MyView, self).mouseMoveEvent(e)

        if self.update_callback_move:
            if (e.buttons() & QtCore.Qt.LeftButton):
                self.update_callback_move(e)

    def setScene(self, scene):
        # for some reason, scene was not set in child widget
        super(MyView, self).setScene(scene)
        self.scene = scene

    def mousePressEvent(self, e):
        super(MyView, self).mousePressEvent(e)

        if self.update_callback_press:
            self.update_callback_press(e)


    def wheelEvent(self, event):
        if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            # if CTRL is pressed while scrolling

            # modify scale level (step)
            if event.delta() > 0:
                self.scale_step += 1
            else:
                self.scale_step -= 1

            # count new scale
            self.scale = self.get_scale()

            # create a new matrix with proper scale (for some reason, modifying self.matrix doesn't work)
            matrix = QtGui.QMatrix()
            matrix.scale(self.scale, self.scale)

            # center on mouse
            self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)

            self.setMatrix(matrix)
        else:
            # if CTRL isn't pressed, control the scrollbars (default behavior)
            super(MyView, self).wheelEvent(event)


    def get_scale(self):
        # do not scale "too far away"
        # TODO: perhaps add this as parameter to __init__?
        if self.scale_step >= 10:
            self.scale_step = 10
        elif self.scale_step <= -10:
            self.scale_step = -10

        # do not scale if step is 0
        if self.scale_step == 0:
            return 1

        # these functions represent scaling well. It isn't too fast or too slow and is convenient for the user.
        elif self.scale_step < 0:
            return -2 / (self.scale_step-2 + 0.0)

        else:
            return 2 ** (self.scale_step)

    def update_scale(self):
        # create a new matrix with proper scale (for some reason, modifying self.matrix doesn't work)
        matrix = QtGui.QMatrix()
        matrix.scale(self.scale, self.scale)

        self.setMatrix(matrix)
