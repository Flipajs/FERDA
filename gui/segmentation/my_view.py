from PyQt6.QtGui import QTransform

__author__ = 'filip@naiser.cz'
from PyQt6 import QtCore, QtGui, QtWidgets

class MyView(QtWidgets.QGraphicsView):
    def __init__(self, update_callback_move=None, update_callback_press=None):
        super(MyView, self).__init__()
        self.setMouseTracking(True)
        self.update_callback_move = update_callback_move
        self.update_callback_press = update_callback_press
        self.scale(1, 1)
        self.scale_step = 0
        self.scene = None

        self.matrix = QtGui.QTransform()
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
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            scale_factor = 1.06

            self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

            m11 = self.transform().m11()
            m22 = self.transform().m22()

            if event.angleDelta().y() > 0:
                # max zoom-out restriction
                if m11 > 10 or m22 > 10:
                    return

                self.scale(scale_factor, scale_factor)
            else:
                # max zoom-in restriction
                if m11 < 0.1 or m22 < 0.1:
                    return

                self.scale(1.0 / scale_factor, 1.0 / scale_factor)
    #
    # def wheelEvent(self, event):
    #     if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
    #         # if CTRL is pressed while scrolling
    #
    #         # modify scale level (step)
    #         if event.delta() > 0:
    #             self.scale_step += 1
    #         else:
    #             self.scale_step -= 1
    #
    #         # count new scale
    #         self.scale = self.get_scale()
    #
    #         # create a new matrix with proper scale (for some reason, modifying self.matrix doesn't work)
    #         matrix = QtGui.QMatrix()
    #         matrix.scale(self.scale, self.scale)
    #
    #         # center on mouse
    #         self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
    #
    #         self.setMatrix(matrix)
    #     else:
    #         # if CTRL isn't pressed, control the scrollbars (default behavior)
    #         super(MyView, self).wheelEvent(event)


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
        matrix = QtGui.QTransform()
        matrix.scale(self.scale, self.scale)

        self.setMatrix(matrix)
