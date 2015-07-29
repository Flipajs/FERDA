__author__ = 'simon'

from PyQt4 import QtGui, QtCore

class MyViewZoomable(QtGui.QGraphicsView, object):
    
    def __init__(self):
        super(MyViewZoomable, self).__init__()
        self.num_scalings = 0

    # def mousePressEvent(self,  event):
    #     pass
    #
    # def mouseDoubleClickEvent(self, QMouseEvent):
    #     pass
    #
    # def mouseReleaseEvent(self, QMouseEvent):
    #     pass

    def keyPressEvent(self, event):
        key_i = QtCore.Qt.Key_I
        key_o = QtCore.Qt.Key_O

        scale_factor = 1.5
        self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)

        m11 = self.transform().m11()
        m22 = self.transform().m22()

        if event.key() == key_o:
            if m11 < 0.1 or m22 < 0.1:
                return

            self.scale(1.0 / scale_factor, 1.0 / scale_factor)
        elif event.key() == key_i:
            if m11 < 0.1 or m22 < 0.1:
                return

            self.scale(scale_factor, scale_factor)

    # def scalingTime(self, x):
    #     factor = 1.0 + self.num_scalings / 300
    #     self.scale(factor, factor)
    #
    # def animFinished(self):
    #     if (self.num_scalings > 0):
    #         self.num_scalings -= 1
    #     else:
    #         self.num_scalings += 1

    # def keyPressEvent(self, event):
    #     key_i = QtCore.Qt.Key_I
    #     key_o = QtCore.Qt.Key_O
    #
    #     self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
    #
    #     m11 = self.transform().m11()
    #     m22 = self.transform().m22()
    #
    #     elif event.key() == key_o:
    #         if m11 < 0.1 or m22 < 0.1:
    #             return
    #
    #         self.scale(1.0 / scale_factor, 1.0 / scale_factor)
    #
    #     if event.key() == key_i:
    #         num = event.delta()
    #         steps = num / 8
    #         self.num_scalings += steps
    #         if(self.num_scalings * steps < 0):
    #             self.num_scalings = steps
    #
    #         time_line = QtCore.QTimeLine(350, self)
    #         time_line.setUpdateInterval(20)
    #         time_line.valueChanged.connect(self.scalingTime)
    #         time_line.finished.connect(self.animFinished)
    #         time_line.start()
    #     elif event.key() == key_o:
    #         pass
    #     else:
    #         super(MyViewZoomable, self).keyPressEvent(event)
    #
