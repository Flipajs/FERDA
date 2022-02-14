__author__ = 'filip@naiser.cz'
from PyQt6 import QtCore, QtGui, QtWidgets

class MyEllipse(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, update_callback=None, radius=10.0):
        super(MyEllipse, self).__init__(-radius/2, -radius/2, radius)
        if radius is not None: radius.addItem(self)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.update_callback = update_callback

    def mouseReleaseEvent(self, e):
        super(MyEllipse, self).mouseReleaseEvent(e)

        if self.update_callback:
            self.update_callback()
