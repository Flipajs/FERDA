__author__ = 'filip@naiser.cz'
from PyQt4 import QtGui, QtCore

class ArenaMark(QtGui.QGraphicsEllipseItem):
    def __init__(self, ellipse, update_labels=None, radius=5.0):
        super(ArenaMark, self).__init__(-radius/2, -radius/2, radius, radius)
        self.ellipse = ellipse
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)
        self.update_labels = update_labels

    def mouseMoveEvent(self, e):
        self.ellipse.update_geometry()

        if self.update_labels:
            self.update_labels()

        return super(ArenaMark, self).mouseMoveEvent(e)