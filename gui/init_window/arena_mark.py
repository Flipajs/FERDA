__author__ = 'filip@naiser.cz'
from PyQt4.QtGui import *


class ArenaMark(QGraphicsEllipseItem):
    def __init__(self, ellipse, update_labels):
        super(ArenaMark, self).__init__(0, 0, 5, 5)
        self.ellipse = ellipse
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.update_labels = update_labels

    def mouseMoveEvent(self, e):
        super(ArenaMark, self).mouseMoveEvent(e)
        self.ellipse.update_geometry()
        self.update_labels()

    def MouseReleaseEvent(self, e):
        super(ArenaMark, self).mouseMoveEvent(e)
        self.ellipse.update_geometry()
        self.update_labels()
