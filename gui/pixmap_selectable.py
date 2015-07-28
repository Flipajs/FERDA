__author__ = 'simon'

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import math

SELECTION_OFFSET = 1.5
SELECTION_LINE_WIDTH = 1
LINE_WIDTH = SELECTION_OFFSET * 2

class Pixmap_Selectable(QtGui.QGraphicsPixmapItem):


    def __init__(self, parent_pixmap, x, y):
        super(Pixmap_Selectable, self).__init__(parent_pixmap)
        self.parent_pixmap = parent_pixmap
        self.x = x
        self.y = y
        self.setFlags(QtGui.QGraphicsItem.ItemIsSelectable)
        self.selection_offset = SELECTION_OFFSET
        self.selection_polygon = self.create_selection_polygon()

    def paint(self, QPainter, QStyleOptionGraphicsItem, QWidget_widget=None):
        if self.isSelected():
            pen = QtGui.QPen(Qt.black, SELECTION_OFFSET, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin)
            QPainter.setPen(pen)
            QPainter.drawPolygon(self.selection_polygon)
            print "jsem tu"
        else:
            super(Pixmap_Selectable, self).paint(self, QPainter, QStyleOptionGraphicsItem, QWidget_widget=None)

    def create_selection_polygon(self):
        p1 = QtCore.QPointF(self.x, self.y)
        p2 = QtCore.QPointF(self.x+50, self.y)
        p3 = QtCore.QPointF(self.x, self.y+50)
        p4 = QtCore.QPointF(self.x + 500, self.y)

        polygon = QtGui.QPolygonF()
        polygon << p1 << p2 << p3 << p4
        return polygon

    def boundingRect(self):
        return self.selection_polygon.boundingRect();

    def shape(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.selection_polygon)
        return path



