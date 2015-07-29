__author__ = 'simon'

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import math

SELECTION_LINE_WIDTH = 2

class Pixmap_Selectable(QtGui.QGraphicsPixmapItem):

    def __init__(self, parent_pixmap, size):
        super(Pixmap_Selectable, self).__init__(parent_pixmap)
        self.parent_pixmap = parent_pixmap
        self.x = self.parent_pixmap.offset().x()
        self.y = self.parent_pixmap.offset().y()
        self.size = size
        self.setFlags(QtGui.QGraphicsItem.ItemIsSelectable)
        self.selection_polygon = self.create_selection_polygon()

        self.clipped = False
        self.color = None

    def paint(self, QPainter, QStyleOptionGraphicsItem, QWidget_widget = None):
        self.parent_pixmap.paint(QPainter, QStyleOptionGraphicsItem, None)
        if self.isSelected():
            pen = QtGui.QPen(Qt.black, SELECTION_LINE_WIDTH, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin)
            QPainter.setPen(pen)
            QPainter.drawPolygon(self.selection_polygon)
        elif self.clipped:
                pen = QtGui.QPen(self.color, SELECTION_LINE_WIDTH, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)
                QPainter.setPen(pen)
                QPainter.drawPolygon(self.selection_polygon)
        else:
            pen = QtGui.QPen(Qt.white, SELECTION_LINE_WIDTH, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)
            QPainter.setPen(pen)
            QPainter.drawPolygon(self.selection_polygon)

    def create_selection_polygon(self):
        p1 = QtCore.QPointF(self.x, self.y)
        p2 = QtCore.QPointF(self.x + self.size, self.y)
        p4 = QtCore.QPointF(self.x, self.y + self.size)
        p3 = QtCore.QPointF(self.x + self.size, self.y + self.size)
        polygon = QtGui.QPolygonF()
        polygon << p1 << p2 << p3 << p4
        return polygon

    def boundingRect(self):
        return self.selection_polygon.boundingRect();

    def shape(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.selection_polygon)
        return path

    def setClipped(self, color):
        self.clipped = False if color == None else True
        self.color = color





