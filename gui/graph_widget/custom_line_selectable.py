from __future__ import unicode_literals
__author__ = 'simon'

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import math

SELECTION_OFFSET = 1.5
SELECTION_LINE_WIDTH = 1
LINE_WIDTH = SELECTION_OFFSET * 2

SENSITIVITY_CONSTANT = SELECTION_OFFSET * 2

class Custom_Line_Selectable(QtGui.QGraphicsLineItem):
    def __init__(self, parent_line, style='default'):
        super(Custom_Line_Selectable, self).__init__(parent_line)
        self.parent_line = parent_line
        self.setFlags(QtGui.QGraphicsItem.ItemIsSelectable)
        self.selection_offset = SELECTION_OFFSET
        self.selection_polygon = self.create_selection_polygon()
        self.pick_polygon = self.create_pick_polygon()
        self.style = style

    def paint(self, QPainter, QStyleOptionGraphicsItem, QWidget_widget=None):
        if self.style == 'default':
            pen = QtGui.QPen(Qt.darkGray, LINE_WIDTH, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)
            QPainter.setPen(pen)
            QPainter.drawLine(self.parent_line)
            if self.isSelected():
                pen = QtGui.QPen(Qt.black, SELECTION_OFFSET, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin)
                QPainter.setPen(pen)
                QPainter.drawPolygon(self.selection_polygon)

        elif self.style == 'chunk_residual':
            pen = QtGui.QPen(QtCore.Qt.DotLine)
            pen.setColor(QtGui.QColor(255, 0, 0, 0x78))
            pen.setWidth(2)
            # pen = QtGui.QPen(Qt.darkGray, LINE_WIDTH, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)
            QPainter.setPen(pen)
            QPainter.drawLine(self.parent_line)
            if self.isSelected():
                pen = QtGui.QPen(Qt.black, SELECTION_OFFSET, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin)
                QPainter.setPen(pen)
                QPainter.drawPolygon(self.selection_polygon)

    def create_selection_polygon(self):
        pi = math.pi
        rad_angle = float(self.parent_line.angle() * pi / 180)
        dx = self.selection_offset * math.sin(rad_angle)
        dy = self.selection_offset * math.cos(rad_angle)

        offset1 = QtCore.QPointF(dx, dy)
        offset2 = QtCore.QPointF(-dx, -dy)

        polygon = QtGui.QPolygonF([self.parent_line.p1() + offset1, self.parent_line.p1() + offset2,\
                                   self.parent_line.p2() + offset2, self.parent_line.p2() + offset1])
        return polygon

    def create_pick_polygon(self):
        pi = math.pi
        rad_angle = float(self.parent_line.angle() * pi / 180)
        dx = self.selection_offset * math.sin(rad_angle)
        dy = self.selection_offset * math.cos(rad_angle)
        offset1 = QtCore.QPointF(dx * SENSITIVITY_CONSTANT, dy * SENSITIVITY_CONSTANT)
        offset2 = QtCore.QPointF(-dx * SENSITIVITY_CONSTANT, -dy * SENSITIVITY_CONSTANT)
        polygon = QtGui.QPolygonF([self.parent_line.p1() + offset1, self.parent_line.p1() + offset2,\
                                   self.parent_line.p2() + offset2, self.parent_line.p2() + offset1])
        return polygon

    def boundingRect(self):
        return self.pick_polygon.boundingRect();

    def shape(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.pick_polygon)
        return path




