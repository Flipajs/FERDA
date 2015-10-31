from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import math

SELECTION_OFFSET = 1.5
SELECTION_LINE_WIDTH = 1
LINE_WIDTH = SELECTION_OFFSET * 2
SENSITIVITY_CONSTANT = SELECTION_OFFSET * 2

__author__ = 'Simon Mandlik'


class Edge:

    def __init__(self, from_x, from_y, to_x, to_y, core_obj):
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y
        self.graphical_object = EdgeGraphical(QtCore.QLineF(from_x, from_y, to_x, to_y), core_obj)
        self.core_obj = core_obj


class EdgeGraphical(QtGui.QGraphicsLineItem):
    def __init__(self, parent_line, core_obj):
        super(EdgeGraphical, self).__init__(parent_line)
        self.core_obj = core_obj
        self.parent_line = parent_line
        self.setFlags(QtGui.QGraphicsItem.ItemIsSelectable)
        self.selection_offset = SELECTION_OFFSET
        self.selection_polygon = self.create_selection_polygon()
        self.pick_polygon = self.create_pick_polygon()
        self.style = core_obj[2]

    def paint(self, painter, style_option_graphics_item, widget=None):
        opacity = 100 + 155 * float(self.core_obj[3])
        if self.style == 'chunk':
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0, opacity), LINE_WIDTH,
                             Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)
        elif self.style == 'line':
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0, opacity), LINE_WIDTH / 1.5,
                             Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)
        else:
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0, opacity), LINE_WIDTH / 1.5,
                             Qt.DotLine, Qt.SquareCap, Qt.RoundJoin)

        painter.setPen(pen)
        painter.drawLine(self.parent_line)

        if self.isSelected():
            pen = QtGui.QPen(Qt.black, SELECTION_OFFSET, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPolygon(self.selection_polygon)

    def create_selection_polygon(self):
        pi = math.pi
        rad_angle = float(self.parent_line.angle() * pi / 180)
        dx = self.selection_offset * math.sin(rad_angle)
        dy = self.selection_offset * math.cos(rad_angle)

        offset1 = QtCore.QPointF(dx, dy)
        offset2 = QtCore.QPointF(-dx, -dy)

        polygon = QtGui.QPolygonF([self.parent_line.p1() + offset1, self.parent_line.p1() + offset2,
                                   self.parent_line.p2() + offset2, self.parent_line.p2() + offset1])
        return polygon

    def create_pick_polygon(self):
        pi = math.pi
        rad_angle = float(self.parent_line.angle() * pi / 180)
        dx = self.selection_offset * math.sin(rad_angle)
        dy = self.selection_offset * math.cos(rad_angle)
        offset1 = QtCore.QPointF(dx * SENSITIVITY_CONSTANT, dy * SENSITIVITY_CONSTANT)
        offset2 = QtCore.QPointF(-dx * SENSITIVITY_CONSTANT, -dy * SENSITIVITY_CONSTANT)
        polygon = QtGui.QPolygonF([self.parent_line.p1() + offset1, self.parent_line.p1() + offset2,
                                   self.parent_line.p2() + offset2, self.parent_line.p2() + offset1])
        return polygon

    def boundingRect(self):
        return self.pick_polygon.boundingRect()

    def shape(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.pick_polygon)
        return path