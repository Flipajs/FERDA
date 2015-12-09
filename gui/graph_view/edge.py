from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import math
from node import TextInfoItem

SELECTION_OFFSET_CHUNK = 1
SELECTION_OFFSET_LINE = 2
SELECTION_LINE_WIDTH = 2
LINE_WIDTH = 2
SENSITIVITY_CONSTANT = SELECTION_OFFSET_CHUNK * 2

__author__ = 'Simon Mandlik'


class Edge:

    def __init__(self, from_x, from_y, to_x, to_y, core_obj, scene, color=None, vertical=False):
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y
        if core_obj[2] == "chunk":
            self.graphical_object = ChunkGraphical(from_x, from_y, to_x, to_y, core_obj, scene, color, vertical)
        elif core_obj[2] == "line":
            self.graphical_object = LineGraphical(QtCore.QLineF(from_x, from_y, to_x, to_y), core_obj, scene, color)
        elif core_obj[2] == "partial":
            self.graphical_object = PartialGraphical(QtCore.QLineF(from_x, from_y, to_x, to_y), core_obj, scene, color)
        self.core_obj = core_obj


class EdgeGraphical(QtGui.QGraphicsLineItem):

    def __init__(self, parent_line, core_obj, scene, color):
        super(EdgeGraphical, self).__init__(parent_line)
        self.core_obj = core_obj
        self.parent_line = parent_line
        self.setFlags(QtGui.QGraphicsItem.ItemIsSelectable)
        self.selection_polygon = self.create_selection_polygon()
        self.pick_polygon = self.create_pick_polygon()
        self.scene = scene

        self.clipped = False
        self.shown = False
        self.info_item = None
        self.color = color
        if self.color:
            self.clipped = True
            self.shown = True

    def show_info(self, loader):
        if not self.info_item:
            self.create_info(loader)
        if not self.clipped:
            print("NAstavuji pozici")
            x, y = self.compute_rect_pos()
            self.info_item.setPos(x, y)
            self.clipped = True
        if not self.shown:
            self.scene.addItem(self.info_item)
            self.shown = True
        self.scene.update()

    def hide_info(self):
        self.scene.removeItem(self.info_item)
        self.shown = False
        # self.clipped = False

    def create_info(self, loader):
        text = loader.get_edge_info(self.core_obj)
        x, y = self.compute_rect_pos()
        self.info_item = TextInfoItem(text, x, y, self.color, self)
        self.info_item.setFlag(QtGui.QGraphicsItem.ItemSendsScenePositionChanges)
        self.info_item.setFlag(QtGui.QGraphicsItem.ItemIsMovable)

    def compute_rect_pos(self):
        x = (self.parent_line.x2() + self.parent_line.x1()) / 2
        y = (self.parent_line.y2() + self.parent_line.y1()) / 2
        return x, y

    def decolor_margins(self):
        self.color = None
        self.scene.update()

    def set_color(self, color):
        self.color = color

    def paint(self, painter, style_option_graphics_item, widget=None):
        if self.clipped:
            pen = QtGui.QPen(self.color, LINE_WIDTH, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.parent_line)
        elif self.isSelected():
            pen = QtGui.QPen(Qt.black, SELECTION_LINE_WIDTH, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPolygon(self.selection_polygon)

    def create_selection_polygon(self):
        pi = math.pi
        rad_angle = float(self.parent_line.angle() * pi / 180)
        dx = SELECTION_OFFSET_LINE * math.sin(rad_angle)
        dy = SELECTION_OFFSET_LINE * math.cos(rad_angle)

        offset1 = QtCore.QPointF(dx, dy)
        offset2 = QtCore.QPointF(-dx, -dy)

        polygon = QtGui.QPolygonF([self.parent_line.p1() + offset1, self.parent_line.p1() + offset2,
                                   self.parent_line.p2() + offset2, self.parent_line.p2() + offset1])
        return polygon

    def create_pick_polygon(self):
        pi = math.pi
        rad_angle = float(self.parent_line.angle() * pi / 180)
        dx = SELECTION_OFFSET_LINE * math.sin(rad_angle)
        dy = SELECTION_OFFSET_LINE * math.cos(rad_angle)
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


class LineGraphical(EdgeGraphical):

    def paint(self, painter, style_option_graphics_item, widget=None):
        opacity = 100 + 155 * abs(self.core_obj[3])
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, opacity), LINE_WIDTH, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(self.parent_line)

        super(LineGraphical, self).paint(painter, style_option_graphics_item, widget=None)


class PartialGraphical(EdgeGraphical):

    def paint(self, painter, style_option_graphics_item, widget=None):
        opacity = 100 + 155 * abs(self.core_obj[3])
        pen = QtGui.QPen(QtGui.QColor(0, 255, 0, opacity), LINE_WIDTH, Qt.DotLine, Qt.SquareCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(self.parent_line)

        super(PartialGraphical, self).paint(painter, style_option_graphics_item, widget=None)


class ChunkGraphical(EdgeGraphical):

    def __init__(self, from_x, from_y, to_x, to_y, core_obj, scene, color, vertical=False):
        self.parent_line = QtCore.QLineF(from_x, from_y, to_x, to_y)

        if vertical:
            self.parent_line_1 = QtCore.QLineF(from_x - LINE_WIDTH, from_y, to_x - LINE_WIDTH, to_y)
            self.parent_line_2 = QtCore.QLineF(from_x + LINE_WIDTH, from_y, to_x + LINE_WIDTH, to_y)
        else:
            self.parent_line_1 = QtCore.QLineF(from_x, from_y + LINE_WIDTH, to_x, to_y + LINE_WIDTH)
            self.parent_line_2 = QtCore.QLineF(from_x, from_y - LINE_WIDTH, to_x, to_y - LINE_WIDTH)

        self.vertical = vertical
        self.selection_polygon = self.create_selection_polygon()
        self.pick_polygon = self.create_pick_polygon()

        super(ChunkGraphical, self).__init__(self.parent_line, core_obj, scene, color)

    def paint(self, painter, style_option_graphics_item, widget=None):
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0), LINE_WIDTH, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)

        painter.setPen(pen)
        painter.drawLine(self.parent_line_1)
        painter.drawLine(self.parent_line_2)

        if self.clipped:
            pen = QtGui.QPen(self.color, LINE_WIDTH, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.parent_line_1)
            painter.drawLine(self.parent_line_2)
        elif self.isSelected():
            pen = QtGui.QPen(Qt.black, SELECTION_LINE_WIDTH, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPolygon(self.selection_polygon)

    def create_selection_polygon(self):
        pi = math.pi
        rad_angle = float(self.parent_line.angle() * pi / 180)
        dx = SELECTION_OFFSET_CHUNK * math.sin(rad_angle)
        dy = SELECTION_OFFSET_CHUNK * math.cos(rad_angle)

        offset1 = QtCore.QPointF(dx, dy)
        offset2 = QtCore.QPointF(-dx, -dy)

        polygon = QtGui.QPolygonF([self.parent_line_2.p1() + offset1, self.parent_line_1.p1() + offset2,
                                   self.parent_line_1.p2() + offset2, self.parent_line_2.p2() + offset1])
        return polygon

    def create_pick_polygon(self):
        pi = math.pi
        rad_angle = float(self.parent_line.angle() * pi / 180)
        dx = SELECTION_OFFSET_CHUNK * math.sin(rad_angle)
        dy = SELECTION_OFFSET_CHUNK * math.cos(rad_angle)
        offset1 = QtCore.QPointF(dx * SENSITIVITY_CONSTANT, dy * SENSITIVITY_CONSTANT)
        offset2 = QtCore.QPointF(-dx * SENSITIVITY_CONSTANT, -dy * SENSITIVITY_CONSTANT)
        polygon = QtGui.QPolygonF([self.parent_line_2.p1() + offset1, self.parent_line_1.p1() + offset2,
                                   self.parent_line_1.p2() + offset2, self.parent_line_2.p2() + offset1])
        return polygon
