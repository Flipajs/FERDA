from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import math
from node import TextInfoItem

SELECTION_OFFSET = 1.5
SELECTION_LINE_WIDTH = 1
LINE_WIDTH = SELECTION_OFFSET * 2
SENSITIVITY_CONSTANT = SELECTION_OFFSET * 2

__author__ = 'Simon Mandlik'


class Edge:

    def __init__(self, from_x, from_y, to_x, to_y, core_obj, scene):
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y
        self.graphical_object = EdgeGraphical(QtCore.QLineF(from_x, from_y, to_x, to_y), core_obj, scene)
        self.core_obj = core_obj


class EdgeGraphical(QtGui.QGraphicsLineItem):
    def __init__(self, parent_line, core_obj, scene):
        super(EdgeGraphical, self).__init__(parent_line)
        self.core_obj = core_obj
        self.parent_line = parent_line
        self.setFlags(QtGui.QGraphicsItem.ItemIsSelectable)
        self.selection_offset = SELECTION_OFFSET
        self.selection_polygon = self.create_selection_polygon()
        self.pick_polygon = self.create_pick_polygon()
        self.style = core_obj[2]
        self.scene = scene

        self.clipped = False
        self.info_item = None
        self.color = None

    def show_info(self):
        self.clipped = True
        if not self.info_item:
            self.create_info()
        self.scene.addItem(self.info_item)

    def hide_info(self):
        self.scene.removeItem(self.info_item)
        self.clipped = False

    def create_info(self):
        # r = self.region

        # vertex = self.project.gm.g.vertex(int(n))
        # best_out_score, _ = self.project.gm.get_2_best_out_vertices(vertex)
        # best_out = best_out_score[0]
        #
        # best_in_score, _ = self.project.gm.get_2_best_in_vertices(vertex)
        # best_in = best_in_score[0]
        #
        # ch = self.project.gm.is_chunk(vertex)
        # ch_info = str(ch)

        # QtGui.QMessageBox.about(self, "My message box",
        #                         "Area = %i\nCentroid = %s\nMargin = %i\nAntlikeness = %f\nIs virtual: %s\nBest in = %s\nBest out = %s\nChunk info = %s" % (r.area(), str(r.centroid()), r.margin_, antlikeness, str(virtual), str(best_in_score[0])+', '+str(best_in_score[1]), str(best_out_score[0])+', '+str(best_out_score[1]), ch_info))
        x = (self.parent_line.x2() + self.parent_line.x1()) / 2
        y = (self.parent_line.y2() + self.parent_line.y1()) / 2
        self.info_item = TextInfoItem("Info there", x, y, self.color, self)
        self.info_item.setFlags(QtGui.QGraphicsItem.ItemIsMovable)

    def color_margins(self, color):
        self.color = color
        self.scene.update()

    def decolor_margins(self):
        self.color = None
        self.scene.update()

    def paint(self, painter, style_option_graphics_item, widget=None):
        opacity = 100 + 155 * abs(self.core_obj[3])
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

        if self.clipped:
            pen = QtGui.QPen(self.color, SELECTION_OFFSET, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)
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