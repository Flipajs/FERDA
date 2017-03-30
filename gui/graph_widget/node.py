from PyQt4 import QtGui, QtCore
import numpy as np
from PyQt4.QtCore import Qt

from gui.graph_widget_loader import WIDTH, HEIGHT
from gui.img_controls.gui_utils import cvimg2qtpixmap

SELECTION_LINE_WIDTH = 2
DEFAULT_INFO_TEXT_OPACITY = 150

__author__ = 'Simon Mandlik'


class Node(QtGui.QGraphicsPixmapItem):

    def __init__(self, parent_pixmap, scene, region, img_manager, relative_margin, width, height):
        super(Node, self).__init__(parent_pixmap)
        self.region = region
        self.img_manager = img_manager
        self.scene = scene
        self.parent_pixmap = parent_pixmap
        self.pixmap_toggled = None
        self.width = width
        self.height = height
        self.relative_margin = relative_margin
        self.x = self.parent_pixmap.offset().x()
        self.y = self.parent_pixmap.offset().y()
        self.setFlags(QtGui.QGraphicsItem.ItemIsSelectable)
        self.selection_polygon = self.create_selection_polygon()

        self.pixmapped = False
        self.toggled = False
        self.clipped = False
        self.shown = False
        self.color = None

        self.info_item = None
        self.info_rect = None

    def toggle(self):
        if not self.toggled:
            if self.pixmap_toggled is None:
                self.create_zoom_pixmap()
            else:
                self.scene.addItem(self.pixmap_toggled)
        else:
            self.scene.removeItem(self.pixmap_toggled)
        self.toggled = False if self.toggled else True

    def show_info(self, loader):
        if not self.info_item or not self.clipped:
            self.create_info(loader)
            self.clipped = True
            self.info_item.set_color(self.color)
        if not self.shown:
            self.scene.addItem(self.info_item)
            self.shown = True
        self.scene.update()

    def hide_info(self):
        self.scene.removeItem(self.info_item)
        self.shown = False
        # self.clipped = False

    def create_info(self, loader):
        text = loader.get_node_info(self.region)
        metrics = QtGui.QFontMetrics(QtGui.QFont())
        longest, rows = get_longest_string_rows(text)
        width = metrics.width(longest)
        height = metrics.height() * (rows + 0.5)
        multiplier_x = 0 if self.x < self.scene.width() / 2 else -1
        multiplier_y = 0 if self.y < self.scene.height() / 2 else -1
        parent_x, parent_y = self.compute_info_rectangle_pos()
        x = parent_x + multiplier_x * width
        y = parent_y + multiplier_y * height
        self.info_item = TextInfoItem(text, x, y, width, height, self.color, self)
        self.info_item.set_parent_point(parent_x, parent_y)
        self.info_item.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.info_item.setFlag(QtGui.QGraphicsItem.ItemSendsScenePositionChanges)

    def set_color(self, color):
        self.color = color

    def decolor_margins(self):
        self.color = None
        self.scene.update()

    def create_zoom_pixmap(self):
        img_toggled = self.img_manager.get_crop(self.region.frame_, self.region, width=self.width * 3, height=self.height * 3, relative_margin=self.relative_margin)
        pixmap = cvimg2qtpixmap(img_toggled)
        self.pixmap_toggled = self.scene.addPixmap(pixmap)
        x, y = self.compute_toggle_rectangle_pos()
        self.pixmap_toggled.setPos(x, y)
        self.pixmap_toggled.setFlags(QtGui.QGraphicsItem.ItemIsMovable)

    def create_pixmap(self):
        if not self.pixmapped:
            self.pixmapped = True
            img = self.img_manager.get_crop(self.region.frame_, self.region, width=self.width,
                                                    height=self.height, relative_margin=self.relative_margin)
            pixmap = cvimg2qtpixmap(img)
            pixmap = self.scene.addPixmap(pixmap)
            self.setParentItem(pixmap)
            self.scene.removeItem(self.parent_pixmap)
            self.parent_pixmap = pixmap
            self.parent_pixmap.setPos(self.x, self.y)

    def compute_rectangle_size(self):
        width, height = self.scene.width(), self.scene.height()
        multiplier_x = 1 if self.x < width / 2 else 0
        multiplier_y = 1 if self.y < height / 2 else 0
        return multiplier_x, multiplier_y

    def compute_info_rectangle_pos(self):
        multiplier_x, multiplier_y = self.compute_rectangle_size()
        return self.x + (multiplier_x) * self.width, self.y + (multiplier_y) * self.height

    def compute_toggle_rectangle_pos(self):
        multiplier_x, multiplier_y = self.compute_rectangle_size()
        if multiplier_x == 0:
            multiplier_x = -3
        if multiplier_y == 0:
            multiplier_y = -3
        return self.x + (multiplier_x) * self.width, self.y + (multiplier_y) * self.height

    def setPos(self, x, y):
        if not(x == self.x and y == self.y):
            self.x, self.y = x, y
            self.parent_pixmap.setPos(x, y)
            if self.pixmap_toggled is not None:
                self.pixmap_toggled.setPos(x + WIDTH / 2, y + HEIGHT / 2)

    def paint(self, painter, style_option_graphics_item, widget=None):
        # self.parent_pixmap.paint(painter, style_option_graphics_item, None)
        if self.clipped:
            pen = QtGui.QPen(self.color, SELECTION_LINE_WIDTH * 1.5, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)
        elif self.isSelected():
            pen = QtGui.QPen(Qt.black, SELECTION_LINE_WIDTH, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin)
        else:
            pen = QtGui.QPen(Qt.white, SELECTION_LINE_WIDTH, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)

        painter.setPen(pen)
        painter.drawPolygon(self.selection_polygon)

    def create_selection_polygon(self):
        p1 = QtCore.QPointF(self.x, self.y)
        p2 = QtCore.QPointF(self.x + self.width, self.y)
        p4 = QtCore.QPointF(self.x, self.y + self.height)
        p3 = QtCore.QPointF(self.x + self.width, self.y + self.height)
        polygon = QtGui.QPolygonF()
        polygon << p1 << p2 << p3 << p4
        return polygon

    def boundingRect(self):
        return self.selection_polygon.boundingRect()

    def shape(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.selection_polygon)
        return path


class TextInfoItem(QtGui.QGraphicsItem):

    def __init__(self, text, x, y, width, height, color, node):
        super(TextInfoItem, self).__init__()
        color.setAlpha(DEFAULT_INFO_TEXT_OPACITY)

        self.color = color
        self.node = node
        self.parent_x = x
        self.parent_y = y
        self.x = x
        self.y = y
        self.text = text
        self.width = width
        self.height = height
        self.bounding_rect = self.create_bounding_rect()
        self.rect = self.create_rectangle()
        self.connecting_line = self.create_connecting_line()
        self.text_item = self.create_text()

    def set_color(self, color):
        self.connecting_line.set_color(color)
        color.setAlpha(DEFAULT_INFO_TEXT_OPACITY)
        self.color = color

    def paint(self, painter, item, widget):
        self.rect.setPen(QtGui.QPen(self.node.color, SELECTION_LINE_WIDTH * 1.5, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin))
        self.rect.setBrush(QtGui.QBrush(self.node.color))
        self.rect.paint(painter, item, widget)

    def create_bounding_rect(self):
        return QtCore.QRectF(self.x, self.y, self.width, self.height)

    def create_connecting_line(self):
        return ConnectingLine(QtCore.QLineF(self.parent_x, self.parent_y, self.x, self.y), self.rect, self.color)

    def create_rectangle(self):
        return QtGui.QGraphicsRectItem(self.bounding_rect, self)

    def create_text(self):
        text_item = QtGui.QGraphicsTextItem()
        text_item.setPos(self.x, self.y)
        text_item.setPlainText(self.text)
        text_item.setParentItem(self.rect)
        r, g, b = self.color.red(), self.color.green(), self.color.blue()
        if r*0.299 + g*0.587 + b * 0.114 < 186:
            text_item.setDefaultTextColor(QtGui.QColor(255, 255, 255))
        return text_item

    def boundingRect(self):
        return self.bounding_rect

    def setPos(self, x, y):
        self.text_item.setPos(x, y)

    def set_parent_point(self, x, y):
        self.parent_x, self.parent_y = x, y

    def itemChange(self, change, value):
        if change == QtGui.QGraphicsItem.ItemPositionHasChanged:
            p1 = QtCore.QPoint(self.parent_x, self.parent_y)
            p2 = p1 - value.toPointF()
            self.connecting_line.setLine(QtCore.QLineF(p1, p2))
        return super(TextInfoItem, self).itemChange(change, value)


class ConnectingLine(QtGui.QGraphicsLineItem):
    
    def __init__(self, line, parent_obj, color):
        super(ConnectingLine, self).__init__(line, parent_obj)
        self.color = color
        
    def paint(self, painter, item, widget=None):
        pen = QtGui.QPen(self.color, SELECTION_LINE_WIDTH, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(self.line())

    def set_color(self, color):
        self.color = color


def get_longest_string_rows(string):
    st = ""
    longest = ""
    rows = 1
    for i in range(len(string)):
        st += string[i]
        if string[i] == "\n":
            rows += 1
            if len(st) > len(longest):
                longest = st
            st = ""
    else:
        st += "\n"
        if len(st) > len(longest):
            longest = st
    return longest, rows


