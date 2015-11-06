from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
from gui.img_controls.utils import cvimg2qtpixmap

SELECTION_LINE_WIDTH = 2

__author__ = 'Simon Mandlik'


class Node(QtGui.QGraphicsPixmapItem):

    def __init__(self, parent_pixmap, scene, region, img_manager, size):
        super(Node, self).__init__(parent_pixmap)
        self.region = region
        self.img_manager = img_manager
        self.scene = scene
        self.parent_pixmap = parent_pixmap
        self.pixmap_toggled = None
        self.size = size
        self.x = self.parent_pixmap.offset().x()
        self.y = self.parent_pixmap.offset().y()
        self.setFlags(QtGui.QGraphicsItem.ItemIsSelectable)
        self.selection_polygon = self.create_selection_polygon()
        self.toggled = False

    def toggle(self):
        if not self.toggled:
            if self.pixmap_toggled is None:
                self.create_pixmap()
            else:
                self.scene.addItem(self.pixmap_toggled)
        else:
            self.scene.removeItem(self.pixmap_toggled)
        self.toggled = False if self.toggled else True

    def create_pixmap(self):
        from graph_visualizer import STEP
        img_toggled = self.img_manager.get_crop(self.region.frame_, [self.region], width=STEP * 3, height=STEP * 3)
        pixmap = cvimg2qtpixmap(img_toggled)
        self.pixmap_toggled = self.scene.addPixmap(pixmap)
        width, height = self.scene.width(), self.scene.height()
        multiplier_x = 0 if self.parent_pixmap.pos().x() < width / 2 else -6
        multiplier_y = 0 if self.parent_pixmap.pos().y() < height / 2 else -6
        self.pixmap_toggled.setPos(self.parent_pixmap.pos().x() + (multiplier_x + 1) * STEP / 2,
                                   self.parent_pixmap.pos().y() + (multiplier_y + 1) * STEP / 2)
        self.pixmap_toggled.setFlags(QtGui.QGraphicsItem.ItemIsMovable)

    def setPos(self, x, y):
        from graph_visualizer import STEP
        if not(x == self.x and y == self.y):
            self.x, self.y = x, y
            self.parent_pixmap.setPos(x, y)
            if self.pixmap_toggled is not None:
                self.pixmap_toggled.setPos(x + STEP / 2, y + STEP / 2)

    def paint(self, painter, style_option_graphics_item, widget=None):
        self.parent_pixmap.paint(painter, style_option_graphics_item, None)
        if self.isSelected():
            pen = QtGui.QPen(Qt.black, SELECTION_LINE_WIDTH, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin)
        else:
            pen = QtGui.QPen(Qt.white, SELECTION_LINE_WIDTH, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)

        painter.setPen(pen)
        painter.drawPolygon(self.selection_polygon)

    def create_selection_polygon(self):
        p1 = QtCore.QPointF(self.x, self.y)
        p2 = QtCore.QPointF(self.x + self.size, self.y)
        p4 = QtCore.QPointF(self.x, self.y + self.size)
        p3 = QtCore.QPointF(self.x + self.size, self.y + self.size)
        polygon = QtGui.QPolygonF()
        polygon << p1 << p2 << p3 << p4
        return polygon

    def boundingRect(self):
        return self.selection_polygon.boundingRect()

    def shape(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.selection_polygon)
        return path
