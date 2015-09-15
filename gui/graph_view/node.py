__author__ = 'Simon Mandlik'

SELECTION_LINE_WIDTH = 2

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
from gui.img_controls.utils import cvimg2qtpixmap
import numpy as np

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
        from graph_visualizer import STEP
        if not self.toggled:
            if self.pixmap_toggled is None:
                # img_toggled = self.im_manager.get_crop(self.region.frame_, [self.region], width=STEP * 3, height=STEP * 3)
                img_toggled = np.zeros((STEP * 3, STEP * 3, 3), dtype=np.uint8)
                img_toggled[:, 0, :] = 255
                pixmap = cvimg2qtpixmap(img_toggled)
                self.pixmap_toggled = self.scene.addPixmap(pixmap)
                self.pixmap_toggled.setPos(self.parent_pixmap.pos().x() + STEP / 2, self.parent_pixmap.pos().y() + STEP / 2)
                self.pixmap_toggled.setFlags(QtGui.QGraphicsItem.ItemIsMovable)
            else:
                self.scene.addItem(self.pixmap_toggled)
        else:
            self.scene.removeItem(self.pixmap_toggled)
        self.toggled = False if self.toggled else True

    def setPos(self, x, y):
        from graph_visualizer import STEP
        if not(x == self.x and y == self.y):
            self.x = x
            self.y = y
            self.parent_pixmap.setPos(x, y)
            if self.pixmap_toggled is not None:
                self.pixmap_toggled.setPos(x + STEP / 2, y + STEP / 2)

    def paint(self, QPainter, QStyleOptionGraphicsItem, QWidget_widget = None):
        self.parent_pixmap.paint(QPainter, QStyleOptionGraphicsItem, None)
        if self.isSelected():
            pen = QtGui.QPen(Qt.black, SELECTION_LINE_WIDTH, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin)
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
