__author__ = 'Simon Mandlik'

SELECTION_LINE_WIDTH = 2

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
from gui.img_controls.utils import cvimg2qtpixmap

class Node(QtGui.QGraphicsPixmapItem):

    def __init__(self, parent_pixmap, scene, region, img, size):
        super(Node, self).__init__(parent_pixmap)
        self.region = region
        self.scene = scene
        self.img = img
        self.img_toggled = None

        self.size = size

        self.parent_pixmap = parent_pixmap
        self.x = 0
        self.y = 0
        self.setFlags(QtGui.QGraphicsItem.ItemIsSelectable)
        self.selection_polygon = self.create_selection_polygon()

        self.toggled = False

    def toggle(self, img_manager):
        from graph_visualizer import STEP
        #TODO
        if not self.toggled:
            if self.img_toggled is None:
                self.img_toggled = self.im_manager.get_crop(self.frame, [self.region], width=STEP, height=STEP)
                pixmap = cvimg2qtpixmap(self.img_toggled)
                pixmap = self.scene.addPixmap(pixmap)
                pixmap.setPos(self.pos() + 20)
        self.toggled = False if self.toggle() else True


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
