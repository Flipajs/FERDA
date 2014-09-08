__author__ = 'flipajs'

from gui.img_controls import img_controls_qt, utils
from random import randint, shuffle
import os, time, math
from PyQt4 import QtCore, QtGui, QtOpenGL
import ImageQt
import cv2
from my_view import *


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s


class ImgControls(QtGui.QMainWindow, img_controls_qt.Ui_MainWindow):
    def __init__(self):
        super(ImgControls, self).__init__()
        self.setupUi(self)
        self.scene = QtGui.QGraphicsScene()
        self.graphics_view = MyView(self.centralwidget)
        self.init_graphic_view()
        self.load_image()

        self.items = []

        self.viewPositions.clicked.connect(self.view_positions)

        item = self.scene.addPixmap(self.pixMap)
        self.items.append(item)
        item.setPos(0, 0)

        triangle = QtGui.QPolygonF()
        height = 40.
        triangle.append(QtCore.QPointF(height/2.,0))
        triangle.append(QtCore.QPointF(-height/2.,-height/2.))
        triangle.append(QtCore.QPointF(-height/2,height/2.))
        triangle.append(QtCore.QPointF(height/2,0))

        matrix = QtGui.QMatrix()
        matrix.rotate(90)
        polygon = matrix.map(triangle)

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtGui.QColor(0, 0, 0xFF, 0x50))
        item = self.scene.addPolygon(polygon)
        item.setPos(100, 100)

        item.setPos(QtGui.QCursor.pos().x(), QtGui.QCursor.pos().y())

        item.setBrush(brush)
        item.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        item.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)
        #item.setFlag(QtGui.QGraphicsItem.ItemIs)
        #item.mapFromScene()
        self.scene.addItem(item)
        self.items.append(item)

        img_q = ImageQt.QImage(100, 100, QImage.Format_ARGB32)
        img_q.fill(QtGui.QColor(0, 0, 0, 0).rgba())

        for i in range(50,70):
            for j in range(50, 70):
                img_q.setPixel(j, i, QtGui.QColor(0, 255, 0, 30).rgba())

        pix_map = QtGui.QPixmap.fromImage(img_q)
        item = self.scene.addPixmap(pix_map)
        item.setPos(100, 100)
        item.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        item.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)

        self.scene.update()
        self.show()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

        if QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_N:
            item, item2 = utils.add_circle()
            self.scene.addItem(item)
            #self.scene.addItem(item2)
            self.items.append(item)

    def resizeEvent(self, QEvent):
        self.graphics_view_full()


    def graphics_view_full(self):
        w = self.width()
        h = self.height()
        left_panel_width = 100
        self.graphics_view.setGeometry(left_panel_width, 0, w-left_panel_width, h)

    def init_graphic_view(self):
        self.graphics_view_full()
        self.graphics_view.setObjectName(_fromUtf8("graphics_view"))
        self.graphics_view.setScene(self.scene)
        #self.graphics_view.setViewport(QtOpenGL.QGLWidget())


    def load_image(self):
        img2 = cv2.imread(os.path.expanduser('~/~dump/eight/frames/0.png'))

        self.pixMap = utils.cvimg2qtpixmap(img2)
        utils.view_add_bg_image(self.graphics_view, self.pixMap)


    def view_positions(self):
        for i in range(len(self.items)):
            print self.items[i].pos().x(), self.items[i].pos().y()

        self.positionsLabel.setText('test')