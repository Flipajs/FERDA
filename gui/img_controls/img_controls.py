__author__ = 'flipajs'

from gui.img_controls import img_controls_qt
from random import randint, shuffle
import os, time, math
import ImageQt
from PyQt4 import QtCore, QtGui, QtOpenGL
import cv2
from my_view import *

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s


class QScene(QtGui.QGraphicsScene):
    def __init__(self, *args, **kwds):
        QtGui.QGraphicsScene.__init__(self, *args, **kwds)


class ImgControls(QtGui.QMainWindow, img_controls_qt.Ui_MainWindow):
    def __init__(self):
        super(ImgControls, self).__init__()
        self.setupUi(self)
        self.scene = QScene()
        self.graphics_view = MyView(self.centralwidget)
        self.init_graphic_view()
        self.load_image()

        item = self.scene.addPixmap(self.pixMap)
        item.setPos(0, 0)
        item = QtGui.QGraphicsRectItem(0, 0, 100, 200)
        item.setPos(100, 100)
        item.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        item.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)
        self.scene.addItem(item)

        #self.graphics_view.rotate(90)
        self.graphics_view.centerOn(0, 500)
        #self.graphics_view.transform()

        self.scene.update()
        self.show()

    def init_graphic_view(self):
        self.graphics_view.setGeometry(0, 0, 700, 500)
        self.graphics_view.setObjectName(_fromUtf8("graphics_view"))
        self.graphics_view.setScene(self.scene)
        #self.graphics_view.setViewport(QtOpenGL.QGLWidget())

    def load_image(self):
        img2 = cv2.imread(os.path.expanduser('~/Downloads/img.jpg'))

        start = time.time()
        self.imgQ2 = ImageQt.QImage(img2.data, img2.shape[1], img2.shape[0], img2.shape[1]*3, 13)
        self.pixMap = QtGui.QPixmap.fromImage(self.imgQ2.rgbSwapped())

        end = time.time()

        gv_w = self.graphics_view.geometry().width()
        gv_h = self.graphics_view.geometry().height()
        im_w = self.pixMap.width()
        im_h = self.pixMap.height()

        if gv_w / float(im_w) <= gv_h / float(im_h):
            val = math.floor((gv_w / float(im_w))*100) / 100
            self.graphics_view.scale(val, val)
        else:
            val = math.floor((gv_h / float(im_h))*100) / 100
            print "HEIGHT", im_h / float(gv_h)
            self.graphics_view.scale(val, val)

        print "image to qt time: ", end - start