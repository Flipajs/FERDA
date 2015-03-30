__author__ = 'filip@naiser.cz'

from PyQt4 import QtGui, QtCore
from PIL import ImageQt

import math


def cvimg2qtpixmap(img):
    img_q = ImageQt.QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, 13)
    pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

    return pix_map


def view_add_bg_image(g_view, pix_map):
    gv_w = g_view.geometry().width()
    gv_h = g_view.geometry().height()
    im_w = pix_map.width()
    im_h = pix_map.height()

    m11 = g_view.transform().m11()
    m22 = g_view.transform().m22()

    if m11 and m22 == 1:
        if gv_w / float(im_w) <= gv_h / float(im_h):
            val = math.floor((gv_w / float(im_w))*100) / 100
            g_view.scale(val, val)
        else:
            val = math.floor((gv_h / float(im_h))*100) / 100
            g_view.scale(val, val)


def add_circle(size=6, color=QtGui.QColor(0x22, 0x22, 0xFF, 0x30)):
    brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
    brush.setColor(color)
    item = QtGui.QGraphicsEllipseItem(0, 0, size, size)

    #item.setPos(QtGui.QCursor.pos().x(), QtGui.QCursor.pos().y())

    item.setBrush(brush)
    item.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
    item.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)

    # brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
    # brush.setColor(QtGui.QColor(0x22, 0xFF, 0x00, 0x30))
    # item2 = QtGui.QGraphicsEllipseItem(size/2 - 0.5, size/2 - 0.5, 1, 1)
    # item2.setPos(0, 0)
    # item2.setBrush(brush)
    # item2.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
    # item2.setParentItem(item)

    return item#, item2