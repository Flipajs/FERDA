__author__ = 'filip@naiser.cz'

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QImage
# import ImageQt
import math
import numpy as np


def cvimg2qtpixmap(img, transparent=False):
    if transparent:
        img_q = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, 13)
        pix_map = QtGui.QPixmap.fromImage(img_q)
        # img_q =
        """
        bmp = QtGui.QBitmap()
        bmp.fromImage(img_q.createAlphaMask())
        pix_map.setMask(bmp)
        """
        """
        qcolor = QtGui.QColor()
        qcolor.setAlpha(200)
        pixmap = QtGui.QPixmap.fromImage(img_q.createAlphaMask())
        pixmap.fill(qcolor)
        """
        """
        for x in range(0,120):
            for y in range(0,120):
                c = pix_map.toImage().pixel(x,y)
                colors = QtGui.QColor(c).getRgbF()
                print "(%s,%s) = %s" % (x, y, colors)
        """

        return pix_map
    else:
        img_q = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, 13)
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
    item = QtWidgets.QGraphicsEllipseItem(0, 0, size, size)

    #item.setPos(QtGui.QCursor.pos().x(), QtGui.QCursor.pos().y())

    item.setBrush(brush)
    item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
    item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)

    brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
    brush.setColor(QtGui.QColor(0x22, 0xFF, 0x00, 0x30))
    item2 = QtWidgets.QGraphicsEllipseItem(size/2 - 0.5, size/2 - 0.5, 1, 1)
    item2.setPos(0, 0)
    item2.setBrush(brush)
    item2.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
    item2.setParentItem(item)

    return item, item2
