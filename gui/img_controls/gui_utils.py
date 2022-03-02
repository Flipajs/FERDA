__author__ = 'filip@naiser.cz'
from PyQt6 import QtCore, QtGui, QtWidgets
import math
import qimage2ndarray


def cvimg2qimage(img):
    return qimage2ndarray.array2qimage(img)


def cvimg2qtpixmap(img):
    return QtGui.QPixmap.fromImage(cvimg2qimage(img))


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
    brush = QtGui.QBrush(QtCore.Qt.BrushStyle.SolidPattern)
    brush.setColor(color)
    item = QtWidgets.QGraphicsEllipseItem(0, 0, size, size)

    #item.setPos(QtGui.QCursor.pos().x(), QtGui.QCursor.pos().y())

    item.setBrush(brush)
    item.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
    item.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

    brush = QtGui.QBrush(QtCore.Qt.BrushStyle.SolidPattern)
    brush.setColor(QtGui.QColor(0x22, 0xFF, 0x00, 0x30))
    item2 = QtWidgets.QGraphicsEllipseItem(size/2 - 0.5, size/2 - 0.5, 1, 1)
    item2.setPos(0, 0)
    item2.setBrush(brush)
    item2.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
    item2.setParentItem(item)

    return item, item2
