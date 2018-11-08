from __future__ import division
from __future__ import unicode_literals
from past.utils import old_div
from PyQt4 import QtCore, QtGui

import utils
from gui.settings import Settings as S_


class BaseMarker(QtGui.QGraphicsEllipseItem, object):
    """An ancestor to all ant markers. Note the changeHandler attribute. The changeHandler's marker_changed method
    is used when mouseReleaseEvent	happens.
    """

    def __init__(self, x, y, size, color, antId, changeHandler=None):
        super(BaseMarker, self).__init__(x, y, size, size)
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)
        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(color)
        self.setBrush(brush)
        self.setFlag(self.ItemSendsGeometryChanges, True)

        # antId is deprecated
        self.antId = antId
        self.id = antId
        self.changeHandler = changeHandler
        self.recently_changed = False

        darkness = 1-old_div((0.299*color.red() + 0.587*color.green() + 0.114*color.blue()),255)
        if darkness > 0.5:
            pen = QtGui.QPen(QtGui.QColor(0xFF, 0xFF, 0xFF, 0xFF))
        else:
            pen = QtGui.QPen(QtGui.QColor(0x00, 0x00, 0x00, 0xFF))

        pen.setWidth(1)
        self.setPen(pen)

        dotsize = float(2)
        self.dot = QtGui.QGraphicsEllipseItem(self.rect().center().x() - old_div(dotsize,2), self.rect().center().y() - old_div(dotsize,2), dotsize, dotsize, self)
        self.dot.setBrush(brush)
        # self.dot.setPen(pen)
        self.dot.setFlag(QtGui.QGraphicsItem.ItemIgnoresParentOpacity, True)
        self.dot.setOpacity(0)

        self.setOpacity(S_.visualization.basic_marker_opacity)

    def mousePressEvent(self, event):
        super(BaseMarker, self).mousePressEvent(event)
        self.setOpacity(.1)
        self.dot.setOpacity(S_.visualization.basic_marker_opacity)

    def mouseReleaseEvent(self, event):
        super(BaseMarker, self).mouseReleaseEvent(event)
        self.setOpacity(S_.visualization.basic_marker_opacity)
        self.dot.setOpacity(0)
        if self.changeHandler is not None:
            self.changeHandler(self.antId)

    def itemChange(self, change, value):
        if change == QtGui.QGraphicsItem.ItemPositionHasChanged:
            self.recently_changed = True
        return super(BaseMarker, self).itemChange(change, value)

    def centerPos(self):
        return QtCore.QPointF(self.pos().x() + old_div(self.rect().width(),2), self.pos().y() + old_div(self.rect().height(),2))

    def setCenterPos(self, x, y):
        self.setPos(x - old_div(self.rect().width(),2), y - old_div(self.rect().height(),2))


class CenterMarker(BaseMarker):
    """Marker that indicates center of the ant. When it is clicked and ctrl is pressed, it moves head and tail marker of
    the same ant."""

    def __init__(self, x, y, size, color, id=-1, changeHandler = None):
        super(CenterMarker, self).__init__(x, y, size, color, id, changeHandler)

        self.head_marker = None
        self.tail_marker = None

    def add_head_marker(self, head_marker):
        """Saves head_marker of the same ant"""
        self.head_marker = head_marker

    def add_tail_marker(self, tail_marker):
        """Saves tail_marker of the same ant"""
        self.tail_marker = tail_marker

    def mouseMoveEvent(self, event):
        super(CenterMarker, self).mouseMoveEvent(event)
        if (self.head_marker is not None) and (self.tail_marker is not None):
            if event.modifiers() == QtCore.Qt.ControlModifier:
                dx = event.pos().x() - event.lastPos().x()
                dy = event.pos().y() - event.lastPos().y()
                self.head_marker.setCenterPos(self.head_marker.centerPos().x() + dx, self.head_marker.centerPos().y() + dy)
                self.tail_marker.setCenterPos(self.tail_marker.centerPos().x() + dx, self.tail_marker.centerPos().y() + dy)


class TailHeadMarker(BaseMarker, object):
    """A marker that indifferently indicates tail or head of an ant. When it is clicked and ctrl is pressed, it rotates
    it's counterpart around the center marker.
    """

    def __init__(self, x, y, size, color, antId, changeHandler = None):
        super(TailHeadMarker, self).__init__(x, y, size, color, antId, changeHandler)

        self.center_marker = None
        self.other_marker = None

    def add_center_marker(self, center_marker):
        """Saves center_marker of the same ant"""
        self.center_marker = center_marker

    def add_other_marker(self, other_marker):
        """Saves the counterpart marker of the same ant"""
        self.other_marker = other_marker

    def mouseMoveEvent(self, event):
        super(TailHeadMarker, self).mouseMoveEvent(event)
        if (self.center_marker is not None) and (self.other_marker is not None):
            if event.modifiers() == QtCore.Qt.ControlModifier:
                x_dist = self.center_marker.centerPos().x() - self.centerPos().x()
                y_dist = self.center_marker.centerPos().y() - self.centerPos().y()
                self.other_marker.setCenterPos(self.center_marker.centerPos().x() + x_dist, self.center_marker.centerPos().y() + y_dist)


class HeadMarker(TailHeadMarker):

    """A marker that indicates head of the ant. It is different from the TailHeadMarker in that it has a dot in contrast
    color in the middle of it."""

    def __init__(self, x, y, size, color, antId, changeHandler = None):
        super(HeadMarker, self).__init__(x, y, size, color, antId, changeHandler)

        r, g, b = utils.visualization_utils.get_contrast_color(color.red(), color.green(), color.blue())

        dotsize = old_div(float(size),2)
        self.head_circle = QtGui.QGraphicsEllipseItem(self.rect().center().x() - old_div(dotsize,2), self.rect().center().y() - old_div(dotsize,2), dotsize, dotsize, self)
        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtGui.QColor(r, g, b))
        self.head_circle.setBrush(brush)
        self.head_circle.setPen(QtGui.QPen(QtCore.Qt.NoPen))


class TailMarker(TailHeadMarker):

    """A marker that indicates tail of the ant. It is currently no different from tailHeadMarker"""

    def __init__(self, x, y, size, color, antId, changeHandler = None):
        super(TailHeadMarker, self).__init__(x, y, size, color, antId, changeHandler)