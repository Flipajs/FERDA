__author__ = 'simon'

from PyQt4 import QtGui, QtCore

SCALE_FACTOR = 1.01

class MyViewZoomable(QtGui.QGraphicsView, object):
    
    def __init__(self, ngv):
        super(MyViewZoomable, self).__init__()
        self.drag = False
        self.ngv = ngv

    def mouseDoubleClickEvent(self, QMouseEvent):
        if self.drag:
            self.setDragMode(QtGui.QGraphicsView.NoDrag)
        else:
            self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
        self.drag = False if self.drag else True

    def keyPressEvent(self, event):
        key_i = QtCore.Qt.Key_I
        key_o = QtCore.Qt.Key_O
        key_b = QtCore.Qt.Key_B
        key_n = QtCore.Qt.Key_N
        event_key = event.key()

        if event_key == key_b or event_key == key_n:
            node = None
            if event_key == key_b:
                node = self.ngv.selected_edge[0]
            else:
                node = self.ngv.selected_edge[1]
            self.ngv.pixmaps[node].setSelected(True)
            self.centerOn(self.ngv.pixmaps[node].parent_pixmap.pos())

        if event_key == key_o:
            self.zoom(False)
        elif event_key == key_i:
            self.zoom(True)
        else:
            super(MyViewZoomable, self).keyPressEvent(event)

    def zoom (self, in_out):
        self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
        m11 = self.transform().m11()
        m22 = self.transform().m22()
        time_line = QtCore.QTimeLine(100, self)
        time_line.setUpdateInterval(5)

        if in_out and not (m11 > 2 or m22 > 2):
            time_line.valueChanged.connect(self.scale_in)
        elif m11 > 0.1 or m22 > 0.1:
            time_line.valueChanged.connect(self.scale_out)

        time_line.start()
        time_line = None

    def scale_out(self, x):
        self.scale(1 - (SCALE_FACTOR - 1), 1 - (SCALE_FACTOR - 1))

    def scale_in(self, x):
        self.scale(SCALE_FACTOR, SCALE_FACTOR)

