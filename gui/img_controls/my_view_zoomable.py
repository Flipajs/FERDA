__author__ = 'simon'

from PyQt4 import QtGui, QtCore

SCALE_FACTOR = 1.01


class MyViewZoomable(QtGui.QGraphicsView):
    
    def __init__(self, ngv):
        super(MyViewZoomable, self).__init__()
        self.drag = False
        self.ngv = ngv
        self.position_from = None
        self.node_position_to = None
        self.node_1 = None
        self.node_2 = None
        self.last_event_go_to = False

    def mouseDoubleClickEvent(self, QMouseEvent):
        if self.drag:
            self.setDragMode(QtGui.QGraphicsView.NoDrag)
        else:
            self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
        self.drag = False if self.drag else True
        self.last_event_go_to = False

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            gsEvent = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, event.pos(), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
            super(MyViewZoomable, self).mousePressEvent(gsEvent)
        else:
            super(MyViewZoomable, self).mousePressEvent(event)

    def keyPressEvent(self, event):
        key_i = QtCore.Qt.Key_I
        key_o = QtCore.Qt.Key_O
        key_b = QtCore.Qt.Key_B
        key_n = QtCore.Qt.Key_N
        # key_q = QtCore.Qt.Key_Q
        # key_w = QtCore.Qt.Key_W
        event_key = event.key()

        if event_key == key_b or event_key == key_n:
            self.go_to_next(event_key, event)
        self.last_event_go_to = False
        if event_key == key_o:
            self.zoom(False)
        elif event_key == key_i:
            self.zoom(True)
        # elif event_key == key_q or event_key == key_w:
        #     self.stretch(event_key == key_q)
        else:
            super(MyViewZoomable, self).keyPressEvent(event)

    def zoom (self, in_out):
        self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
        m11 = self.transform().m11()
        m22 = self.transform().m22()
        time_line = QtCore.QTimeLine(100, self)
        time_line.setUpdateInterval(1)

        if in_out and not (m11 > 3 or m22 > 3):
            time_line.valueChanged.connect(self.scale_in)
        elif m11 > 0.1 or m22 > 0.1:
            time_line.valueChanged.connect(self.scale_out)

        time_line.start()
        time_line = None

    def scale_out(self, x):
        self.scale(1 - (SCALE_FACTOR - 1), 1 - (SCALE_FACTOR - 1))

    def scale_in(self, x):
        self.scale(SCALE_FACTOR, SCALE_FACTOR)

    def go_to_next(self, event_key, event):
        key_b = QtCore.Qt.Key_B
        self.node_1 = self.ngv.selected_edge.core_obj[0]
        self.node_2 = self.ngv.selected_edge.core_obj[0]
        if event_key == key_b:
            self.ngv.pixmaps[self.node_1].setSelected(True)
            self.node_position_to = self.ngv.pixmaps[self.node_1].parent_pixmap.pos()
        else:
            self.ngv.pixmaps[self.node_2].setSelected(True)
            self.node_position_to = self.ngv.pixmaps[self.node_2].parent_pixmap.pos()
        if not self.last_event_go_to:
            self.position_from = self.ngv.selected_edge.core_obj[1]
        elif event_key == key_b:
            self.position_from = self.ngv.pixmaps[self.node_2].parent_pixmap.pos()
        else:
            self.position_from = self.ngv.pixmaps[self.node_1].parent_pixmap.pos()
        self.last_event_go_to = True
        time_line = QtCore.QTimeLine(500, self)
        time_line.valueChanged.connect(self.center)
        time_line.start()

    def center(self, z):
        x1 = self.position_from.x()
        y1 = self.position_from.y()
        x2 = self.node_position_to.x()
        y2 = self.node_position_to.y()
        print self.position_from
        print self.node_position_to
        print str(x1 + (z * (x2 - x1))), str(y1 + (z * (y2 - y1)))
        point = QtCore.QPointF(x1 + (z * (x2 - x1)), y1 + (z * (y2 - y1)))
        self.centerOn(point)

    # def stretch(self, shrink):
    #     self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
    #     m11 = self.transform().m11()
    #     m22 = self.transform().m22()
    #     time_line = QtCore.QTimeLine(100, self)
    #     time_line.setUpdateInterval(1)
    #
    #     if shrink and not (m11 > 3 or m22 > 3):
    #         time_line.valueChanged.connect(self.scale_shrink)
    #     elif m11 > 0.1 or m22 > 0.1:
    #         time_line.valueChanged.connect(self.scale_stretch)
    #
    #     time_line.start()
    #
    # def scale_shrink(self, x):
    #     self.scale(1 - (SCALE_FACTOR - 1), 1)
    #
    # def scale_stretch(self, x):
    #     self.scale(SCALE_FACTOR, 1)


