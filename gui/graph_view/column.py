__author__ = 'Simon Mandlik'

from PyQt4 import Qt, QtCore, QtGui
from core.region.region import Region
from gui.graph_view.node import Node
from gui.graph_view.edge import Edge
from gui.img_controls.utils import cvimg2qtpixmap

STEP = 20
FROM_TOP = STEP

class Column():

    def __init__(self, frame, empty=False):

        self.empty = empty
        self.x = 0
        self.frame = frame
        self.objects = []

        if self.empty:
            self.compress_marker = QtGui.QGraphicsTextItem('...')
            self.objects.append(0)

    def add_object(self, object, position):
        self.objects[position] = object

    def is_free(self, position=0):
        if position < 0:
            return False
        if isinstance(self.objects[position], (Region, Node)):
            return False
        elif isinstance(self.objects[position], tuple):
            if self.objects[position][2] == "chunk":
                return False
        return True

    def contains(self, item):
        return True if item in self.objects else False

    def get_position_object(self, object):
        return self.objects.index(object)

    def get_x(self):
        return self.x

    def add_crop_to_col(self, im_manager):
        for object in self.objects:
            if isinstance(object, Region):
                img = im_manager.get_crop(object._frame, object)
                pixmap = cvimg2qtpixmap(img)
                node = Node(pixmap)
                self.objects[self.objects.index(object)] = node

    def set_x(self, x):
        self.x = x

    def draw(self, vertically, scene, frame_columns):
        #vyresit vertically
        self.get_position()
        if self.empty:
            self.compress_marker.setPos(self.x + STEP/4, 0)
            self.compress_marker.show()
        else:
            self.show_frame_number(vertically, scene)
            for object in self.objects:
                if isinstance(object, Region):

                    # object.set_x(self.x)
                    # nastavit pos
                    object.draw()
                elif isinstance(object, tuple) and object[1]._frame == self.frame:
                    to_x = self.x
                    to_y = self.get_position_object(object[1]) * STEP + STEP/2
                    from_x = frame_columns[object[0]._frame].get_x() + STEP
                    from_y = frame_columns[object[0]._frame].get_position_object(object) * STEP + STEP/2
                    if vertically:
                        from_x, from_y = from_y, from_x
                    edge = Edge(from_x, from_y, to_x, to_y, object)
                    scene.addItem(edge)

    def show_frame_number(self, vertically, scene):
        text = Qt.QGraphicsTextItem(str(self.frame))
        x = self.x
        y = 0
        if vertically:
            x, y = y, x
        text.setPos(x, y)
        scene.addItem(text)




