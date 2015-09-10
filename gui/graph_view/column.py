__author__ = 'Simon Mandlik'

from PyQt4 import Qt, QtCore, QtGui
from core.region.region import Region
from gui.graph_view.node import Node
from gui.graph_view.edge import Edge
from gui.img_controls.utils import cvimg2qtpixmap


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
        if position < len(self.objects):
            if self.objects[position] == object or isinstance(self.objects[position], tuple):
                pass
            else:
                self.objects[position] = object
        else:
            while len(self.objects) < position:
                self.objects.append(None)
            else:
                self.objects.append(object)

    def is_free(self, position=0, object=None):
        if position < 0:
            return False
        elif position > len(self.objects) - 1:
            return True
        elif object == self.objects[position]:
            return True
        elif isinstance(self.objects[position], (Region, Node)):
            return False
        elif isinstance(self.objects[position], tuple):
            if self.objects[position][2] == "chunk":
                return False
        else:
            return True

    def contains(self, item):
        if item in self.objects:
            return True
        else:
            for object in self.objects:
                if isinstance(object, tuple):
                    if object[0] is item or object[1] is item:
                        return True
        return False

    def get_position_object(self, object):
        try:
            return self.objects.index(object)
        except:
            for item in self.objects:
                if isinstance(item, tuple):
                    if item[0] == object or item[1] == object:
                            return self.objects.index(item)

    def get_x(self):
        return self.x

    def add_crop_to_col(self, im_manager, size):
        for object in self.objects:
            if isinstance(object, Region):
                img = im_manager.get_crop(object.frame_, [object])
                pixmap = cvimg2qtpixmap(img)
                node = Node(object, pixmap, size, img)
                self.objects[self.objects.index(object)] = node

    def set_x(self, x):
        self.x = x

    def draw(self, vertically, scene, frame_columns):


        from graph_visualizer import STEP, FROM_TOP

        p1 = QtCore.QPointF(self.x, 0)
        p2 = QtCore.QPointF(self.x + STEP, 0)
        p4 = QtCore.QPointF(self.x, STEP*len(self.objects))
        p3 = QtCore.QPointF(self.x + STEP, STEP*len(self.objects))
        polygon = QtGui.QGraphicsRectItem(QtCore.QRectF(p1, p4))
        scene.addItem(polygon)

        if self.empty:
            if isinstance(self.frame, tuple):
                self.compress_marker.setPos(self.x + STEP/4 - 5, 0)
                scene.addItem(self.compress_marker)
            else:
                self.show_frame_number(vertically, scene, True)
        else:
            self.show_frame_number(vertically, scene)
            for object in self.objects:
                if isinstance(object, Node):
                    x = self.x
                    y = self.objects.index(object) * STEP
                    if vertically:
                        x, y = y, x
                    object.set_pos(x, y)
                    scene.addItem(object)
                    #pridat pixmapu

                elif object is None or isinstance(object, Region):
                    #TODO
                    pass

                elif isinstance(object, tuple) and object[1].frame_ == self.frame:
                    from_x = self.x
                    from_y = FROM_TOP + self.objects.index(object) * STEP + STEP/2
                    to_x = frame_columns[object[0].frame_].x + STEP
                    to_y = FROM_TOP + frame_columns[object[0].frame_].get_position_object(object) * STEP + STEP/2

                    if vertically:
                        from_x, from_y = from_y, from_x
                    edge = Edge(from_x, from_y, to_x, to_y, object)
                    scene.addItem(edge.graphical_object)

                    #vykreslit i nody se kteryma sousedi
                    #partial?
                else:
                    pass

    def show_frame_number(self, vertically, scene, empty = False):
        from graph_visualizer import STEP

        if empty:
            text = "_" + str(self.frame) + "_"
        else:
            text = str(self.frame)
        text_obj = QtGui.QGraphicsTextItem(text)
        y = 0
        x = self.x + STEP/2 - ((len(text) - 1) /2) * 20
        print(self.x)
        print(x)
        if vertically:
            x, y = y, x
        text_obj.setPos(x, y)
        scene.addItem(text_obj)




