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
        if position < len(self.objects):
            try:
                self.objects[position] = object
            except:
                pass
        else:
            while len(self.objects) < position:
                self.objects.append(None)
            else:
                self.objects.append(object)

    def is_free(self, position=0):
        if position < 0:
            return False
        elif position > len(self.objects) - 1:
            return True
        if isinstance(self.objects[position], (Region, Node)):
            return False
        elif isinstance(self.objects[position], tuple):
            if self.objects[position][2] == "chunk" or self.objects[position][2] == "line":
                return False
        return True

    def contains(self, item):
        return True if item in self.objects else False

    def get_position_object(self, object):
        return self.objects.index(object)

    def get_x(self):
        return self.x

    def add_crop_to_col(self, im_manager, size):
        for object in self.objects:
            if isinstance(object, Region):
                img = im_manager.get_crop(object.frame_, [object])
                pixmap = cvimg2qtpixmap(img)
                node = Node(object, pixmap, size)
                self.objects[self.objects.index(object)] = node

    def set_x(self, x):
        self.x = x

    def draw(self, vertically, scene, frame_columns):
        #vyresit vertically
        # self.get_position()

        # p1 = QtCore.QPointF(self.x, 0)
        # p2 = QtCore.QPointF(self.x + STEP, 0)
        # p4 = QtCore.QPointF(self.x, STEP*len(self.objects))
        # p3 = QtCore.QPointF(self.x + STEP, STEP*len(self.objects))
        # polygon = QtGui.QGraphicsRectItem(QtCore.QRectF(p1, p4))
        # scene.addItem(polygon)

        if self.empty:
            self.compress_marker.setPos(self.x + STEP/4, 0)
            scene.addItem(self.compress_marker)
        else:
            self.show_frame_number(vertically, scene)
            for object in self.objects:
                if isinstance(object, Node):
                    pass
                    # object.set_x(self.x)
                    # vertically
                    # nastavit pos
                    #pridat pixmapu
                    # object.draw()
                elif object is None or isinstance(object, Region):
                    pass

                elif isinstance(object, tuple) and object[1].frame_ == self.frame:
                    from_x = self.x
                    from_y = self.get_position_object(object[1]) * STEP + STEP/2
                    to_x = frame_columns[object[0].frame_].get_x() + STEP
                    try:
                        to_y = frame_columns[object[0].frame_].get_position_object(object) * STEP + STEP/2

                        if vertically:
                            from_x, from_y = from_y, from_x
                        edge = Edge(from_x, from_y, to_x, to_y, object)
                        scene.addItem(edge.graphical_object)

                    except:
                        pass
                    #vykreslit i nody se kteryma sousedi
                    #partial?
                else:
                    pass

    def show_frame_number(self, vertically, scene):
        text = Qt.QGraphicsTextItem(str(self.frame))
        x = self.x
        y = 0
        if vertically:
            x, y = y, x
        text.setPos(x, y)
        scene.addItem(text)




