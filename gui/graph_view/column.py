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
            self.compress_marker = QtGui.QGraphicsTextItem('. . .')
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
                if isinstance(object, Node):
                    if item is Node.region:
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
                elif isinstance(item, Node):
                    if object == item.region:
                        return self.objects.index(item)

    def get_x(self):
        return self.x

    def add_crop_to_col(self, im_manager, size):
        for object in self.objects:
            if isinstance(object, Region):

                import time
                # start = time.time()
                # img = im_manager.get_crop(object.frame_, [object])
                # end = time.time()
                # print ("img = im_manager.get_crop(object.frame_, [object])  Exectime: " + str(end - start))
                # start = time.time()
                # pixmap = cvimg2qtpixmap(img)
                # end = time.time()
                # print ("pixmap = cvimg2qtpixmap(img)  Exectime: " + str(end - start))
                # start = time.time()
                # node = Node(object, pixmap, size, img)
                # end = time.time()
                # print ("node = Node(object, pixmap, size, img)  Exectime: " + str(end - start))
                # start = time.time()
                # self.objects[self.objects.index(object)] = node
                # end = time.time()
                # print ("self.objects[self.objects.index(object)] = node  Exectime: " + str(end - start))

                img = im_manager.get_crop(object.frame_, [object])
                pixmap = cvimg2qtpixmap(img)
                node = Node(object, pixmap, size, img)
                self.objects[self.objects.index(object)] = node

    def set_x(self, x):
        self.x = x

    def draw(self, vertically, scene, frame_columns):
        # #TODO pak smazat, vyvojove ucely
        # from graph_visualizer import STEP, FROM_TOP
        # if not self.empty:
        #     if vertically:
        #         p1 = QtCore.QPointF(0, self.x)
        #         p4 = QtCore.QPointF(STEP*len(self.objects), self.x + STEP)
        #     else:
        #         p1 = QtCore.QPointF(self.x, FROM_TOP)
        #         p4 = QtCore.QPointF(self.x + STEP, STEP*len(self.objects) + FROM_TOP)
        #     polygon = QtGui.QGraphicsRectItem(QtCore.QRectF(p1, p4))
        #     scene.addItem(polygon)

        if self.empty:
            self.show_compress_marker(vertically, scene)
        else:
            self.show_frame_number(vertically, scene)

            for object in self.objects:
                if isinstance(object, Node):
                    self.show_node(self, object, vertically, scene)

                elif object is None or isinstance(object, Region):
                    #TODO MUZE SE TO STAT? CO PAK?
                    self.show_node(self, object, vertically, scene)
                #
                # elif isinstance(object, tuple) and object[1].frame_ == self.frame:
                #     self.show_edge(object, frame_columns, vertically, scene)

    def show_edge(self, edge, frame_columns, vertically, scene, dir=None, node=None):
        from graph_visualizer import STEP, FROM_TOP, SPACE_BETWEEN_VER, SPACE_BETWEEN_HOR, GAP

        from_x = self.x
        if node is None:
            node = edge[1]
        position = self.get_position_object(node)
        from_y = GAP + FROM_TOP + position * STEP + STEP / 2 + SPACE_BETWEEN_VER * position

        if not edge[2] is "partial":
            column_left = frame_columns[edge[0].frame_]
            position = column_left.get_position_object(edge[0])
            to_x = column_left.x + STEP
            to_y = GAP + FROM_TOP + position * STEP + STEP/2 + SPACE_BETWEEN_VER * position
            self.show_node(column_left, edge[0], vertically, scene)
            #TODO co kdyz sem prijde volani od line a PARTIAL a node je uz vykreslena, co kdyz je uz vykreslena

        elif edge[2] == "partial" :
            to_y = from_y
            to_x = self.x - SPACE_BETWEEN_HOR / 2
            if not dir == "left":
                from_x += STEP
                to_x += STEP + SPACE_BETWEEN_HOR
            #TODO co kdyz sem prijde volani od line a PARTIAL a node je uz vykreslena, co kdyz je uz vykreslena

        self.show_node(self, node, vertically, scene)

        if vertically:
            from_x, from_y, to_x, to_y = from_y, from_x, to_y, to_x
        edge = Edge(from_x, from_y, to_x, to_y, edge)
        scene.addItem(edge.graphical_object)

    def show_node(self, col, object, vertically, scene):
        from graph_visualizer import STEP, FROM_TOP, SPACE_BETWEEN_VER, GAP
        position = col.get_position_object(object)

        x = col.x
        y = GAP + FROM_TOP + position * STEP + SPACE_BETWEEN_VER * position

        if vertically:
            x, y = y, x

        if isinstance(object, Node):
            object.set_pos(x, y)
            scene.addItem(object)
            # pridat pixmapu, zjisti, jestli uz je atd..

            if not (object.x == x and object.y == y):
                object.set_pos(x, y)
            #zobrazit

        #TODO pak smazat vyvojove ucely
        p1 = QtCore.QPointF(x, y)
        p4 = QtCore.QPointF(x + STEP, y + STEP)
        polygon = QtGui.QGraphicsRectItem(QtCore.QRectF(p1, p4))
        scene.addItem(polygon)

    def show_compress_marker(self, vertically, scene):
        from graph_visualizer import STEP, FROM_TOP
        if isinstance(self.frame, tuple):
            x = self.x + STEP/4 - 12.5
            y = FROM_TOP
            if vertically:
                x, y = y, x - 17.5
                string_len = len(str(self.frame[0] if isinstance(self.frame, tuple) else self.frame)) / 2
                self.compress_marker.setPlainText((" " * string_len + ".\n") * 3)
            self.compress_marker.setPos(x, y)
            self.compress_marker.setDefaultTextColor(QtGui.QColor(0, 0, 0, 120))
            scene.addItem(self.compress_marker)
        else:
            self.show_frame_number(vertically, scene, True)

    def show_compress_marker(self, vertically, scene):
        from graph_visualizer import STEP, FROM_TOP
        if isinstance(self.frame, tuple):
            x = self.x + STEP/4 - 12.5
            y = FROM_TOP
            if vertically:
                x, y = y, x - 17.5
                string_len = len(str(self.frame[0] if isinstance(self.frame, tuple) else self.frame)) / 2
                self.compress_marker.setPlainText((" " * string_len + ".\n") * 3)
            self.compress_marker.setPos(x, y)
            self.compress_marker.setDefaultTextColor(QtGui.QColor(0, 0, 0, 120))
            scene.addItem(self.compress_marker)
        else:
            self.show_frame_number(vertically, scene, True)

    def show_frame_number(self, vertically, scene, empty = False):
        from graph_visualizer import STEP, FROM_TOP
        text = str(self.frame)
        text_obj = QtGui.QGraphicsTextItem(text)
        y = FROM_TOP
        if empty:
            text_obj.setDefaultTextColor(QtGui.QColor(0, 0, 0, 120))
        x = self.x + STEP/(4 if empty else 2)

        if vertically:
            x, y = y, x - 10
        else:
            x -= (len(text)) / 2.0 * 10
        text_obj.setPos(x, y)
        scene.addItem(text_obj)




