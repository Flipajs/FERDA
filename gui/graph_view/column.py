__author__ = 'Simon Mandlik'

from PyQt4 import QtCore, QtGui
from core.region.region import Region
from gui.graph_view.node import Node
from gui.graph_view.edge import Edge
from gui.img_controls.utils import cvimg2qtpixmap
import numpy as np
import time


class Column():

    def __init__(self, frame, scene, im_manager, empty=False):

        self.scene = scene
        self.im_manager = im_manager
        self.empty = empty
        self.x = 0
        self.frame = frame
        self.frame_sign = None
        self.compress_marker = None
        self.objects = []
        self.edges = {}
        self.nodes = {}

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

    def add_crop_to_col(self):
        from graph_visualizer import STEP
        for object in self.objects:
            if object not in self.nodes.keys():
                if isinstance(object, Region):
                    # img = self.im_manager.get_crop(self.frame, [object], width=STEP, height=STEP)
                    img = np.zeros((STEP, STEP, 3), dtype=np.uint8)
                    img[:,:,0] = 255
                    pixmap = cvimg2qtpixmap(img)
                    node = Node(self.scene.addPixmap(pixmap), self.scene, object, self.im_manager, STEP)
                    self.nodes[object] = node

                elif isinstance(object, tuple):
                    if object[0].frame_ == self.frame:
                        region = object[0]
                        if region in self.nodes.keys():
                            return
                        # img = self.im_manager.get_crop(self.frame, [region], width=STEP, height=STEP)
                        img = np.zeros((STEP, STEP, 3), dtype=np.uint8)
                        img[:,:,0] = 255
                        pixmap = cvimg2qtpixmap(img)
                        node = Node(self.scene.addPixmap(pixmap), self.scene, region, self. im_manager, STEP)
                        self.nodes[region] = node

    def set_x(self, x):
        self.x = x

    def draw(self, compress_axis, vertically):
        if self.empty:
            self.show_compress_marker(compress_axis, vertically)
        else:
            self.show_frame_number(vertically)
            for object in self.objects:
                if isinstance(object, Region):
                    self.show_node(object, vertically)

    def show_edge(self, edge, frame_columns, vertically, dir=None, node=None):
        from graph_visualizer import STEP, FROM_TOP, SPACE_BETWEEN_VER, SPACE_BETWEEN_HOR, GAP
        from_x = self.x
        if node is None:
            node = edge[1]
        position = self.get_position_object(node)
        from_y = GAP + FROM_TOP + position * STEP + STEP / 2 + SPACE_BETWEEN_VER * position

        if not (edge[2] is "partial"):
            column_left = frame_columns[edge[0].frame_]
            position = column_left.get_position_object(edge[0])
            to_x = column_left.x + STEP
            to_y = GAP + FROM_TOP + position * STEP + STEP/2 + SPACE_BETWEEN_VER * position
            column_left.show_node(edge[0], vertically)
        elif edge[2] == "partial" :
            to_y = from_y
            to_x = self.x - SPACE_BETWEEN_HOR / 2
            if not dir == "left":
                from_x += STEP
                to_x += STEP + SPACE_BETWEEN_HOR

        self.show_node(node, vertically)

        if vertically:
            from_x, from_y, to_x, to_y = from_y, from_x, to_y, to_x

        if edge in self.edges.keys():
            self.scene.removeItem(self.edges[edge].graphical_object)

        edge_obj = Edge(from_x, from_y, to_x, to_y, edge)
        self.edges[edge] = edge_obj
        self.scene.addItem(edge_obj.graphical_object)

    def show_node(self, region, vertically):
        from graph_visualizer import STEP, FROM_TOP, SPACE_BETWEEN_VER, GAP
        position = self.get_position_object(region)

        x = self.x
        y = GAP + FROM_TOP + position * STEP + SPACE_BETWEEN_VER * position

        if vertically:
            x, y = y, x

        if not (region in self.nodes.keys()):
            # img = self.im_manager.get_crop(self.frame, region, width=STEP, height=STEP)
            img = np.zeros((STEP, STEP, 3), dtype=np.uint8)
            img[:, :, 0] = 255
            pixmap = cvimg2qtpixmap(img)
            node = Node(self.scene.addPixmap(pixmap), self.scene, region, self.im_manager, STEP)
            node.hide()
            self.nodes[region] = node

        self.nodes[region].setPos(x, y)
        self.nodes[region].show()

    def show_compress_marker(self, compress_axis, vertically):
        from graph_visualizer import STEP, FROM_TOP
        if isinstance(self.frame, tuple):
            if self.compress_marker is None:
                self.compress_marker = QtGui.QGraphicsTextItem()
                self.objects.append(0)
                self.compress_marker.setDefaultTextColor(QtGui.QColor(0, 0, 0, 120))
                self.scene.addItem(self.compress_marker)
            x = self.x + STEP / 4 - 12.5
            y = FROM_TOP
            if vertically:
                x, y = y, x - 17.5
                string_len = len(str(self.frame[0] if isinstance(self.frame, tuple) else self.frame)) / 2
                self.compress_marker.setPlainText((" " * string_len + ".\n") * 3)
            else:
                self.compress_marker.setPlainText(". . .")
            self.compress_marker.setPos(x, y)
            self.compress_marker.show()
            if not compress_axis:
                self.compress_marker.hide()
        else:
            self.show_frame_number(vertically, compress_axis, True)

    def show_frame_number(self, vertically, compress_axis=True, empty = False):
        from graph_visualizer import STEP, FROM_TOP
        text = str(self.frame)
        text_obj = QtGui.QGraphicsTextItem(text) if self.frame_sign is None else self.frame_sign
        y = FROM_TOP
        if empty:
            text_obj.setDefaultTextColor(QtGui.QColor(0, 0, 0, 120))
        x = self.x + STEP / (4 if empty else 2)

        if vertically:
            x, y = y, x - 10
        else:
            x -= (len(text)) / 2.0 * 10

        if self.frame_sign is None:
            text_obj.setPos(x, y)
            self.frame_sign = text_obj
            self.scene.addItem(text_obj)
        else:
            self.frame_sign.setPos(x, y)
            self.frame_sign.show()
        if not compress_axis:
            self.frame_sign.hide()




