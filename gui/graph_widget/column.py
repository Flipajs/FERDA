import threading
from PyQt4 import QtGui
import numpy as np
from core.region.region import Region
from gui.graph_widget_loader import FROM_TOP, SPACE_BETWEEN_HOR, SPACE_BETWEEN_VER, GAP
from gui.graph_widget.edge import Edge
from gui.graph_widget.node import Node
from gui.img_controls.utils import cvimg2qtpixmap

__author__ = 'Simon Mandlik'


class Column:

    def __init__(self, frame, scene, im_manager, relative_margin, width, height, empty=False):
        self.scene = scene
        self.im_manager = im_manager
        self.empty = empty
        self.x = 0
        self.frame = frame
        self.relative_margin = relative_margin
        self.width = width
        self.height = height

        self.frame_sign = None

        self.objects = []
        self.edges = {}
        self.items_nodes = {}
        self.regions_images = {}

        self.compress_marker = QtGui.QGraphicsTextItem()
        self.objects.append(0)
        self.compress_marker.setDefaultTextColor(QtGui.QColor(0, 0, 0, 120))
        self.scene.addItem(self.compress_marker)

    def get_start_frame(self):
        if isinstance(self.frame, tuple):
            return  self.frame[0]
        else:
            return self.frame

    def get_end_frame(self):
        if isinstance(self.frame, tuple):
            return  self.frame[1]
        else:
            return self.frame

    def add_object(self, to_add, position):
        if position < len(self.objects):
            if not(self.objects[position] == to_add or isinstance(self.objects[position], tuple)):
                self.objects[position] = to_add
        else:
            while len(self.objects) < position:
                self.objects.append(None)
            else:
                self.objects.append(to_add)

    def is_free(self, position=0, item=None):
        if position < 0:
            return False
        elif position > len(self.objects) - 1:
            return True
        elif item == self.objects[position]:
            return True
        elif isinstance(self.objects[position], (Region, Node)):
            return False
        elif isinstance(self.objects[position], tuple):
            if self.objects[position][2] == "chunk":
                return False
        return True

    def contains(self, item):
        if item in self.objects:
            return True
        else:
            for obj in self.objects:
                if isinstance(obj, tuple):
                    if obj[0] is item or obj[1] is item:
                        return True
        return False

    def get_position_item(self, item_to_locate):
        if item_to_locate in self.objects:
            return self.objects.index(item_to_locate)
        else:
            for item in self.objects:
                if isinstance(item, tuple):
                    if item[0] == item_to_locate or item[1] == item_to_locate:
                            return self.objects.index(item)
                elif isinstance(item, Node):
                    if item_to_locate == item.region:
                        return self.objects.index(item)

    def prepare_images(self):
        for item in self.objects:
            if not (item in (self.items_nodes.keys() + self.regions_images.keys()) or item is None):
                if isinstance(item, tuple):
                    if item[0].frame_ == self.frame:
                        region = item[0]
                    elif item[1].frame_ == self.frame:
                        region = item[1]
                    else:
                        continue
                else:
                    region = item
                if region in self.items_nodes.keys():
                    continue

                if not isinstance(region, int):
                    img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    img[:,:,0] = 255
                    # img = self.im_manager.get_crop(self.frame, region,  width=self.width, height=self.height, relative_margin=self.relative_margin)
                    self.regions_images[region] = img

    def add_crop_to_col(self):
        for item in self.objects:
            if item not in self.items_nodes.keys():
                if not item:
                    continue
                if isinstance(item, tuple):
                    if item[0].frame_ == self.frame:
                        item = item[0]
                    elif item[1].frame_ == self.frame:
                        item = item[0]
                    else:
                        continue
                    if item in self.items_nodes.keys():
                        continue
                if item not in self.regions_images.keys():
                    img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    img[:, :, 0] = 255
                    self.regions_images[item] = img
                    # img = self.im_manager.get_crop(self.frame, item,  width=self.width, height=self.height, relative_margin=self.relative_margin)
                else:
                    img = self.regions_images[item]
                pixmap = cvimg2qtpixmap(img)
                node = Node(self.scene.addPixmap(pixmap), self.scene, item, self. im_manager, self.relative_margin,  self.width, self.height)
                node.parent_pixmap.hide()
                self.items_nodes[item] = node

    def set_x(self, x):
        self.x = x

    def draw(self, compress_axis, vertically, frames_columns):
        if self.empty:
            self.show_compress_marker(compress_axis, vertically)
        else:
            self.show_frame_number(vertically)
            for item in self.objects:
                if isinstance(item, Region):
                    self.show_node(item, vertically)
                elif isinstance(item, tuple):
                    if item[2] is "partial":
                        self.show_node()
                    if item[0].frame_ == self.frame:
                        self.show_node(item[0], vertically)
                    elif item[1].frame_ == self.frame:
                        self.show_node(item[1], vertically)
                        self.show_edge(item, frames_columns, vertically)

    def show_edge(self, edge, frame_columns, vertically, direction=None, node=None):
        from_x = self.x
        if node is None:
            node = edge[1]
        position = self.get_position_item(node)
        from_y = GAP + FROM_TOP + position * self.height + self.height / 2 + SPACE_BETWEEN_VER * position

        if edge[2] is not "partial":
            column_left = frame_columns[edge[0].frame_]
            position = column_left.get_position_item(edge[0])
            to_x = column_left.x + self.width
            to_y = GAP + FROM_TOP + position * self.height + self.height / 2 + SPACE_BETWEEN_VER * position
        else:
            to_y = from_y
            to_x = self.x - SPACE_BETWEEN_HOR / 2.5
            if not direction == "left":
                from_x += self.width
                to_x += self.width + SPACE_BETWEEN_HOR * 4 / 5.0

        if vertically:
            from_x, from_y, to_x, to_y = from_y, from_x, to_y, to_x

        if edge in self.edges.keys():
            self.scene.removeItem(self.edges[edge].graphical_object)
        edge_obj = Edge(from_x, from_y, to_x, to_y, edge, self.scene, vertically)
        self.edges[edge] = edge_obj

        if edge[2] is "chunk":
            edge_obj.graphical_object.setZValue(-1)
        elif edge[2] is "partial":
            edge_obj.graphical_object.setZValue(-2)
        else:
            edge_obj.graphical_object.setZValue(-3)

        self.scene.addItem(edge_obj.graphical_object)

    def show_node(self, region, vertically, compressed=True):
            position = self.get_position_item(region)
            x = self.x
            y = GAP + FROM_TOP + position * self.height + SPACE_BETWEEN_VER * position

            if vertically:
                x, y = y, x
            if region not in self.items_nodes.keys():
                if region not in self.regions_images.keys():
                    if compressed:
                        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        img[:, :, 0] = 255
                    else:
                        img = self.im_manager.get_crop(self.frame, region, width=self.width, height=self.height, relative_margin=self.relative_margin)
                    self.regions_images[region] = img
                else:
                    img = self.regions_images[region]
                pixmap = cvimg2qtpixmap(img)
                node = Node(self.scene.addPixmap(pixmap), self.scene, region, self.im_manager, self.relative_margin,  self.width, self.height)
                self.items_nodes[region] = node
            self.items_nodes[region].setPos(x, y)
            self.items_nodes[region].parent_pixmap.show()

    def show_compress_marker(self, compress_axis, vertically):
        if isinstance(self.frame, tuple):
            x = self.x + self.width / 4 - 12.5
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

    def show_frame_number(self, vertically, compress_axis=True, empty=False):
        text = str(self.frame)
        text_obj = QtGui.QGraphicsTextItem(text) if self.frame_sign is None else self.frame_sign
        y = FROM_TOP
        if empty:
            text_obj.setDefaultTextColor(QtGui.QColor(0, 0, 0, 120))
        x = self.x + self.width / (4 if empty else 2)

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