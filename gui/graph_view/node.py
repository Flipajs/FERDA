from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
from gui.img_controls.utils import cvimg2qtpixmap
from vis_loader import WIDTH, HEIGHT, RELATIVE_MARGIN

SELECTION_LINE_WIDTH = 2

__author__ = 'Simon Mandlik'


class Node(QtGui.QGraphicsPixmapItem):

    def __init__(self, parent_pixmap, scene, region, img_manager, relative_margin, width, height):
        super(Node, self).__init__(parent_pixmap)
        self.region = region
        self.img_manager = img_manager
        self.scene = scene
        self.parent_pixmap = parent_pixmap
        self.pixmap_toggled = None
        self.width = width
        self.height = height
        self.relative_margin = relative_margin
        self.x = self.parent_pixmap.offset().x()
        self.y = self.parent_pixmap.offset().y()
        self.setFlags(QtGui.QGraphicsItem.ItemIsSelectable)
        self.selection_polygon = self.create_selection_polygon()

        self.toggled = False
        self.clipped = False
        self.color = None

        self.info_item = None
        self.info_rect = None

    def toggle(self):
        if not self.toggled:
            if self.pixmap_toggled is None:
                self.create_pixmap()
            else:
                self.scene.addItem(self.pixmap_toggled)
        else:
            self.scene.removeItem(self.pixmap_toggled)
        self.toggled = False if self.toggled else True

    def show_info(self):
        self.clipped = True
        if not self.info_item:
            self.create_info()
        self.scene.addItem(self.info_item)

    def hide_info(self):
        self.scene.removeItem(self.info_item)
        self.clipped = False

    def create_info(self):
        r = self.region

        # vertex = self.project.gm.g.vertex(int(n))
        # best_out_score, _ = self.project.gm.get_2_best_out_vertices(vertex)
        # best_out = best_out_score[0]
        #
        # best_in_score, _ = self.project.gm.get_2_best_in_vertices(vertex)
        # best_in = best_in_score[0]
        #
        # ch = self.project.gm.is_chunk(vertex)
        # ch_info = str(ch)

        # QtGui.QMessageBox.about(self, "My message box",
        #                         "Area = %i\nCentroid = %s\nMargin = %i\nAntlikeness = %f\nIs virtual: %s\nBest in = %s\nBest out = %s\nChunk info = %s" % (r.area(), str(r.centroid()), r.margin_, antlikeness, str(virtual), str(best_in_score[0])+', '+str(best_in_score[1]), str(best_out_score[0])+', '+str(best_out_score[1]), ch_info))
        x, y = self.compute_rectangle_size()
        self.info_item = TextInfoItem("hulala", x, y, self.color, self)
        self.info_item.setFlags(QtGui.QGraphicsItem.ItemIsMovable)

    def color_margins(self, color):
        self.color = color
        self.scene.update()

    def decolor_margins(self):
        self.color = None
        self.scene.update()

    def create_pixmap(self):
        img_toggled = self.img_manager.get_crop(self.region.frame_, self.region, width=self.width * 3, height=self.height * 3, relative_margin=self.relative_margin)
        pixmap = cvimg2qtpixmap(img_toggled)
        self.pixmap_toggled = self.scene.addPixmap(pixmap)
        multiplier_x, multiplier_y = self.compute_rectangle_size()
        self.pixmap_toggled.setPos(self.x + (multiplier_x) * self.width / 2,
                                   self.y + (multiplier_y) * self.height / 2)
        self.pixmap_toggled.setFlags(QtGui.QGraphicsItem.ItemIsMovable)

    def compute_rectangle_size(self):
        width, height = self.scene.width(), self.scene.height()
        multiplier_x = 1 if self.x < width / 2 else -5
        multiplier_y = 1 if self.y < height / 2 else -5
        return multiplier_x, multiplier_y

    def setPos(self, x, y):
        if not(x == self.x and y == self.y):
            self.x, self.y = x, y
            self.parent_pixmap.setPos(x, y)
            if self.pixmap_toggled is not None:
                self.pixmap_toggled.setPos(x + WIDTH / 2, y + HEIGHT / 2)

    def paint(self, painter, style_option_graphics_item, widget=None):
        self.parent_pixmap.paint(painter, style_option_graphics_item, None)
        if self.clipped:
            pen = QtGui.QPen(self.color, SELECTION_LINE_WIDTH * 1.5, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin)
        elif self.isSelected():
            pen = QtGui.QPen(Qt.black, SELECTION_LINE_WIDTH, Qt.DashLine, Qt.SquareCap, Qt.RoundJoin)
        else:
            pen = QtGui.QPen(Qt.white, SELECTION_LINE_WIDTH, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin)

        painter.setPen(pen)
        painter.drawPolygon(self.selection_polygon)

    def create_selection_polygon(self):
        p1 = QtCore.QPointF(self.x, self.y)
        p2 = QtCore.QPointF(self.x + self.width, self.y)
        p4 = QtCore.QPointF(self.x, self.y + self.height)
        p3 = QtCore.QPointF(self.x + self.width, self.y + self.height)
        polygon = QtGui.QPolygonF()
        polygon << p1 << p2 << p3 << p4
        return polygon

    def boundingRect(self):
        return self.selection_polygon.boundingRect()

    def shape(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.selection_polygon)
        return path


class TextInfoItem(QtGui.QGraphicsItem):

    def __init__(self, text, x, y, color, node):
        super(TextInfoItem, self).__init__()
        self.color = color
        self.node = node
        self.x = x
        self.y = y
        self.bounding_rect = self.create_bounding_rect()
        self.rect = self.create_rectangle()
        self.text = self.create_text(text)
        self.text.setPos(self.x, self.y)

    def set_color(self, color):
        self.color = color

    def paint(self, painter, item, widget):
        painter.setBrush(self.node.color)
        painter.setBrush(self.node.color)
        self.rect.paint(painter, item, widget)
        self.text.paint(painter, item, widget)

    def create_bounding_rect(self):
        return QtCore.QRectF(QtCore.QPointF(self.x, self.y), QtCore.QPointF(self.x + self.node.width * 3, self.y + self.node.height * 2))

    def create_rectangle(self):
        return QtGui.QGraphicsRectItem(self.bounding_rect, self)

    def create_text(self, text):
        return QtGui.QGraphicsTextItem(text, self.rect)

    def boundingRect(self):
        return self.bounding_rect


