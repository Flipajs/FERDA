from gui.arena.my_ellipse import MyEllipse
from gui.arena.my_view    import MyView

__author__ = 'flipajs'

from PyQt4 import QtGui, QtCore, Qt
import cv2
import sys
import math
import copy
from core.project.project import Project
from gui.img_controls import utils
import numpy as np


class ArenaEditor(QtGui.QWidget):
    RED = 0
    GREEN = 1
    ALPHA = 3
    def __init__(self, img, project):
        super(ArenaEditor, self).__init__()

        self.setMouseTracking(True)

        self.background = img
        self.project = project

        self.view = MyView(update_callback_move=self.mouse_moving)
        self.scene = QtGui.QGraphicsScene()

        self.view.setScene(self.scene)
        self.scene.addPixmap(utils.cvimg2qtpixmap(img))
        self.view.setMouseTracking(True)

        # store the current paint mode "polygons" or "paint"
        self.mode = "polygons"

        ##########################
        #  PAINT MODE VARIABLES  #
        ##########################
        self.pen_color = QtCore.Qt.blue
        self.pen_size = 10
        self.color = "blue"
        self.last_pos = QtCore.QPoint()
        bg_height, bg_width = self.background.shape[:2]
        bg_size = QtCore.QSize(bg_width, bg_height)
        fmt = QtGui.QImage.Format_ARGB32
        self.image = QtGui.QImage(bg_size, fmt)
        self.image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.img_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.image))

        ##########################
        # POLYGON MODE VARIABLES #
        ##########################
        # MyEllipse[]
        # all independent points (they are not yet part of any polygon)
        self.point_items = []

        # QPolygonItem[]
        # holds instances of QPolygonItems
        self.polygon_items = []

        # MyEllipse[][]
        # holds sets of all used points. Each list corresponds to one polygon
        self.ellipses_items = []

        ##########################
        #          GUI           #
        ##########################

        self.setLayout(QtGui.QHBoxLayout())
        self.layout().setAlignment(QtCore.Qt.AlignBottom)

        # left panel widget
        widget = QtGui.QWidget()
        widget.setLayout(QtGui.QVBoxLayout())
        widget.layout().setAlignment(QtCore.Qt.AlignTop)
        #set width to 300px
        widget.setMaximumWidth(300)
        widget.setMinimumWidth(300)

        # SWITCH button and key shortcut
        self.action_switch = QtGui.QAction('switch', self)
        self.action_switch.triggered.connect(self.switch_mode)
        self.action_switch.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_X))
        self.addAction(self.action_switch)

        self.switch_button = QtGui.QPushButton("Switch modes \n (key_X)")
        self.switch_button.clicked.connect(self.switch_mode)
        widget.layout().addWidget(self.switch_button)

        # CLEAR button and key shortcut
        self.action_clear = QtGui.QAction('clear', self)
        self.action_clear.triggered.connect(self.reset)
        self.action_clear.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_C))
        self.addAction(self.action_clear)

        self.clear_button = QtGui.QPushButton("Clear paint area \n (key_C)")
        self.clear_button.clicked.connect(self.reset)
        widget.layout().addWidget(self.clear_button)

        self.label = QtGui.QLabel()
        self.set_label_text()
        widget.layout().addWidget(self.label)

        # DRAW button and key shortcut
        self.action_paint_polygon = QtGui.QAction('paint_polygon', self)
        self.action_paint_polygon.triggered.connect(self.paint_polygon)
        self.action_paint_polygon.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_D))
        self.addAction(self.action_paint_polygon)
        self.poly_button = QtGui.QPushButton("Draw polygons \n (key_D)")
        self.poly_button.clicked.connect(self.paint_polygon)
        widget.layout().addWidget(self.poly_button)

        # COLOR SWITCH button (no shortcut yet)
        self.color_button = QtGui.QPushButton("Switch color to red")
        self.color_button.clicked.connect(self.switch_color)
        self.color_button.setVisible(False)
        widget.layout().addWidget(self.color_button)

        # PEN SIZE slider
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.setGeometry(30, 40, 50, 30)
        self.slider.setRange(10, 50)
        self.slider.setTickInterval(5)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider.valueChanged[int].connect(self.change_value)
        self.slider.setVisible(False)
        widget.layout().addWidget(self.slider)

        # complete the gui
        self.layout().addWidget(widget)
        self.layout().addWidget(self.view)

    def switch_mode(self):
        # switch modes
        if self.mode == "polygons":
            self.mode = "paint"
            # display only the necessary widgets in the left panel
            self.poly_button.setVisible(False)
            self.color_button.setVisible(True)
            self.slider.setVisible(True)
        else:
            self.mode = "polygons"
            self.poly_button.setVisible(True)
            self.color_button.setVisible(False)
            self.slider.setVisible(False)
        # refresh text in QLabel
        self.set_label_text()

    def switch_color(self):
        # change colors
        if self.color == "blue":
            self.color = "red"
            self.color_button.setText("Switch color to blue")
        else:
            self.color = "blue"
            self.color_button.setText("Switch color to red")
        # refresh text in QLabel
        self.set_label_text()

    def change_value(self, value):
        # change pen size
        self.pen_size = value
        # refresh text in QLabel
        self.set_label_text()

    def set_label_text(self):
        if self.mode == "polygons":
            self.label.setText("Mode: Polygons")
        else:
            self.label.setText("Mode: Paint \nColor: %s \nPen size: %s" % (self.color, self.pen_size))

    def clear_image(self):
        # remove all drawn lines
        self.image.fill(QtGui.qRgba(0, 0, 0, 0))
        # remove the old pixmap from scene
        self.scene.removeItem(self.img_pixmap)
        # create a new pixmap
        self.img_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.image))

    def clear(self):
        print "Clearing the area"

        # erase all polygons
        for polygon_item in self.polygon_items:
            self.scene.removeItem(polygon_item)

        # erase all points from polygons
        for point_items in self.ellipses_items:
            for point in point_items:
                self.scene.removeItem(point)

        # erase all independent points
        for point in self.point_items:
            self.scene.removeItem(point)

        # clear the image
        self.clear_image()

    def reset(self):
        # clear view
        self.clear()

        # clear all lists
        self.point_items = []
        self.ellipses_items = []
        self.polygon_items = []

    def mousePressEvent(self, event):
        # get event position and calibrate to scene
        cursor = QtGui.QCursor()
        pos = cursor.pos()
        pos = self.get_scene_pos(pos)
        
        if self.mode == "polygons":
            # in the polygons mode, try to pick one point
            precision = 25
            ok = True
            for pt in self.point_items:
                # check if the clicked pos isn't too close to any other already chosen point
                dist = self.get_distance(pt, pos)
                if dist < precision:
                    ok = False
            if ok:
                self.point_items.append(self.pick_point(pos, 10))
        else:
            # in the paint mode, paint the event position
            self.draw(pos)

    def mouse_moving(self, event):
        if self.mode == "paint":
            # while the mouse is moving, paint it's position
            point = self.view.mapToScene(event.pos())
            if self.is_in_scene(point):
                self.draw(point)
        # do nothing in "polygons" mode

    def draw(self, point):
        # change float to int (QPointF -> QPoint)
        if type(point) == QtCore.QPointF:
            point = point.toPoint()

        # use current pen color
        if self.color == "blue":
            value = QtGui.qRgba(0, 0, 255, 100)
        else:
            value = QtGui.qRgba(255, 0, 0, 100)

        # paint the area around the point position
        for i in range(point.x() - self.pen_size/2, point.x() + self.pen_size/2):
            for j in range(point.y() - self.pen_size/2, point.y() + self.pen_size/2):
                self.image.setPixel(i, j, value)

        # set new image and pixmap
        self.scene.removeItem(self.img_pixmap)
        self.img_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.image))

    def get_distance(self, pt_a, pt_b):
        # simple method that returns the absolute distance of two points
        return math.sqrt((pt_b.x() - pt_a.x()) ** 2 + (pt_b.y() - pt_a.y()) ** 2)

    def pick_point(self, position, size):
        # picks and marks a point in the polygon mode
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        ellipse = MyEllipse(update_callback=self.repaint_polygons)
        ellipse.setBrush(brush)
        ellipse.setPos(QtCore.QPoint(position.x(), position.y()))
        self.scene.addItem(ellipse)
        return ellipse

    def paint_polygon(self):
        # check if polygon can be created
        if len(self.point_items) > 2:
            print "Polygon complete, drawing it"

            # create the polygon
            polygon = QtGui.QPolygonF()
            for el in self.point_items:
                # use all selected points
                polygon.append(QtCore.QPointF(el.x(), el.y()))

            # input the polygon into all polygons' list and draw it
            self.polygon_items.append(self.paint_polygon_(polygon))

            # store all the points (ellipses), too
            self.ellipses_items.append(self.point_items)

            # clear temporary points' storage
            self.point_items = []
        else:
            print "Polygon is too small, pick at least 3 points"

    def paint_polygon_(self, polygon):
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.Dense5Pattern)
        return self.scene.addPolygon(polygon, brush=brush)

    def repaint_polygons(self, my_ellipse):
        # clear the canvas
        self.clear()

        # self.debug()
        self.polygon_items = []
        tmp_ellipses = []
        tmp_points = []

        # go through all saved points and recreate the polygons according to the new points' position
        for points in self.ellipses_items:
            polygon = QtGui.QPolygonF()
            tmp_ellipse = []
            for point in points:
                qpt = QtCore.QPointF(point.x(), point.y())
                polygon.append(qpt)
                tmp_ellipse.append(self.pick_point(qpt, 10))
            self.polygon_items.append(self.paint_polygon_(polygon))
            tmp_ellipses.append(tmp_ellipse)
        self.ellipses_items = tmp_ellipses

        for point in self.point_items:
            pos = QtCore.QPoint(point.x(), point.y())
            tmp_points.append(self.pick_point(pos, 10))
        self.point_items = tmp_points

    def get_scene_pos(self, point):
        map_pos = self.view.mapFromGlobal(point)
        scene_pos = self.view.mapFromScene(QtCore.QPoint(0, 0))
        map_pos.setY(map_pos.y() - scene_pos.y())
        map_pos.setX(map_pos.x() - scene_pos.x())
        if self.is_in_scene(map_pos):
            return map_pos
        else:
            print "Out of bounds [%s, %s]" % (map_pos.x(), map_pos.y())
            return QtCore.QPoint(0, 0)

    def is_in_scene(self, point):
        height, width = self.background.shape[:2]
        if self.scene.itemsBoundingRect().contains(point) and point.x() <= width and point.y() <= height:
            return True
        else:
            return False

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    im = cv2.imread('/home/dita/PycharmProjects/sample2.png')
    # im = cv2.imread('/Users/flipajs/Desktop/red_vid.png')
    p = Project()

    ex = ArenaEditor(im, p)
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()
