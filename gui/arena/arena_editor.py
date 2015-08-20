from gui.arena.my_ellipse import MyEllipse

__author__ = 'flipajs'

from PyQt4 import QtGui, QtCore, Qt
import cv2
import sys
import math
from core.project.project import Project
from gui.img_controls import utils


class ArenaEditor(QtGui.QWidget):
    def __init__(self, img, project):
        super(ArenaEditor, self).__init__()

        # TODO: 1) fix the 'get_scene_pos() function', it behaves strangely and sometimes returns wrong numbers
        # TODO: 2) when independent points are moved, they get erased
        # TODO: 3) the callback in 'my_ellipse' had to be changed to 'mouseReleaseEvent', otherwise it doesn't work

        self.img = img
        self.project = project

        self.setLayout(QtGui.QVBoxLayout())

        self.view = QtGui.QGraphicsView()
        self.scene = QtGui.QGraphicsScene()

        self.view.setScene(self.scene)
        self.scene.addPixmap(utils.cvimg2qtpixmap(img))

        # MyEllipse[]
        # all independent points (they are not yet part of any polygon)
        self.point_items = []

        # QPolygonItem[]
        # holds instances of QPolygonItems
        self.polygon_items = []

        # MyEllipse[][]
        # holds sets of all used points. Each list corresponds to one polygon
        self.ellipses_items = []

        # TODO: mode switcher - polygons x paint
        # self.mode = "polygons"
        # self.mode_item =

        # draw chosen polygon when 'D' is pressed
        self.action_paint_polygon = QtGui.QAction('paint_polygon', self)
        self.action_paint_polygon.triggered.connect(self.paint_polygon)
        self.action_paint_polygon.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_D))
        self.addAction(self.action_paint_polygon)

        # clear all that has been drawn and all selected points when 'C' is pressed
        self.action_clear = QtGui.QAction('clear', self)
        self.action_clear.triggered.connect(self.reset)
        self.action_clear.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_C))
        self.addAction(self.action_clear)

        # switch modes (polygons x paint) when 'X' is pressed
        self.action_switch = QtGui.QAction('switch', self)
        self.action_switch.triggered.connect(self.switch)
        self.action_switch.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_X))
        self.addAction(self.action_switch)

        self.layout().addWidget(self.view)

    def switch(self):
        if self.mode == "polygons":
            self.mode = "paint"
        else:
            self.mode = "polygons"

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

        # self.debug()

    def reset(self):
        # clear view
        self.clear()

        # clear all lists
        self.point_items = []
        self.ellipses_items = []
        self.polygon_items = []

        # self.debug()

    def mousePressEvent(self, event):
        cursor = QtGui.QCursor()
        # pos = self.get_scene_pos(cursor.pos())
        pos = cursor.pos()
        precision = 20

        ok = True
        """
        for pt in self.point_items:
            # check if the clicked pos isn't too close to any other already chosen point
            dist = self.get_distance(pt, pos)
            # print "Distance is: %s" % dist
            if dist < precision:
                ok = False
        """
        if ok:
            self.point_items.append(self.paint_point(self.get_scene_pos(pos), precision))
            # print "Adding [%s, %s] to points" % (pos.x(), pos.y())
        # else:
            # print "Point [%s, %s] has already been chosen, ignoring" % (pos.x(), pos.y())

    def get_distance(self, pt_a, pt_b):
        return math.sqrt((pt_b.x() - pt_a.x()) ** 2 + (pt_b.y() - pt_a.y()) ** 2)

    def paint_point(self, position, size):
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        ellipse = MyEllipse(QtGui.QGraphicsEllipseItem(position.x() - size / 2, position.y() - size / 2, size, size),
                            update_callback=self.repaint_polygons)
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
        new_pos = QtCore.QPoint(my_ellipse.x(), my_ellipse.y())
        my_ellipse.setPos(new_pos)

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
                tmp_ellipse.append(self.paint_point(qpt, 10))
            self.polygon_items.append(self.paint_polygon_(polygon))
            tmp_ellipses.append(tmp_ellipse)
        self.ellipses_items = tmp_ellipses

        for point in self.point_items:
            tmp_points.append(self.paint_point(point, 10))
        self.point_items = tmp_points
        # self.debug()

    def get_scene_pos(self, point):
        map_pos = self.view.mapFromGlobal(point)
        scene_pos = self.view.mapFromScene(QtCore.QPoint(0, 0))
        map_pos.setY(map_pos.y() - scene_pos.y())
        map_pos.setX(map_pos.x() - scene_pos.x())
        print "Adjusting position of [%s, %s] to [%s, %s]" % (point.x(), point.y(), map_pos.x(), map_pos.y())
        return map_pos

    def debug(self):
        print "Polygons: %s" % self.polygon_items
        print "Points: %s" % self.point_items
        print "Ellipses: %s" % self.ellipses_items


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    im = cv2.imread('/home/dita/PycharmProjects/sample2.png')
    p = Project()

    ex = ArenaEditor(im, p)
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()
