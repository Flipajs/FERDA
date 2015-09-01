from gui.arena.my_ellipse import MyEllipse
from gui.arena.my_popup   import MyPopup
from gui.arena.my_view    import MyView

__author__ = 'dita'

from PyQt4 import QtGui, QtCore, Qt
import cv2
import sys
import math
from core.project.project import Project
from gui.img_controls     import utils
import numpy as np


class ArenaEditor(QtGui.QWidget):
    DEBUG = True

    def __init__(self, img, project):
        # TODO: add support for the original arena editor (circle)

        super(ArenaEditor, self).__init__()

        self.setMouseTracking(True)

        self.background = img
        self.project = project

        self.view = MyView(update_callback_move=self.mouse_moving)
        self.scene = QtGui.QGraphicsScene()

        self.view.setScene(self.scene)

        # background image
        self.scene.addPixmap(utils.cvimg2qtpixmap(self.background))
        self.view.setMouseTracking(True)

        # store the current paint mode "polygons" or "paint" or "circles"
        self.mode = ""
        self.color = "Blue"

        # store last 10 QImages to support the "undo" function
        # undo button can only be pushed in paint mode, but affects polygon painting too (all polygons are saved
        #    as one step)
        self.backup = []

        ##########################
        #  PAINT MODE VARIABLES  #
        ##########################
        self.pen_size = 30
        # image to store all progress
        bg_height, bg_width = self.background.shape[:2]
        bg_size = QtCore.QSize(bg_width, bg_height)
        fmt = QtGui.QImage.Format_ARGB32
        self.paint_image = QtGui.QImage(bg_size, fmt)
        self.paint_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.paint_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.paint_image))
        self.save()

        ##########################
        # POLYGON MODE VARIABLES #
        ##########################
        # all independent points (they are not yet part of any polygon)
        self.point_items = [] # type MyEllipse[]

        # holds colors of polygons "Red" or "Blue"
        self.polygon_colors = [] # type string[]

        # holds sets of all used points. Each list corresponds to one polygon
        self.polygon_points = [] # type MyEllipse[][]

        # temporary image to work with polygons
        self.poly_image = QtGui.QImage(bg_size, fmt)
        self.poly_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.poly_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.poly_image))

        ##########################
        # CIRCLES MODE VARIABLES #
        ##########################

        # TODO: add circles mode variables

        # temporary image to work with circles
        self.circles_image = QtGui.QImage(bg_size, fmt)
        self.circles_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.circles_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.circles_image))

        # create the main view and left panel with buttons
        self.make_gui()

    def switch_color(self):
        text = self.sender().text()
        if self.DEBUG:
            print "Setting color to %s" % text
        # make sure no other button stays pushed
        for button in self.color_buttons:
            if button.text() != text:
                button.setChecked(False)
            else:
                button.setChecked(True)
        self.color = text

    def switch_mode(self):
        value = self.sender().text()
        if value == "Paint mode":
            # don't do anything if paint mode is already active
            if self.mode == "paint":
                return
            self.set_paint_mode()

        elif value == "Polygon mode":
            # don't do anything if polygons mode is already active
            if self.mode == "polygons":
                return
            self.set_polygons_mode()

        else:
            # don't do anything if polygons mode is already active
            if self.mode == "circles":
                return
            self.set_circles_mode()

    def set_paint_mode(self):

        # join the temporary images with paint image
        self.save()
        self.refresh_image(self.merge_images())
        # clean the temporary images
        self.clear_poly_image()
        self.clear_circles_image()

        # clean after polygon drawing
        # - remove all points and polygons
        self.remove_items()
        # - clear the memory
        self.point_items = []
        self.polygon_points = []
        self.polygon_colors = []

        # TODO: add cleanup after circle drawing

        self.mode = "paint"

        # adjust widgets displayed in the left panel
        self.poly_button.setVisible(False)
        self.undo_button.setVisible(True)
        self.slider.setVisible(True)
        for button in self.color_buttons:
            button.setVisible(True)
        self.clear_button.setVisible(True)
        self.popup_button.setVisible(True)
        self.pen_label.setVisible(True)
        self.circle_label.setVisible(False)
        self.set_label_text()

    def set_polygons_mode(self):

        self.mode = "polygons"

        # TODO: add cleanup after circle drawing

        # cleanup after paint mode is not needed

        self.poly_button.setVisible(True)
        self.undo_button.setVisible(False)
        self.slider.setVisible(False)
        for button in self.color_buttons:
            button.setVisible(True)
        # hide "Eraser" button, there are no erasers in polygons mode
        self.color_buttons[2].setVisible(False)
        # in case "Eraser" was chosen as a color, switch it to blue
        if self.color == "Eraser":
            self.color = "Blue"
            self.color_buttons[0].setChecked(True)
            self.color_buttons[2].setChecked(False)
        self.clear_button.setVisible(True)
        self.popup_button.setVisible(True)
        self.pen_label.setVisible(False)
        self.circle_label.setVisible(False)

    def set_circles_mode(self):

        # join the temporary images with paint image
        self.save()
        self.refresh_image(self.merge_images())
        # clean the temporary images
        self.clear_poly_image()
        self.clear_circles_image()

        # clean after polygon drawing
        # - remove all points and polygons
        self.remove_items()
        # - clear the memory
        self.point_items = []
        self.polygon_points = []
        self.polygon_colors = []

        # cleanup after paint mode is not needed

        self.mode = "circles"
        self.poly_button.setVisible(False)
        self.undo_button.setVisible(False)
        self.slider.setVisible(False)
        for button in self.color_buttons:
            button.setVisible(False)
        self.clear_button.setVisible(False)
        self.popup_button.setVisible(False)
        self.pen_label.setVisible(False)
        self.circle_label.setVisible(True)

    def popup(self):
        """
        converts image to numpy arrays
        :return: tuple (arena_mask, occultation_mask)
        True in arena_masks means that the point is INSIDE the arena
        True in occultation_mask means that the point is a place to hide
        """
        r = QtGui.qRgba(255, 0, 0, 255)
        b = QtGui.qRgba(0, 0, 255, 255)
        p = QtGui.qRgba(175, 0, 175, 255)
        img = self.merge_images()

        bg_height, bg_width = self.background.shape[:2]

        arena_mask = np.zeros((bg_height, bg_width), dtype=np.bool)
        occultation_mask = np.zeros((bg_height, bg_width), dtype=np.bool)

        for i in range(0, bg_width):
            for j in range(0, bg_height):
                color = QtGui.QColor(img.pixel(i, j))
                if self.DEBUG:
                    if color.blue() > 250:
                        img.setPixel(i, j, b)
                    if color.red() > 250:
                        img.setPixel(i, j, r)
                    if color.red() > 250 and color.blue() > 250:
                        img.setPixel(i, j, p)
                if color.blue() > 250:
                    occultation_mask[j, i] = True
                if color.red() > 250:
                    arena_mask[j, i] = True
        if self.DEBUG:
            self.w = MyPopup(img)
            self.w.show()
            self.w.showMaximized()
            self.w.setFocus()
        return arena_mask, occultation_mask

    def change_value(self, value):
        """
        change pen size
        :param value: new pen size
        :return: None
        """
        # change pen size
        self.pen_size = value
        # refresh text in QLabel
        self.set_label_text()

    def set_label_text(self):
        """
        changes the label to show current pen settings
        :return: None
        """
        if self.mode == "paint":
            self.pen_label.setText("Pen size: %s" % self.pen_size)

    def remove_items(self):
        """
        remove all points from polygons mode from the scene
        :return:
        """
        # erase all points from polygons
        for point_items in self.polygon_points:
            for point in point_items:
                self.scene.removeItem(point)

        # erase all independent points
        for point in self.point_items:
            self.scene.removeItem(point)

    def reset(self):
        """
        clear everything and start over
        :return: None
        """
        self.remove_items()
        self.clear_poly_image()
        self.clear_paint_image()
        self.point_items = []
        self.polygon_points = []
        self.polygon_colors = []

    def mousePressEvent(self, event):
        # get event position and calibrate to scene
        cursor = QtGui.QCursor()
        pos = cursor.pos()
        pos = self.get_scene_pos(pos)
        if type(pos) != QtCore.QPoint:
            return

        if self.mode == "polygons":
            # in the polygons mode, try to pick one point
            precision = 20
            ok = True
            for pt in self.point_items:
                # check if the clicked pos isn't too close to any other already chosen point
                dist = self.get_distance(pt, pos)
                if dist < precision:
                    ok = False
            for points in self.polygon_points:
                for pt in points:
                    dist = self.get_distance(pt, pos)
                    if dist < precision:
                        ok = False
            if ok:
                self.point_items.append(self.pick_point(pos))
        elif self.mode == "paint":
            # in the paint mode, paint the event position
            self.save()
            self.draw(pos)

    def mouseReleaseEvent(self, event):
        self.save()

    def mouse_moving(self, event):
        if self.mode == "paint":
            # while the mouse is moving, paint it's position
            point = self.view.mapToScene(event.pos())
            if self.is_in_scene(point):
                self.draw(point)
        # do nothing in "polygons" mode

    def save(self):
        """
        Saves current image temporarily (to use with "undo()" later)
        :return:
        """
        # save last 10 images
        img = self.paint_image.copy()
        self.backup.append(img)
        if len(self.backup) > 10:
            self.backup.pop(0)

    def undo(self):
        if self.mode == "paint":
            lenght = len(self.backup)
            if lenght > 0:
                img = self.backup.pop(lenght-1)
                self.refresh_image(img)

    def refresh_image(self, img):
        self.paint_image = img
        self.scene.removeItem(self.paint_pixmap)
        self.paint_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(img))

    def refresh_poly_image(self):
        self.scene.removeItem(self.poly_pixmap)
        self.poly_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.poly_image))

    def refresh_circles_image(self):
        self.scene.removeItem(self.circles_pixmap)
        self.circles_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.circles_image))

    def clear_paint_image(self):
        # remove all drawn lines
        self.paint_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.refresh_image(self.paint_image)

    def clear_poly_image(self):
        # remove all drawn lines
        self.poly_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.refresh_poly_image()

    def clear_circles_image(self):
        # remove all drawn lines
        self.circles_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.refresh_circles_image()

    def draw(self, point):
        """
        paint a point with a pen in paint mode
        :param point: point to be drawn
        :return: None
        """
        # change float to int (QPointF -> QPoint)
        if type(point) == QtCore.QPointF:
            point = point.toPoint()

        # use current pen color
        if self.color == "Blue":
            value = QtGui.qRgba(0, 0, 255, 100)
        elif self.color == "Red":
            value = QtGui.qRgba(255, 0, 0, 100)
        else:
            value = QtGui.qRgba(0, 0, 0, 0)

        # paint the area around the point position
        bg_height, bg_width = self.background.shape[:2]
        for i in range(point.x() - self.pen_size/2, point.x() + self.pen_size/2):
            for j in range(point.y() - self.pen_size/2, point.y() + self.pen_size/2):
                if i >= 0 and i <= bg_width and j >= 0 and j <= bg_height:
                    self.paint_image.setPixel(i, j, value)

        # set new image and pixmap
        self.refresh_image(self.paint_image)

    def get_distance(self, pt_a, pt_b):
        """
        simple method that returns the distance of two points (A, B)
        :param pt_a: Point A
        :param pt_b: Point B
        :return: float distance
        """
        return math.sqrt((pt_b.x() - pt_a.x()) ** 2 + (pt_b.y() - pt_a.y()) ** 2)

    def pick_point(self, position):
        """
        create a point that the user has chosen to be a future part of a polygon
        :param position:
        :return: QGraphicsItem (from MyEllipse)
        """
        # picks and marks a point in the polygon mode
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        ellipse = MyEllipse(update_callback=self.repaint_polygons)
        ellipse.setBrush(brush)
        ellipse.setPos(QtCore.QPoint(position.x(), position.y()))
        self.scene.addItem(ellipse)
        return ellipse

    def paint_polygon(self):
        """
        tries to create a new polygon from currently selected points (MyEllipses)
        :return: bool, whether the polygon was drawn
        """
        # check if polygon can be created
        if len(self.point_items) > 2:
            if self.DEBUG:
                print "Polygon complete, drawing it"

            # create the polygon
            polygon = QtGui.QPolygonF()
            for el in self.point_items:
                # use all selected points
                polygon.append(QtCore.QPointF(el.x(), el.y()))

            # draw the polygon and save it's color
            self.paint_polygon_(polygon, self.color)
            self.polygon_colors.append(self.color)

            # store all the points (ellipses), too
            self.polygon_points.append(self.point_items)

            # clear temporary points' storage
            self.point_items = []

            return True
        else:
            if self.DEBUG:
                print "Polygon is too small, pick at least 3 points"
            return False

    def paint_polygon_(self, polygon, color):
        """
        paints a polygon, when it's color and position are known
        :param polygon: QPolygonF to be painted
        :param color: "Red" or "Blue"
        :return: None
        """
        # setup the painter
        painter = QtGui.QPainter()
        painter.begin(self.poly_image)
        brush = QtGui.QBrush()
        # paint the polygon
        if color == "Red":
            qc = QtGui.QColor(255, 0, 0, 100)
        else:
            qc = QtGui.QColor(0, 0, 255, 100)
        pen = QtGui.QPen(qc)
        brush.setColor(qc)
        brush.setStyle(QtCore.Qt.SolidPattern)
        painter.setBrush(brush)
        painter.setPen(pen)
        painter.drawPolygon(polygon)
        painter.end()
        # refresh the image
        self.refresh_poly_image()

    def repaint_polygons(self):
        """
        repaints all the polygons that are now changeable.
        :return: None
        """
        # clear the canvas
        self.remove_items()
        self.clear_poly_image()

        tmp_ellipses = []
        tmp_points = []

        # go through all saved points and recreate the polygons according to the new points' position
        i = 0
        for points in self.polygon_points:
            polygon = QtGui.QPolygonF()
            tmp_ellipse = []
            for point in points:
                qpt = QtCore.QPointF(point.x(), point.y())
                polygon.append(qpt)
                tmp_ellipse.append(self.pick_point(qpt))
            self.paint_polygon_(polygon, self.polygon_colors[i])
            i += 1
            tmp_ellipses.append(tmp_ellipse)
        self.polygon_points = tmp_ellipses

        for point in self.point_items:
            pos = QtCore.QPoint(point.x(), point.y())
            tmp_points.append(self.pick_point(pos))
        self.point_items = tmp_points


    def merge_images(self):
        """
        merges the 3 images (paint, polygons and circles) into one result
        :return: the final image
        """
        bg_height, bg_width = self.background.shape[:2]
        bg_size = QtCore.QSize(bg_width, bg_height)
        fmt = QtGui.QImage.Format_ARGB32
        result = QtGui.QImage(bg_size, fmt)
        result.fill(QtGui.qRgba(0, 0, 0, 0))
        p = QtGui.QPainter()
        p.begin(result)
        p.drawImage(0, 0, self.poly_image)
        p.drawImage(0, 0, self.paint_image)
        p.drawImage(0, 0, self.circles_image)
        p.end()
        return result

    def get_scene_pos(self, point):
        """
        converts point coordinates to scene coordinate system
        :param point: QPoint or QPointF
        :return: QPointF or False
        """
        map_pos = self.view.mapFromGlobal(point)
        scene_pos = self.view.mapFromScene(QtCore.QPoint(0, 0))
        map_pos.setY(map_pos.y() - scene_pos.y())
        map_pos.setX(map_pos.x() - scene_pos.x())
        if self.is_in_scene(map_pos):
            return map_pos
        else:
            if self.DEBUG:
                print "Out of bounds [%s, %s]" % (map_pos.x(), map_pos.y())
            return False

    def is_in_scene(self, point):
        """
        checks if the point is inside the scene
        :param point: Qpoint or QPointF
        :return: True or False
        """
        height, width = self.background.shape[:2]
        if self.scene.itemsBoundingRect().contains(point) and point.x() <= width and point.y() <= height:
            return True
        else:
            return False

    def make_gui(self):
        """
        Creates the widget. It is a separate method purely to save space
        :return: None
        """

        ##########################
        #          GUI           #
        ##########################

        self.setLayout(QtGui.QHBoxLayout())
        self.layout().setAlignment(QtCore.Qt.AlignBottom)

        # left panel widget
        widget = QtGui.QWidget()
        widget.setLayout(QtGui.QVBoxLayout())
        widget.layout().setAlignment(QtCore.Qt.AlignTop)
        # set left panel widget width to 300px
        widget.setMaximumWidth(300)
        widget.setMinimumWidth(300)

        label = QtGui.QLabel()
        label.setWordWrap(True)
        label.setText("Welcome to arena editor! Paint the outside of the arena with red and use blue to mark possible"
                      " hiding places. Unresolvable colors will be considered red.")
        widget.layout().addWidget(label)

        # SWITCH button and key shortcut
        mode_switch_group = QtGui.QButtonGroup(widget)

        polymode_button = QtGui.QRadioButton("Polygon mode")
        mode_switch_group.addButton(polymode_button)
        polymode_button.toggled.connect(self.switch_mode)
        widget.layout().addWidget(polymode_button)

        paintmode_button = QtGui.QRadioButton("Paint mode")
        mode_switch_group.addButton(paintmode_button)
        paintmode_button.toggled.connect(self.switch_mode)
        widget.layout().addWidget(paintmode_button)

        circlemode_button = QtGui.QRadioButton("Circles mode")
        mode_switch_group.addButton(circlemode_button)
        circlemode_button.toggled.connect(self.switch_mode)
        widget.layout().addWidget(circlemode_button)

        # color switcher widget
        color_widget = QtGui.QWidget()
        color_widget.setLayout(QtGui.QHBoxLayout())

        self.color_buttons = []
        blue_button = QtGui.QPushButton("Blue")
        blue_button.setCheckable(True)
        blue_button.setChecked(True)
        blue_button.clicked.connect(self.switch_color)
        color_widget.layout().addWidget(blue_button)
        self.color_buttons.append(blue_button)

        red_button = QtGui.QPushButton("Red")
        red_button.setCheckable(True)
        red_button.clicked.connect(self.switch_color)
        color_widget.layout().addWidget(red_button)
        self.color_buttons.append(red_button)

        eraser_button = QtGui.QPushButton("Eraser")
        eraser_button.setCheckable(True)
        eraser_button.clicked.connect(self.switch_color)
        color_widget.layout().addWidget(eraser_button)
        self.color_buttons.append(eraser_button)

        widget.layout().addWidget(color_widget)

        self.pen_label = QtGui.QLabel()
        self.pen_label.setWordWrap(True)
        self.pen_label.setText("")
        widget.layout().addWidget(self.pen_label)

        self.circle_label = QtGui.QLabel()
        self.circle_label.setWordWrap(True)
        self.circle_label.setText("Sorry, not supported yet")
        widget.layout().addWidget(self.circle_label)

        # PEN SIZE slider
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.setGeometry(30, 40, 50, 30)
        self.slider.setRange(10, 50)
        self.slider.setTickInterval(5)
        self.slider.setValue(30)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider.valueChanged[int].connect(self.change_value)
        self.slider.setVisible(False)
        widget.layout().addWidget(self.slider)

        # UNDO key shortcut
        self.action_undo = QtGui.QAction('undo', self)
        self.action_undo.triggered.connect(self.undo)
        self.action_undo.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Z))
        self.addAction(self.action_undo)

        self.undo_button = QtGui.QPushButton("Undo \n (key_Z)")
        self.undo_button.clicked.connect(self.undo)
        widget.layout().addWidget(self.undo_button)

        # DRAW button and key shortcut
        self.action_paint_polygon = QtGui.QAction('paint_polygon', self)
        self.action_paint_polygon.triggered.connect(self.paint_polygon)
        self.action_paint_polygon.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_D))
        self.addAction(self.action_paint_polygon)

        self.poly_button = QtGui.QPushButton("Draw polygons \n (key_D)")
        self.poly_button.clicked.connect(self.paint_polygon)
        widget.layout().addWidget(self.poly_button)

        # CLEAR button and key shortcut
        self.action_clear = QtGui.QAction('clear', self)
        self.action_clear.triggered.connect(self.reset)
        self.action_clear.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_C))
        self.addAction(self.action_clear)

        self.clear_button = QtGui.QPushButton("Clear paint area \n (key_C)")
        self.clear_button.clicked.connect(self.reset)
        widget.layout().addWidget(self.clear_button)

        self.popup_button = QtGui.QPushButton("Done!")
        self.popup_button.clicked.connect(self.popup)
        widget.layout().addWidget(self.popup_button)

        self.set_label_text()

        polymode_button.toggle()

        # complete the gui
        self.layout().addWidget(widget)
        self.layout().addWidget(self.view)


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
