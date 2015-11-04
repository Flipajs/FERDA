from gui.arena.my_popup   import MyPopup
from gui.arena.my_view    import MyView
from utils.video_manager import VideoManager

__author__ = 'dita'

from PyQt4 import QtGui, QtCore, Qt
import cv2
import sys
import math
from core.project.project import Project
from gui.img_controls     import utils
import numpy as np

"""
widget
in:
 video
funkce:
 oznacit na frame colormark
 musi byt unikatni
 pro kazdou barvu maska
 kliknout a prejit na frame
 zoom
out
 pole id barvy, maska, obrazek src

branch
 rgb histogram
 irg hist demo
 get color samples:
"""

class ColormarksPicker(QtGui.QWidget):
    DEBUG = True

    def __init__(self, video_path):

        super(ColormarksPicker, self).__init__()

        self.vid_manager = VideoManager(video_path)

        self.view = MyView(update_callback_move=self.mouse_moving, update_callback_press=self.mouse_press_event)
        self.scene = QtGui.QGraphicsScene()

        self.view.setScene(self.scene)

        # background image
        self.frame = -1
        self.next_frame()

        self.view.setMouseTracking(True)

        self.color = "Blue"

        # store last 10 QImages to support the "undo" function
        # undo button can only be pushed in paint mode
        self.backup = []

        self.pen_size = 30

        # image to store all progress
        bg_height, bg_width = self.background.shape[:2]
        bg_size = QtCore.QSize(bg_width, bg_height)
        fmt = QtGui.QImage.Format_ARGB32
        self.paint_image = QtGui.QImage(bg_size, fmt)
        self.paint_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.paint_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.paint_image))

        self.save()

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

        self.pen_label.setText("Pen size: %s" % self.pen_size)

    def reset(self):
        """
        clear everything and start over
        :return: None
        """
        self.clear_paint_image()
        self.point_items = []

    def mouse_press_event(self, event):
        point = self.view.mapToScene(event.pos())
        if self.is_in_scene(point):
            self.save()
            self.draw(point)

    def mouse_moving(self, event):
        # while the mouse is moving, paint it's position
        point = self.view.mapToScene(event.pos())
        if self.is_in_scene(point):
            self.draw(point)

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
        lenght = len(self.backup)
        if lenght > 0:
            img = self.backup.pop(lenght-1)
            self.refresh_image(img)

    def refresh_image(self, img):
        self.paint_image = img
        self.scene.removeItem(self.paint_pixmap)
        self.paint_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(img))

    def clear_paint_image(self):
        # remove all drawn lines
        self.paint_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.refresh_image(self.paint_image)

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
                if i >= 0 and i < bg_width and j >= 0 and j < bg_height:
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

    def next_frame(self):
        self.frame += 1
        self.background = self.vid_manager.seek_frame(self.frame)
        self.scene.addPixmap(utils.cvimg2qtpixmap(self.background))

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
        label.setText("")
        widget.layout().addWidget(label)

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

        # PEN SIZE slider
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.setGeometry(30, 40, 50, 30)
        self.slider.setRange(10, 50)
        self.slider.setTickInterval(5)
        self.slider.setValue(30)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider.valueChanged[int].connect(self.change_value)
        self.slider.setVisible(True)
        widget.layout().addWidget(self.slider)

        # UNDO key shortcut
        self.action_undo = QtGui.QAction('undo', self)
        self.action_undo.triggered.connect(self.undo)
        self.action_undo.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Z))
        self.addAction(self.action_undo)

        self.undo_button = QtGui.QPushButton("Undo \n (key_Z)")
        self.undo_button.clicked.connect(self.undo)
        widget.layout().addWidget(self.undo_button)

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

        self.next_frame_button = QtGui.QPushButton("Next frame!")
        self.next_frame_button.clicked.connect(self.next_frame)
        widget.layout().addWidget(self.next_frame_button)

        self.set_label_text()

        # complete the gui
        self.layout().addWidget(widget)
        self.layout().addWidget(self.view)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    p = Project()
    p.load("/home/dita/PycharmProjects/FERDA projects/testc5/c5.fproj")

    ex = ColormarksPicker(p.video_paths[0])
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()
