import random
from gui.arena.my_popup   import MyPopup
from gui.arena.my_view    import MyView
from utils.video_manager import VideoManager

__author__ = 'dita'

from PyQt4 import QtGui, QtCore
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

        # TODO: Perhaps use one VideoManager for the whole project?
        self.vid_manager = VideoManager(video_path)

        self.view = MyView(update_callback_move=self.mouse_moving, update_callback_press=self.mouse_press_event)
        self.scene = QtGui.QGraphicsScene()

        self.view.setScene(self.scene)

        # background image
        self.frame = -1
        self.old_pixmap = None
        self.background = None

        self.view.setMouseTracking(True)

        self.color = "Blue"

        # store last 10 QImages to support the "undo" function
        # undo button can only be pushed in paint mode
        self.backup = []

        # image to store all progress
        bg_height, bg_width = 1024, 1024
        bg_size = QtCore.QSize(bg_width, bg_height)
        fmt = QtGui.QImage.Format_ARGB32
        self.paint_image = QtGui.QImage(bg_size, fmt)
        self.paint_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.paint_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.paint_image))

        self.pen_size = 10
        self.make_gui()

        self.pick_id = 0
        self.pick_mask = np.zeros((bg_height, bg_width))

        self.avgcount = 0
        self.avgr = 0
        self.avgg = 0
        self.avgb = 0

        self.masks = []

        self.save()

        # create the main view and left panel with buttons

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
        length = len(self.backup)
        if length > 0:
            img = self.backup.pop(length-1)
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

        value = QtGui.qRgba(0, 0, 255, 100)

        # paint the area around the point position
        fromx = point.x() - self.pen_size/2
        tox = point.x() + self.pen_size/2
        fromy = point.y() - self.pen_size/2
        toy = point.y() + self.pen_size/2

        bg_height, bg_width = self.background.shape[:2]
        for i in range(fromx, tox):
            for j in range(fromy, toy):
                if 0 <= i < bg_width and 0 <= j < bg_height and self.pick_mask[i][j] == 0:
                    self.paint_image.setPixel(i, j, value)
                    difb, difg, difr = self.background[i][j]
                    self.avgr += difr
                    self.avgg += difg
                    self.avgb += difb
                    self.avgcount += 1
        self.pick_mask[fromx : tox][fromy : toy].fill(1)

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

    def next_color(self):
        r = self.avgr/self.avgcount+0.0
        g = self.avgg/self.avgcount+0.0
        b = self.avgb/self.avgcount+0.0
        print ("Color is %s,%s,%s" % (r,g,b))

        self.w = ColorPopup(QtGui.QColor(r, g, b))
        self.w.setGeometry(QtCore.QRect(100, 100, 400, 200))
        self.w.show()

        # save current color mask and data
        self.masks.append((self.pick_id, self.pick_mask, self.frame))

        # prepare for a next one
        self.pick_mask.fill(0)
        self.pick_id += 1
        self.avgcount = 0
        self.avgr, self.avgg, self.avgb = 0, 0, 0

    def random_frame(self):
        # vid_manager's random frame can't be used, because it doesn't return frame id which colormarks_picker uses
        self.frame = random.randint(0, self.vid_manager.total_frame_count())
        self.draw_frame()

    def next_frame(self):
        self.frame += 1
        self.draw_frame()

    def prev_frame(self):
        self.frame -= 1
        self.draw_frame()

    def draw_frame(self):
        print "Going to frame %s" % self.frame
        if self.frame <= 0:
            self.prev_frame_button.setEnabled(False)
        else:
            self.prev_frame_button.setEnabled(True)

        if self.frame >= self.vid_manager.total_frame_count()-1:
            self.next_frame_button.setEnabled(False)
        else:
            self.next_frame_button.setEnabled(True)

        self.background = self.vid_manager.seek_frame(self.frame)

        # remove previous image (scene gets cluttered with unused pixmaps and is slow)
        if self.old_pixmap is not None:
            self.scene.removeItem(self.old_pixmap)
        self.old_pixmap = self.scene.addPixmap(utils.cvimg2qtpixmap(self.background))
        self.view.update_scale()

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
        left_panel = QtGui.QWidget()
        left_panel.setLayout(QtGui.QVBoxLayout())
        left_panel.layout().setAlignment(QtCore.Qt.AlignTop)
        # set left panel widget width to 300px
        left_panel.setMaximumWidth(300)
        left_panel.setMinimumWidth(300)

        self.pen_label = QtGui.QLabel()
        self.pen_label.setWordWrap(True)
        self.pen_label.setText("")
        left_panel.layout().addWidget(self.pen_label)

        # PEN SIZE slider
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.setGeometry(30, 40, 50, 30)
        self.slider.setRange(2, 30)
        self.slider.setTickInterval(1)
        self.slider.setValue(self.pen_size)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider.valueChanged[int].connect(self.change_value)
        self.slider.setVisible(True)
        left_panel.layout().addWidget(self.slider)

        # UNDO key shortcut
        self.action_undo = QtGui.QAction('undo', self)
        self.action_undo.triggered.connect(self.undo)
        self.action_undo.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Z))
        self.addAction(self.action_undo)

        self.undo_button = QtGui.QPushButton("Undo \n (key_Z)")
        self.undo_button.clicked.connect(self.undo)
        left_panel.layout().addWidget(self.undo_button)

        # CLEAR button and key shortcut
        self.action_clear = QtGui.QAction('clear', self)
        self.action_clear.triggered.connect(self.reset)
        self.action_clear.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_C))
        self.addAction(self.action_clear)

        self.clear_button = QtGui.QPushButton("Clear paint area \n (key_C)")
        self.clear_button.clicked.connect(self.reset)
        left_panel.layout().addWidget(self.clear_button)

        self.popup_button = QtGui.QPushButton("Done!")
        self.popup_button.clicked.connect(self.popup)
        left_panel.layout().addWidget(self.popup_button)

        self.next_frame_button = QtGui.QPushButton("Next frame!")
        self.next_frame_button.clicked.connect(self.next_frame)
        left_panel.layout().addWidget(self.next_frame_button)

        self.prev_frame_button = QtGui.QPushButton("Previous frame!")
        self.prev_frame_button.clicked.connect(self.prev_frame)
        left_panel.layout().addWidget(self.prev_frame_button)

        self.random_frame_button = QtGui.QPushButton("Random frame!")
        self.random_frame_button.clicked.connect(self.random_frame)
        left_panel.layout().addWidget(self.random_frame_button)

        self.next_color_button = QtGui.QPushButton("Next color")
        self.next_color_button.clicked.connect(self.next_color)
        left_panel.layout().addWidget(self.next_color_button)

        self.set_label_text()

        # complete the gui
        self.layout().addWidget(left_panel)
        self.layout().addWidget(self.view)

        self.next_frame()
        self.update()


class ColorPopup(QtGui.QWidget):
    def __init__(self, color):
        QtGui.QWidget.__init__(self)
        self.color = color

    def paintEvent(self, e):
        dc = QtGui.QPainter(self)

        dc.setBrush(self.color);
        dc.setPen(QtGui.QPen(self.color));
        dc.drawRect(0, 0, 100,100);


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
