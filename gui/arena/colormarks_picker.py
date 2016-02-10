import sys
import random
import gc
from functools import partial
from gui.arena.my_popup import MyPopup
from gui.arena.my_view import MyView
from utils.video_manager import VideoManager
from PyQt4 import QtGui, QtCore
from core.project.project import Project
from gui.img_controls import utils
import numpy as np

__author__ = 'dita'

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
        self.backup = []

        # image to store all progress
        bg_height, bg_width = 1024, 1024
        bg_size = QtCore.QSize(bg_width, bg_height)
        fmt = QtGui.QImage.Format_ARGB32
        self.paint_image = QtGui.QImage(bg_size, fmt)
        self.paint_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.paint_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.paint_image))

        self.pen_size = 10

        self.pick_id = 0
        self.pick_mask = np.zeros((bg_height, bg_width))

        self.masks = {}

        self.make_gui()
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
        self.set_pen_label_text()

    def set_pen_label_text(self):
        """
        changes the label to show current pen settings
        :return: None
        """

        self.pen_label.setText("Pen size: %s" % self.pen_size)

    def set_color_label_text(self):
        """
        changes the label to show current pen settings
        :return: None
        """

        self.color_label.setText("You are currently editing colormark with id %s" % self.pick_id)

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
        mask = self.pick_mask.copy()
        self.backup.append((img, mask))
        if len(self.backup) > 10:
            self.backup.pop(0)

    def undo(self):
        length = len(self.backup)
        print "Lenght is %s" % length
        if length > 0:
            img, mask = self.backup.pop(length - 1)
            self.refresh_image(img)
            self.pick_mask = mask

    def clear_undo_history(self):
        self.backup = []

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

        if self.color == "Blue":
            paint = QtGui.qRgba(0, 0, 255, 100)
        else:
            # TODO: FIX THIS! This isn't erasing enything.. Seems as if the image colors were unchangeable
            paint = QtGui.qRgba(0, 0, 0, 0)

        # paint the area around the point position
        fromx = point.x() - self.pen_size / 2
        tox = point.x() + self.pen_size / 2
        fromy = point.y() - self.pen_size / 2
        toy = point.y() + self.pen_size / 2

        bg_height, bg_width = self.background.shape[:2]
        for i in range(fromx, tox):
            for j in range(fromy, toy):
                if 0 <= i < bg_width and 0 <= j < bg_height and self.pick_mask[i][j] == 0:
                    c = self.paint_image.pixel(i, j)
                    tmp = QtGui.QColor(c)
                    #print "%s %s %s %s" % (tmp.red(), tmp.green(), tmp.blue(), tmp.alpha())
                    self.paint_image.setPixel(i, j, paint)


        self.pick_mask[fromx: tox, fromy: toy] = 1

        # set new image and pixmap
        self.refresh_image(self.paint_image)

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

    def save_edits(self):
        if len(np.nonzero(self.pick_mask)[0]) == 0:
            return

        r, g, b = self.get_avg_color(self.frame, self.pick_mask)

        frame = self.masks.get(self.pick_id, [0, self.frame])[1]

        self.masks[self.pick_id] = (np.copy(self.pick_mask), frame)
        self.color_grid.modify_color(self.pick_id, r, g, b)

    def get_avg_color(self, frame, mask):
        img = self.vid_manager.seek_frame(frame)
        sumr, sumg, sumb, count = 0, 0, 0, 0
        nzero = np.nonzero(mask)
        for i, j in zip(nzero[0], nzero[1]):
            difb, difg, difr = img[j][i]
            sumr += difr
            sumg += difg
            sumb += difb
            count += 1
        if count == 0:
            return 255, 255, 255
        else:
            return sumr/count+0.0, sumg/count+0.0, sumb/count+0.0

    def switch_color(self):
        text = self.sender().text()
        # make sure the other button never stays pushed
        for button in self.color_buttons:
            if button.text() != text:
                button.setChecked(False)
            else:
                button.setChecked(True)
        self.color = text

    def new_color(self):
        # something was changed
        self.save_edits()

        # prepare for a next change
        self.set_color_label_text()
        self.clear_paint_image()
        self.pick_mask.fill(0)
        self.clear_undo_history()

        self.pick_id = self.color_grid.add_color(255, 255, 255)
        self.masks[self.pick_id] = (np.copy(self.pick_mask), self.frame)

        self.set_color_label_text()

    def move_mask(self, source, target):
        self.masks[target] = self.masks[source]
        self.masks.pop(source)

    def delete_color(self):
        self.color_grid.delete_color(self.pick_id)
        self.masks.pop(self.pick_id)
        for key in sorted(self.masks.iterkeys()):
            if key > self.pick_id:
                self.move_mask(key, key-1)
        self.pick_id = 0
        self.show_mask(self.pick_id, False)
        gc.collect()

    def show_mask(self, mask_id, save=True):
        # something was changed
        if save:
            self.save_edits()

        # prepare for a next change
        self.clear_paint_image()
        self.pick_mask.fill(0)
        self.clear_undo_history()

        self.pick_id = mask_id
        data = self.masks.get(mask_id, [np.zeros((1024, 1024)), self.frame])
        print "Showing mask..."
        print data[0]
        self.frame = data[1]
        self.draw_frame()

        fmt = QtGui.QImage.Format_ARGB32
        bg_size = QtCore.QSize(self.background.shape[0], self.background.shape[1])
        image = QtGui.QImage(bg_size, fmt)
        image.fill(QtGui.qRgba(0, 0, 0, 0))

        mask = data[0]
        print "Showing mask..."
        print mask
        value = QtGui.qRgba(0, 0, 255, 100)
        nzero = np.nonzero(mask)
        for i, j in zip(nzero[0], nzero[1]):
            image.setPixel(i, j, value)

        self.pick_mask = np.copy(mask)
        self.refresh_image(image)
        self.set_color_label_text()

    def random_frame(self):
        # vid_manager's random frame can't be used, because it doesn't return frame id which colormarks_picker uses
        if not self.is_mask_empty(self.pick_mask):
            self.new_color()
        self.frame = random.randint(0, self.vid_manager.total_frame_count())
        self.draw_frame()

    def next_frame(self):
        if not self.is_mask_empty(self.pick_mask):
            self.new_color()
        self.frame += 1
        self.draw_frame()

    def prev_frame(self):
        if not self.is_mask_empty(self.pick_mask):
            self.new_color()
        self.frame -= 1
        self.draw_frame()

    def draw_frame(self):
        print "Going to frame %s" % self.frame
        if self.frame <= 0:
            self.prev_frame_button.setEnabled(False)
        else:
            self.prev_frame_button.setEnabled(True)

        if self.frame >= self.vid_manager.total_frame_count() - 1:
            self.next_frame_button.setEnabled(False)
        else:
            self.next_frame_button.setEnabled(True)

        self.background = self.vid_manager.seek_frame(self.frame)

        # remove previous image (scene gets cluttered with unused pixmaps and is slow)
        if self.old_pixmap is not None:
            self.scene.removeItem(self.old_pixmap)
        self.old_pixmap = self.scene.addPixmap(utils.cvimg2qtpixmap(self.background))
        self.view.update_scale()

        # adjust mask frame if frame is changed and nothing was drawn in the mask yet
        data = self.masks.get(self.pick_id)
        if data is not None and self.is_mask_empty(data[0]) and self.is_mask_empty(self.pick_mask) and data[1] != self.frame:
            self.masks[self.pick_id] = (data[0], self.frame)

    def is_mask_empty(self, mask):
        nzero = np.nonzero(mask)
        return nzero[0].size == 0

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
        self.left_panel = QtGui.QWidget()
        self.left_panel.setLayout(QtGui.QVBoxLayout())
        self.left_panel.layout().setAlignment(QtCore.Qt.AlignTop)
        # set left panel widget width to 300px
        self.left_panel.setMaximumWidth(300)
        self.left_panel.setMinimumWidth(300)

        self.pen_label = QtGui.QLabel()
        self.pen_label.setWordWrap(True)
        self.pen_label.setText("")
        self.left_panel.layout().addWidget(self.pen_label)

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
        self.left_panel.layout().addWidget(self.slider)

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

        eraser_button = QtGui.QPushButton("Eraser")
        eraser_button.setCheckable(True)
        eraser_button.clicked.connect(self.switch_color)
        color_widget.layout().addWidget(eraser_button)
        self.color_buttons.append(eraser_button)
        self.left_panel.layout().addWidget(color_widget)

        # UNDO key shortcut
        self.action_undo = QtGui.QAction('undo', self)
        self.action_undo.triggered.connect(self.undo)
        self.action_undo.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Z))
        self.addAction(self.action_undo)

        self.undo_button = QtGui.QPushButton("Undo \n (key_Z)")
        self.undo_button.clicked.connect(self.undo)
        self.left_panel.layout().addWidget(self.undo_button)

        self.popup_button = QtGui.QPushButton("Done!")
        self.popup_button.clicked.connect(self.popup)
        self.left_panel.layout().addWidget(self.popup_button)

        self.next_frame_button = QtGui.QPushButton("Next frame!")
        self.next_frame_button.clicked.connect(self.next_frame)
        self.left_panel.layout().addWidget(self.next_frame_button)

        self.prev_frame_button = QtGui.QPushButton("Previous frame!")
        self.prev_frame_button.clicked.connect(self.prev_frame)
        self.left_panel.layout().addWidget(self.prev_frame_button)

        self.random_frame_button = QtGui.QPushButton("Random frame!")
        self.random_frame_button.clicked.connect(self.random_frame)
        self.left_panel.layout().addWidget(self.random_frame_button)

        self.new_color_button = QtGui.QPushButton("New color")
        self.new_color_button.clicked.connect(self.new_color)
        self.left_panel.layout().addWidget(self.new_color_button)

        self.color_label = QtGui.QLabel()
        self.color_label.setWordWrap(True)
        self.color_label.setText("")
        self.left_panel.layout().addWidget(self.color_label)

        self.color_grid = ColorGridWidget(update_callback_picked=self.show_mask)
        self.left_panel.layout().addWidget(self.color_grid)

        self.delete_color_button = QtGui.QPushButton("Delete current color")
        self.delete_color_button.clicked.connect(self.delete_color)
        self.left_panel.layout().addWidget(self.delete_color_button)

        self.set_pen_label_text()
        self.set_color_label_text()

        # complete the gui
        self.layout().addWidget(self.left_panel)
        self.layout().addWidget(self.view)

        self.next_frame()
        self.update()


class ColorGridWidget(QtGui.QWidget):
    def __init__(self, max_cols=5, update_callback_picked=None, update_callback_deleted=None):
        super(ColorGridWidget, self).__init__(None)
        self.max_cols = max_cols
        self.colors = {}
        self.buttons = {}
        self.last_index = 0

        self.update_callback_picked = update_callback_picked
        self.update_callback_deleted = update_callback_deleted

        self.col = 0
        self.posx = 0
        self.posy = 0

        self.grid = QtGui.QGridLayout()
        self.setLayout(self.grid)

    def clicked(self, button):
        print_dict_keys(self.buttons)
        print_dict_keys(self.colors)
        button_id = int(button.text())
        print "Selected %s" % button_id
        color = self.colors.get(button_id, None)
        if color is None:
            return
        self.repaint_all_buttons()
        print_dict_keys(self.buttons)
        print_dict_keys(self.colors)

        self.update_callback_picked(button_id)

    def add_color(self, r, g, b):
        print ("Adding color")
        self.colors[self.last_index] = (r,g,b)
        button = QtGui.QPushButton()
        button.setText("%s" % self.last_index)
        button.clicked.connect(partial(self.clicked, button))
        button.setStyleSheet('QPushButton {background-color: #%02x%02x%02x; color: #%02x%02x%02x;}' % (
        r, g, b, 255 - r, 255 - g, 255 - b))
        self.buttons[self.last_index] = button
        self.grid.addWidget(button, self.posx, self.posy)
        self.last_index += 1
        self.col += 1
        if self.col >= self.max_cols:
            self.col = 0
            self.posy = 0
            self.posx += 1
        else:
            self.posy += 1

        return self.last_index-1

    def modify_color(self, idd, r, g, b):
        if idd not in self.colors:
            self.add_color(r, g, b)
        else:
            self.colors[idd] = (r,g,b)
            self.buttons[idd].setStyleSheet('QPushButton {background-color: #%02x%02x%02x; color: #%02x%02x%02x;}' % (
            r, g, b, 255 - r, 255 - g, 255 - b))

    def delete_color(self, idd):
        print "Deleting id: %s" % idd

        if idd == self.last_index-1:
            self.colors[idd] = None
            del (self.colors[idd])
            btn = self.buttons[idd]
            self.grid.removeWidget(btn)
            btn.deleteLater()
            del (self.buttons[idd])
            self.last_index -= 1
            self.repaint_all_buttons()
            print_dict_keys(self.buttons)
            print_dict_keys(self.colors)
            return

        del (self.colors[idd])
        btn = self.buttons[idd]
        self.grid.removeWidget(btn)
        btn.deleteLater()
        del (self.buttons[idd])
        self.last_index -= 1


        for c_id in sorted(self.colors.keys()):
            if c_id > idd:
                self.colors[c_id-1] = self.colors[c_id]
                self.colors[c_id] = None
                self.buttons[c_id-1] = self.buttons[c_id]
                self.buttons[c_id] = None

        del (self.buttons[self.last_index])
        del (self.colors[self.last_index])

        self.repaint_all_buttons()

    def repaint_all_buttons(self):
        self.col = 0
        self.posx = 0
        self.posy = 0

        for id in sorted(self.buttons.keys()):
            print "Repainting %s" % id
            self.grid.removeWidget(self.buttons[id])
            self.buttons[id].setText("%s" % id)
            self.grid.addWidget(self.buttons[id], self.posx, self.posy)
            self.col += 1
            if self.col >= self.max_cols:
                self.col = 0
                self.posy = 0
                self.posx += 1
            else:
                self.posy += 1

def print_dict_keys(dic):
    output = ""
    for key in sorted(dic.keys()):
        output += (str(key) + " ")
    print output


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
