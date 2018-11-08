from __future__ import print_function
from __future__ import absolute_import
import sys
from .my_view import MyView
from .my_scene import MyScene
from PyQt4 import QtGui, QtCore
import numpy as np
import cv2
# warning: qimage2ndarray could get confused with multiple Qt versions installed (PyQt, PySide, ...) and end with
# AttributeError: 'module' object has no attribute 'QImage'
from qimage2ndarray import array2qimage


__author__ = 'dita'


class Painter(QtGui.QWidget):
    """ Painter widget that can be used in all painting applications"""

    def __init__(self, image, pen_size=10, undo_len=10, debug=False, update_callback=None, paint_name="PINK", paint_r=255, paint_g=0, paint_b=238, paint_a=100):
        """
        :param image: CV2 image (bgr format expected) """

        super(Painter, self).__init__()

        self.DEBUG = debug
        self.update_callback = update_callback

        self.background = None
        self.paint_pixmap = None
        self.w = None
        self.h = None

        self.disable_drawing = False

        # WIDGET SETUP
        self.view = MyView(update_callback_move=self.mouse_moving, update_callback_press=self.mouse_press_event)
        self.scene = MyScene(update_callback_release=self.mouse_released)
        self.view.setScene(self.scene)

        # store last 10 QImages to support the "undo()" function
        self.backup = []
        self.undo_len = undo_len

        # show background in one pixmap
        if image is not None:
            self.set_image(image)

        self.overlay_pixmap = None
        self.overlay2_pixmap = None

        # PAINT SETUP
        self.pen_size = pen_size / 2
        self.eraser = 1  # 1 for painting (eraser off), 0 for erasing

        self.colors = {}  # dictionary: [name] : (mask, color, pixmap)
        self.add_color(paint_name, paint_r, paint_g, paint_b, paint_a) # add the first - default color
        self.set_pen_color(paint_name)

        # create the main view and left panel with buttons
        self.make_gui()

    def set_size(self, w, h):
        # create empty overlay - this can be used to set masks from outside the painter
        self.w = w
        self.h = h
        bg_size = QtCore.QSize(self.w, self.h)
        fmt = QtGui.QImage.Format_ARGB32
        overlay_image = QtGui.QImage(bg_size, fmt)
        overlay_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.overlay_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(overlay_image))
        self.overlay2_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(overlay_image))

        for color in self.colors.itervalues():
            if color[0] is None:
                color[0] = np.zeros((self.h, self.w))

    def set_image(self, image):
        self.background = array2qimage(bgr2rgb(image))
        if self.paint_pixmap:
            self.scene.removeItem(self.paint_pixmap)
        self.paint_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.background))

        if self.w != self.background.width() or self.h != self.background.height():
            self.set_size(self.background.width(), self.background.height())

    def set_image_visible(self, visibility):
        """ Toggles background visibility
        :param visibility: new visibility (True/False)
        :return: None
        """
        self.paint_pixmap.setVisible(visibility)

    def set_overlay_visible(self, visibility):
        """ Toggles overlay visibility
        :param visibility: new visibility (True/False)
        :return: None
        """
        self.overlay_pixmap.setVisible(visibility)

    def set_overlay2_visible(self, visibility):
        """ Toggles overlay visibility
        :param visibility: new visibility (True/False)
        :return: None
        """
        self.overlay2_pixmap.setVisible(visibility)

    def set_masks_visible(self, visibility):
        """ Toggles masks visibility
        :param visibility: new visibility (True/False)
        :return: None
        """
        for color, data in self.colors.iteritems():
            if data[2]:
                data[2].setVisible(visibility)

    def set_overlay(self, img):
        """ Deletes the old overlay image and pixmap and replaces them with a new image. The image should have an alpha
        channel, otherwise it can hide other scene contents.
        :param img: a new image to use, None to delete overlay completely.
        :return: None
        """
        if self.overlay_pixmap and self.overlay_pixmap in self.scene.items():
            self.scene.removeItem(self.overlay_pixmap)
        if img:
            self.overlay_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(img))
            self.overlay_pixmap.setZValue(9)

    def set_overlay2(self, img):
        """ Deletes the old overlay image and pixmap and replaces them with a new image. The image should have an alpha
        channel, otherwise it can hide other scene contents.
        :param img: a new image to use, None to delete overlay completely.
        :return: None
        """
        if self.overlay2_pixmap and self.overlay2_pixmap in self.scene.items():
            self.scene.removeItem(self.overlay2_pixmap)
        if img:
            self.overlay2_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(img))
            self.overlay2_pixmap.setZValue(9)

    def draw_mask(self, name):
        """ Paints the mask with given color on the image"""

        qimg = mask2qimage(self.colors[name][0], self.colors[name][1])

        # add pixmap to scene and move it to the foreground
        # delete old pixmap
        self.scene.removeItem(self.colors[name][2])
        self.colors[name][2] = self.scene.addPixmap(QtGui.QPixmap.fromImage(qimg))
        self.colors[name][2].setZValue(20)

    def reset_masks(self):
        for name, data in self.colors.iteritems():

            # reset mask
            mask = np.zeros((self.h, self.w))
            color = data[1]
            # remove old pixmap
            self.scene.removeItem(data[2])
            # prepare new qimage from empty mask
            qimg = mask2qimage(mask, data[1])
            # add new pixmap
            pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(qimg))

            self.colors[name] = [mask, color, pixmap]

    def add_color(self, name, r, g, b, a=100):
        """ Adds a new color option to painter.
        :param name: The name and unique ID of new color. Using duplicate names will overwrite previous data and can
        have unexpected behavior.
        :param r, g, b: R, G, B color values (respectively)
        :param a: Alpha channel, 100 by default
        :return: None
        """
        
        # prepare new mask
        if self.h is None or self.w is None:
            mask = None
        else:
            mask = np.zeros((self.h, self.w))
        # save color data
        color = (r, g, b, a)
        # fill the dictionary (pixmap will be set once the mask is not empty)
        self.colors[name] = [mask, color, None]

    def add_color_(self, name, rgba):
        self.add_color(name, rgba[0], rgba[1], rgba[2], rgba[3])

    def get_result(self):
        """ Retrieves painter data
        :return: Dictionary with color names as keys [color_name] : (ndarray mask, (r, g, b, a), pixmap)
        """
        return self.colors

    def set_pen_size(self, value):
        """ Change pen size
        :param value: new pen size
        :return: None
        """
        # change pen size
        self.pen_size = value / 2

    def set_pen_color(self, name):
        """ Change pen size
        :param value: new pen color (name)
        :return: None
        """
        if name:
            self.color_name = name
            self.eraser = 1
        else:
            self.eraser = 0

    def mouse_press_event(self, event):
        point = self.view.mapToScene(event.pos())
        if self.is_in_scene(point) and not self.disable_drawing:
            self.save()
            self.draw(point)

    def mouse_moving(self, event):
        # while the mouse is moving, paint it's position
        point = self.view.mapToScene(event.pos())
        if self.is_in_scene(point) and not self.disable_drawing:
            # no save here!
            self.draw(point)

    def mouse_released(self, event):
        if self.update_callback:
            self.update_callback()

    def save(self):
        """
        Saves current image state (to use with "undo()" later)
        :return:
        """
        # save the image and mask
        mask = self.colors[self.color_name][0].copy()
        name = self.color_name
        self.backup.append((name, mask))

        # remove the oldest save if backup size got larger than self.undo_size
        if len(self.backup) > self.undo_len:
            self.backup.pop(0)

    def undo(self):
        # get number of elements in backup
        length = len(self.backup)
        if self.DEBUG:
            print("Length is %s" % length)

        # proceed to undo if there is something to undo
        if length > 0:
            name, mask = self.backup.pop(length - 1)

            # set mask to previous state
            self.colors[name][0] = mask

            # repaint the color
            self.draw_mask(name)

            # also inform parent widget
            if self.update_callback:
                self.update_callback()

    def draw(self, point):
        """ Draw a point with a pen
        :param point: point to be drawn
        :return: None
        """
        # change float to int (QPointF -> QPoint)
        if type(point) == QtCore.QPointF:
            point = point.toPoint()

        # paint the area around the point position
        fromx = point.x() - self.pen_size
        tox = point.x() + self.pen_size
        fromy = point.y() - self.pen_size
        toy = point.y() + self.pen_size

        # use color paint
        self.colors[self.color_name][0][fromy: toy, fromx: tox] = self.eraser

        # set new image and pixmap
        self.draw_mask(self.color_name)

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
                print("Out of bounds [%s, %s]" % (map_pos.x(), map_pos.y()))
            return False

    def is_in_scene(self, point):
        """
        checks if the point is inside the scene
        :param point: Qpoint or QPointF
        :return: True or False
        """
        if self.scene.itemsBoundingRect().contains(point) and point.x() <= self.w and point.y() <= self.h:
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

        # UNDO key shortcut
        self.action_undo = QtGui.QAction('undo', self)
        self.action_undo.triggered.connect(self.undo)
        self.action_undo.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Z))
        self.addAction(self.action_undo)


        # complete the gui
        self.layout().addWidget(self.view)


def mask2pixmap(mask, color):
    r = mask * color[0]
    g = mask * color[1]
    b = mask * color[2]
    a = mask * color[3]
    image = np.dstack((r, g, b, a))
    return rgba2qimage(image)

def bgr2rgb(image):
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    return np.dstack((r, g, b))

def numpy2qimage(image):
    if type(image) == QtGui.QImage:
        return image
    height, width, channels = image.shape
    bytesPerLine = channels * width
    return QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

def rgba2qimage(image):
    if type(image) == QtGui.QImage:
        return image
    height, width, channels = image.shape
    bytesPerLine = channels * width
    return QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_ARGB32)


def mask2qimage(mask, color):
    # create a RGBA image from mask and color data
    transposed = mask[..., None]*color
    # convert to Qt compatible qimage
    qimg = array2qimage(transposed)
    return qimg


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    image = cv2.imread('/home/dita/vlcsnap-2016-08-16-17h28m57s150.png')
    image = numpy2qimage(image)

    ex = Painter(image)
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()
