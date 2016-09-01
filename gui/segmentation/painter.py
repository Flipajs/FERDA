import sys
from my_view import MyView
from PyQt4 import QtGui, QtCore, Qt
import numpy as np
import cv2
from qimage2ndarray import array2qimage

__author__ = 'dita'


class Painter(QtGui.QWidget):
    """ Painter widget that can be used in all painting applications"""

    def __init__(self, image, pen_size=10, undo_len=10, debug=False, update_callback=None, paint_name = "PINK", paint_r=255, paint_g=0, paint_b=238, paint_a=100):
        """ """

        super(Painter, self).__init__()

        self.DEBUG = debug
        self.update_callback = update_callback

        # WIDGET SETUP
        self.view = MyView(update_callback_move=self.mouse_moving, update_callback_press=self.mouse_press_event)
        self.scene = QtGui.QGraphicsScene()
        self.view.setScene(self.scene)
        self.view.setMouseTracking(True)

        # store last 10 QImages to support the "undo()" function
        self.backup = []
        self.undo_len = undo_len

        self.background = numpy2qimage(image)
        self.scene.addPixmap(QtGui.QPixmap.fromImage(self.background))

        # create empty image and pixmap to view painting
        bg_size = QtCore.QSize(self.background.width(), self.background.height())
        fmt = QtGui.QImage.Format_ARGB32
        self.paint_image = QtGui.QImage(bg_size, fmt)
        self.paint_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.paint_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.paint_image))

        self.overlay_image = QtGui.QImage(bg_size, fmt)
        self.overlay_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.overlay_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.overlay_image))
      
        self.bg_width = self.paint_image.width()
        self.bg_height = self.paint_image.height()

        # PAINT SETUP
        # current color ("Color" or "Eraser"), purple by default
        self.colors = {}
        self.eraser = False
        self.add_color(paint_name, paint_r, paint_g, paint_b, paint_a)
        self.pen_size = pen_size
        self.set_pen_color(paint_name)

        # create the main view and left panel with buttons
        self.make_gui()

    def set_image(self, img):
        """ Deletes the old image and pixmap and replaces them with a new image	
        :param img: a new image to use
        :return: None
        """
        self.paint_image = img
        self.scene.removeItem(self.paint_pixmap)
        self.paint_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.paint_image))

    def set_overlay(self, img):
        """ Deletes the old image and pixmap and replaces them with a new image
        :param img: a new image to use
        :return: None
        """
        self.overlay_image = img
        self.scene.removeItem(self.overlay_pixmap)
        self.overlay_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(self.overlay_image))
        self.overlay_pixmap.setZValue(9)

    def draw_mask(self, name):
        self.scene.removeItem(self.colors[name][2])
        w, h = self.background.width(), self.background.height()
        r = self.colors[name][0] * self.colors[name][1][0]
        g = self.colors[name][0] * self.colors[name][1][1]
        b = self.colors[name][0] * self.colors[name][1][2]
        a = self.colors[name][0] * self.colors[name][1][3]
        r.shape = ((w*h))
        g.shape = ((w*h))
        b.shape = ((w*h))
        a.shape = ((w*h))
        rgba = np.dstack((r, g, b, a))
        rgba.shape = ((w, h, 4))
        qimg = array2qimage(rgba)
        self.colors[name][2] = self.scene.addPixmap(QtGui.QPixmap.fromImage(qimg))
        self.colors[name][2].setZValue(10)

    def add_color(self, name, r, g, b, a=100):
        mask = np.zeros((self.bg_width, self.bg_height))
        color = (r, g, b, a)
        self.colors[name] = [mask, color, None]
        self.colors[name][2] = None

    def get_result(self):
        return self.colors

    def set_pen_size(self, value):
        """ Change pen size
        :param value: new pen size
        :return: None
        """
        # change pen size
        self.pen_size = value

    def set_pen_color(self, name):
        """ Change pen size
        :param value: new pen color (name)
        :return: None
        """
        if name:
            self.color_name = name
            self.eraser = False
        else:
            self.eraser = True

    def mouse_press_event(self, event):
        point = self.view.mapToScene(event.pos())
        if self.is_in_scene(point):
            self.save()
            self.draw(point)
        if self.update_callback:
            self.update_callback()

    def mouse_moving(self, event):
        # while the mouse is moving, paint it's position
        point = self.view.mapToScene(event.pos())
        if self.is_in_scene(point):
            # no save here!
            self.draw(point)

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
            print "Length is %s" % length
        # proceed to undo if there is something to undo
        if length > 0:
            img, mask, name = self.backup.pop(length - 1)
            self.refresh_image(img)
            self.colors[name][0] = mask
            # TODO: this doesn't work, since it can overwrite a mask that is different from the origin
            if self.update_callback:
                self.update_callback()

    def clear_undo_history(self):
        self.backup = []

    def clear_paint_image(self):
        # remove all drawn lines
        self.paint_image.fill(QtGui.qRgba(0, 0, 0, 0))
        self.refresh_image(self.paint_image)

    def draw(self, point):
        """ Draw a point with a pen
        :param point: point to be drawn
        :return: None
        """
        # change float to int (QPointF -> QPoint)
        if type(point) == QtCore.QPointF:
            point = point.toPoint()

        # paint the area around the point position
        fromx = point.x() - self.pen_size / 2
        tox = point.x() + self.pen_size / 2
        fromy = point.y() - self.pen_size / 2
        toy = point.y() + self.pen_size / 2

        # use color paint
        if self.eraser:
            self.colors[self.color_name][0][fromy: toy, fromx: tox] = 0
        # use eraser
        else:
            self.colors[self.color_name][0][fromy: toy, fromx: tox] = 1

        # set new image and pixmap
        self.draw_mask(self.color_name)
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
        if self.scene.itemsBoundingRect().contains(point) and point.x() <= self.bg_width and point.y() <= self.bg_height:
            return True
        else:
            return False

    def is_mask_empty(self, mask):
        nzero = np.nonzero(mask)
        return nzero[0].size == 0

    def refresh_image(self, img):
        """
        deletes the old image and pixmap and replaces them with the image given
        :param img: the new image to use
        :return: None
        """
        self.paint_image = img
        self.scene.removeItem(self.paint_pixmap)
        self.paint_pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(img))

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
    print color
    image = np.dstack((r, g, b, a))
    return rgba2qimage(image)
    

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

def mask2qimage(image):
    if type(image) == QtGui.QImage:
        return image
    height, width = image.shape
    bytesPerLine = width
    return QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)


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
