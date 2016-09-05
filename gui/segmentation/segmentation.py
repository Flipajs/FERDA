import sys
import random
import gc
from functools import partial
from PyQt4 import QtGui, QtCore
import painter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *

from sklearn.ensemble import RandomForestClassifier

__author__ = 'dita'


class SegmentationPicker(QtGui.QWidget):

    def __init__(self, img, done_callback=None, pen_size=10, undo_len=10, debug=False, paint_r=255, paint_g=0, paint_b=238, paint_a=255):

        super(SegmentationPicker, self).__init__()

        self.DEBUG = debug
        self.done_callback = done_callback

        self.img_path = img
        self.image = cv2.imread(self.img_path)
        self.h, self.w, c = self.image.shape
        self.pen_size = pen_size

        self.color = [paint_r, paint_g, paint_b, paint_a]

        self.cur_color = "PINK"
        self.cur_eraser = False

        self.prepare_images()
        self.make_gui()

    def prepare_images(self):

        # create a blurred image
        blur = 33
        a = 0
        b = 37
        blur_image = cv2.GaussianBlur(self.image, (blur, blur), 0)

        # find edges on the blurred image
        self.edges = cv2.Canny(blur_image, a, b)

        # original img - shifted img
        self.shiftx = get_shift(self.image, shift_x=2, shift_y=0)
        self.shifty = get_shift(self.image, shift_x=0, shift_y=2)

        self.avg = get_avg(self.image)

        # channel difs
        self.bg = np.asarray(self.image[:,:,0], dtype=np.int32) - np.asarray(self.image[:,:,1], dtype=np.int32)
        self.gr = np.asarray(self.image[:,:,1], dtype=np.int32) - np.asarray(self.image[:,:,2], dtype=np.int32)
        self.rb = np.asarray(self.image[:,:,2], dtype=np.int32) - np.asarray(self.image[:,:,0], dtype=np.int32)

    def done(self):
        # get user data from painter
        result = self.view.get_result()
        background = result["PINK"]
        foreground = result["GREEN"]

        # prepare learning data
        # X contains tuples of data for each evaluated unit-pixel (R, G, B, edge?)
        # y contains classifications for all pixels respectively
        X = []
        y = []
 
        # loop all nonzero pixels from foregound (ants) and background and add them to testing data
        nzero = np.nonzero(background[0])
        if len(nzero[0]) == 0:
            self.view.set_overlay(None)
            return
        for i, j in zip(nzero[0], nzero[1]):
            self.get_data(i, j, X, y, 0)

        nzero = np.nonzero(foreground[0])
        if len(nzero[0]) == 0:
            self.view.set_overlay(None)
            return
        for i, j in zip(nzero[0], nzero[1]):
            self.get_data(i, j, X, y, 1)

        # create the classifier
        rfc = RandomForestClassifier("""class_weight='balanced'""")
        rfc.fit(X, y)

        h, w, c = self.image.shape

        data = np.dstack((self.image[:,:,2].reshape((h*w, 1)),
                          self.image[:,:,1].reshape((h*w, 1)),
                          self.image[:,:,0].reshape((h*w, 1)),
                          self.avg.reshape((h*w, 1)),
                          self.shiftx.reshape((h*w, 1)),
                          self.shifty.reshape((h*w, 1)),
                          self.bg.reshape((h*w, 1)),
                          self.gr.reshape((h*w, 1)),
                          self.rb.reshape((h*w, 1))))

        # reshape the image so it contains 4-tuples, each descripting a single pixel
        data.shape = ((h*w, 9))

        # prepare a mask and predict result for data (current image)
        # mask1 = np.zeros((h*w, c))
        mask1 = rfc.predict_proba(data)

        # reshape mask to be a grid, not a list
        mask1 = mask1[:, 1]
        mask1.shape = ((h, w))

        # create a rgba image from mask
        r = np.zeros((h, w), dtype=np.uint8)
        g = np.asarray(mask1 * 255, dtype=np.uint8)
        b = np.zeros((h, w), dtype=np.uint8)
        a = np.full((h, w), 100, dtype=np.uint8)
        rgb = np.dstack((b, g, r, a))

        foo = painter.rgba2qimage(rgb)
        self.view.set_overlay(foo)

    def pink(self):
        self.cur_color = "PINK"
        self.cur_eraser = False
        self.set_color()

    def green(self):
        self.cur_color = "GREEN"
        self.cur_eraser = False
        self.set_color()

    def set_color(self):
        if self.cur_eraser:
            self.view.set_pen_color(None)
            self.color_buttons["eraser"].setChecked(True)
        else:
            self.view.set_pen_color(self.cur_color)
            for color, btn in self.color_buttons.iteritems():
                if color == self.cur_color.lower():
                    btn.setChecked(True) 
                else:
                    btn.setChecked(False)

    def set_eraser(self):
        if self.cur_eraser:
            self.cur_eraser = False
        else:
            self.cur_eraser = True
        self.set_color()

    def toggle_bg(self):
        c = self.toggle_bg_button.isChecked()
        self.toggle_bg_button.setChecked(c)
        self.view.set_image_visible(not c)

    def show_edges(self):
        shift_up = get_shift(self.image, shift_x=-1, shift_y=0)
        shift_down = get_shift(self.image, shift_x=1, shift_y=0)
        shift_left = get_shift(self.image, shift_x=0, shift_y=-1)
        shift_right = get_shift(self.image, shift_x=0, shift_y=1)
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        img_sum = shift_up + shift_down + shift_left + shift_right + img
        avg = img_sum / 5
        
        plt.imshow(avg)
        plt.show()

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

        # drawing area
        self.view = painter.Painter(self.image, pen_size=5, paint_name="PINK", paint_r=255, paint_g=0, paint_b=238, paint_a=255, update_callback=self.done)
        self.view.add_color("GREEN", 0, 255, 0, 255)

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
        self.slider.valueChanged[int].connect(self.view.set_pen_size)
        self.slider.setVisible(True)
        self.left_panel.layout().addWidget(self.slider)

        # color switcher widget
        color_widget = QtGui.QWidget()
        color_widget.setLayout(QtGui.QHBoxLayout())

        self.color_buttons = {}
        pink_button = QtGui.QPushButton("Pink")
        pink_button.setCheckable(True)
        pink_button.setChecked(True)
        pink_button.clicked.connect(self.pink)
        color_widget.layout().addWidget(pink_button)
        self.color_buttons["pink"] = pink_button

        green_button = QtGui.QPushButton("Green")
        green_button.setCheckable(True)
        green_button.clicked.connect(self.green)
        color_widget.layout().addWidget(green_button)
        self.color_buttons["green"] = green_button

        eraser_button = QtGui.QPushButton("Eraser")
        eraser_button.setCheckable(True)
        eraser_button.clicked.connect(self.set_eraser)
        color_widget.layout().addWidget(eraser_button)
        self.left_panel.layout().addWidget(color_widget)
        self.color_buttons["eraser"] = eraser_button

        # UNDO key shortcut
        self.action_undo = QtGui.QAction('undo', self)
        self.action_undo.triggered.connect(self.view.undo)
        self.action_undo.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Z))
        self.addAction(self.action_undo)

        self.undo_button = QtGui.QPushButton("Undo \n (key_Z)")
        self.undo_button.clicked.connect(self.view.undo)
        self.left_panel.layout().addWidget(self.undo_button)

        self.popup_button = QtGui.QPushButton("Done!")
        self.popup_button.clicked.connect(self.done)
        self.left_panel.layout().addWidget(self.popup_button)

        self.edge_button = QtGui.QPushButton("Show edges")
        self.edge_button.clicked.connect(self.show_edges)
        self.left_panel.layout().addWidget(self.edge_button)

        self.toggle_bg_button = QtGui.QPushButton("Toggle background")
        self.toggle_bg_button.setCheckable(True)
        self.toggle_bg_button.setChecked(False)
        self.toggle_bg_button.clicked.connect(self.toggle_bg)
        self.left_panel.layout().addWidget(self.toggle_bg_button)

        self.color_label = QtGui.QLabel()
        self.color_label.setWordWrap(True)
        self.color_label.setText("")
        self.left_panel.layout().addWidget(self.color_label)

        # complete the gui
        self.layout().addWidget(self.left_panel)
        self.layout().addWidget(self.view)

    def get_data(self, i, j, X, y, classification):
        b, g, r = self.image[i][j]
        sx = self.shiftx[i][j]
        sy = self.shifty[i][j]
        a = self.avg[i][j]
        c = self.bg[i][j]
        d = self.gr[i][j]
        e = self.rb[i][j]
        X.append((b, g, r, a, sx, sy, c, d, e))
        y.append(classification)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    # ex = SegmentationPicker('/home/dita/vlcsnap-2016-08-16-17h28m57s150.png')
    # ex = SegmentationPicker('/Users/flipajs/Pictures/vlcsnap-2016-08-31-15h33m56s683.png')
    ex = SegmentationPicker('/home/dita/img_67.png')
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()
