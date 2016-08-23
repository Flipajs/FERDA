import sys
import random
import gc
from functools import partial
from PyQt4 import QtGui, QtCore
import painter
import cv2
import numpy as np

from sklearn.ensemble import RandomForestClassifier

__author__ = 'dita'


class SegmentationPicker(QtGui.QWidget):

    def __init__(self, img, done_callback=None, pen_size=10, undo_len=10, debug=False, paint_r=255, paint_g=0, paint_b=238):

        super(SegmentationPicker, self).__init__()

        self.DEBUG = debug
        self.done_callback = done_callback

        self.img_path = img
        self.image = cv2.imread(self.img_path)
        self.pen_size = pen_size

        self.color = [paint_r, paint_g, paint_b]

        self.make_gui()

    def done(self):
        result = self.view.get_result()
        background = result["PINK"]
        foreground = result["GREEN"]

        X = []
        y = []
 
        nzero = np.nonzero(background[0])
        print type(nzero[0])
        for i, j in zip(nzero[0], nzero[1]):
            b, g, r = self.image[j][i]
            X.append((b, g, r))
            y.append(0)

        nzero = np.nonzero(foreground[0])
        for i, j in zip(nzero[0], nzero[1]):
            b, g, r = self.image[j][i]
            X.append((b, g, r))
            y.append(1)

        rfc = RandomForestClassifier()
        rfc.fit(X, y)

        h, w, c = self.image.shape
        print h, w, c
        self.image.shape = ((h*w, c))
        print self.image.shape
        mask1 = np.zeros_like(self.image)
        mask2 = (h*w)
        mask1 = rfc.predict(self.image)
        mask1.shape = ((h, w))
        self.image.shape = ((h, w, c))
        mask1 = np.asarray(mask1 * 255, dtype=np.uint8)
        print "Done"
        cv2.imshow("Fooo", mask1)
        cv2.waitKey(0)
        print mask1

    def pink(self):
        self.view.set_pen_color("PINK")
        self.color_buttons[1].setChecked(False)
        self.color_buttons[2].setChecked(False)

    def green(self):
        self.view.set_pen_color("GREEN")
        self.color_buttons[0].setChecked(False)
        self.color_buttons[2].setChecked(False)

    def set_eraser(self):
        self.view.set_pen_color(None)

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
        self.view = painter.Painter(self.image)
        self.view.add_color("GREEN", 0, 255, 0)

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

        self.color_buttons = []
        pink_button = QtGui.QPushButton("Pink")
        pink_button.setCheckable(True)
        pink_button.setChecked(True)
        pink_button.clicked.connect(self.pink)
        color_widget.layout().addWidget(pink_button)
        self.color_buttons.append(pink_button)

        green_button = QtGui.QPushButton("Green")
        green_button.setCheckable(True)
        green_button.clicked.connect(self.green)
        color_widget.layout().addWidget(green_button)
        self.color_buttons.append(green_button)

        eraser_button = QtGui.QPushButton("Eraser")
        eraser_button.setCheckable(True)
        eraser_button.clicked.connect(self.set_eraser)
        color_widget.layout().addWidget(eraser_button)
        self.left_panel.layout().addWidget(color_widget)
        self.color_buttons.append(eraser_button)

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

        self.color_label = QtGui.QLabel()
        self.color_label.setWordWrap(True)
        self.color_label.setText("")
        self.left_panel.layout().addWidget(self.color_label)

        # complete the gui
        self.layout().addWidget(self.left_panel)
        self.layout().addWidget(self.view)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    ex = SegmentationPicker('/home/dita/vlcsnap-2016-08-16-17h28m57s150.png')
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()
