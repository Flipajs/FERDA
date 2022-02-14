from PyQt6 import QtCore, QtGui, QtWidgets
import cv2, sys
import numpy as np
from . import painter

class CannyTest(QtWidgets.QWidget):
    def __init__(self, image):
        super(CannyTest, self).__init__()
        self.v1 = 100
        self.v2 = 100
        self.image = cv2.imread(image)
        self.blur = 15
        self.h, self.w, foo= self.image.shape
        self.make_gui()
        self.repaint()

    def slide1(self, value):
        self.v1 = value
        self.repaint()

    def slide2(self, value):
        self.v2 = value
        self.repaint()

    def slide3(self, value):
        if value % 2 == 0:
            self.blur = value +1
        else:
            self.blur = value
        self.repaint()

    def repaint(self):
        self.label_blur.setText("Blur: %d" % self.blur)
        self.blur_image = cv2.GaussianBlur(self.image, (self.blur, self.blur), 0)
        foo = painter.numpy2qimage(self.blur_image)
        self.view.set_image(foo)
        self.label.setText("Top: %d, Bottom: %d" % (self.v1, self.v2))
        edges = cv2.Canny(self.blur_image, self.v1, self.v2)
        edges.shape = ((self.h, self.w))
        r = np.zeros((self.h, self.w), dtype=np.uint8)
        g = np.asarray(edges, dtype=np.uint8)
        b = np.zeros((self.h, self.w), dtype=np.uint8)
        a = np.full((self.h, self.w), 100, dtype=np.uint8)
        rgb = np.dstack((b, g, r, a))
        foo = painter.rgba2qimage(rgb)
        self.view.set_overlay(foo)

    def make_gui(self):
        """
        Creates the widget. It is a separate method purely to save space
        :return: None
        """

        ##########################
        #          GUI           #
        ##########################

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setAlignment(QtCore.Qt.AlignBottom)

        # drawing area
        self.view = painter.Painter(self.image, paint_name="PINK", paint_r=255, paint_g=0, paint_b=238, paint_a=255)
        self.view.add_color("GREEN", 0, 255, 0, 255)

        # left panel widget
        self.left_panel = QtWidgets.QWidget()
        self.left_panel.setLayout(QtWidgets.QVBoxLayout())
        self.left_panel.layout().setAlignment(QtCore.Qt.AlignTop)
        # set left panel widget width to 300px
        self.left_panel.setMaximumWidth(300)
        self.left_panel.setMinimumWidth(300)

        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider1.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider1.setGeometry(30, 40, 50, 30)
        self.slider1.setRange(0, 200)
        self.slider1.setTickInterval(5)
        self.slider1.setValue(self.v1)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider1.valueChanged[int].connect(self.slide1)
        self.slider1.setVisible(True)
        self.left_panel.layout().addWidget(self.slider1)

        self.slider2 = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider2.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider2.setGeometry(30, 40, 50, 30)
        self.slider2.setRange(0, 200)
        self.slider2.setTickInterval(5)
        self.slider2.setValue(self.v2)
        self.slider2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider2.valueChanged[int].connect(self.slide2)
        self.slider2.setVisible(True)
        self.left_panel.layout().addWidget(self.slider2)

        self.slider3 = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider3.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider3.setGeometry(30, 40, 50, 30)
        self.slider3.setRange(0, 100)
        self.slider3.setTickInterval(5)
        self.slider3.setValue(self.blur)
        self.slider3.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider3.valueChanged[int].connect(self.slide3)
        self.slider3.setVisible(True)
        self.left_panel.layout().addWidget(self.slider3)

        self.label = QtWidgets.QLabel()
        self.label.setWordWrap(True)
        self.label.setText("")
        self.left_panel.layout().addWidget(self.label)

        self.label_blur = QtWidgets.QLabel()
        self.label_blur.setWordWrap(True)
        self.label_blur.setText("")
        self.left_panel.layout().addWidget(self.label_blur)

        self.pen_label = QtWidgets.QLabel()
        self.pen_label.setWordWrap(True)
        self.pen_label.setText("")
        self.left_panel.layout().addWidget(self.pen_label)

        # complete the gui
        self.layout().addWidget(self.left_panel)
        self.layout().addWidget(self.view)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    ex = CannyTest('/home/dita/vlcsnap-2016-08-16-17h28m57s150.png')
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()

