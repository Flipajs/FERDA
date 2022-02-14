from PyQt5 import QtCore, QtGui, QtWidgets
import cv2, sys
import numpy as np
from . import painter
import matplotlib.pyplot as plt

class DiffTest(QtWidgets.QWidget):
    def __init__(self, image):
        super(DiffTest, self).__init__()
        self.shift_x = 0
        self.shift_y = 2
        self.image = cv2.imread(image)
        self.make_gui()
        self.h, self.w, foo = self.image.shape
        self.repaint()

    def slide1(self, value):
        self.shift_x = value
        self.repaint()

    def slide2(self, value):
        self.shift_y = value
        self.repaint()

    def repaint(self):
        self.label.setText("X: %d, Y: %d" % (self.shift_x, self.shift_y))

        M = np.float32([[1,0,self.shift_x],[0,1,self.shift_y]])
        g1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.blur_image = cv2.GaussianBlur(g1, (3, 3), 0.3)
        self.dst = cv2.warpAffine(self.image, M, (self.w, self.h))
        g2 = cv2.cvtColor(self.dst, cv2.COLOR_BGR2GRAY)
        self.blur_dst = cv2.GaussianBlur(g2, (3, 3), 0.3)

        self.view2.set_overlay(painter.numpy2qimage(self.dst))

    def show_edges(self):
        # g1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # g2 = cv2.cvtColor(self.dst, cv2.COLOR_BGR2GRAY)
        # dif = np.asarray(g1, dtype=np.int32) - np.asarray(g2, dtype=np.int32)
        dif = np.abs(np.asarray(self.blur_image, dtype=np.int32) - np.asarray(self.blur_dst, dtype=np.int32))
        #g3 = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
        plt.imshow(dif)
        plt.show()

    def add_alpha(self, image):
        # get color channels from the image and format each channel not to be a w*h table but a w*h long list of pixel values
        h, w = image.shape
        """
        print image.shape
        red = image[:,:,2]
        red.shape = ((h*w, 1))
        green = image[:,:,1]
        green.shape = ((h*w, 1))
        blue = image[:,:,0]
        blue.shape = ((h*w, 1))"""
    
        foo = np.full((h, w), 100, dtype=np.uint8)
    
        # create a 4D image that has edge value as the fourth channel
        return np.dstack((image, image, image, foo))

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
        self.view1 = painter.Painter(self.image, paint_name="PINK", paint_r=255, paint_g=0, paint_b=238, paint_a=255)
        self.view2 = painter.Painter(self.image, paint_name="PINK", paint_r=255, paint_g=0, paint_b=238, paint_a=255)

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
        self.slider1.setRange(0, 50)
        self.slider1.setTickInterval(2)
        self.slider1.setValue(self.shift_x)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider1.valueChanged[int].connect(self.slide1)
        self.slider1.setVisible(True)
        self.left_panel.layout().addWidget(self.slider1)

        self.slider2 = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider2.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider2.setGeometry(30, 40, 50, 30)
        self.slider2.setRange(0, 50)
        self.slider2.setTickInterval(2)
        self.slider2.setValue(self.shift_y)
        self.slider2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider2.valueChanged[int].connect(self.slide2)
        self.slider2.setVisible(True)
        self.left_panel.layout().addWidget(self.slider2)

        self.label = QtWidgets.QLabel()
        self.label.setWordWrap(True)
        self.label.setText("")
        self.left_panel.layout().addWidget(self.label)

        self.edge_button = QtWidgets.QPushButton("Show edges")
        self.edge_button.clicked.connect(self.show_edges)
        self.left_panel.layout().addWidget(self.edge_button)

        # complete the gui
        self.layout().addWidget(self.left_panel)
        self.layout().addWidget(self.view1)
        self.layout().addWidget(self.view2)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    ex = DiffTest('/home/dita/img_67.png')
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()

