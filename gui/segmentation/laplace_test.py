from __future__ import absolute_import
from PyQt4 import QtGui, QtCore
import cv2, sys
import numpy as np
from . import painter
import matplotlib.pyplot as plt

class LaplacianTest(QtGui.QWidget):
    def __init__(self, image):
        super(LaplacianTest, self).__init__()
        self.k1 = 21
        self.k2 = 21
        self.image = cv2.imread(image)
        self.sigma1 = 50
        self.sigma2 = 50
        self.make_gui()
        self.h, self.w, foo= self.image.shape
        self.slide3(self.sigma1)
        self.slide4(self.sigma2)
        self.repaint()

    def slide1(self, value):
        self.k1 = (value*2) + 1
        self.repaint()

    def slide2(self, value):
        self.k2 = (value*2) + 1
        self.repaint()

    def slide3(self, value):
        self.sigma1 = value / float(10)
        self.repaint()

    def slide4(self, value):
        self.sigma2 = value / float(10)
        self.repaint()

    def repaint(self):
        self.label1.setText("K1: %d, Sigma1: %f" % (self.k1, self.sigma1))
        self.blur_image1 = cv2.GaussianBlur(self.image, (self.k1, self.k1), self.sigma1)
        self.grey1 = cv2.cvtColor(self.blur_image1, cv2.COLOR_BGR2GRAY)

        self.label2.setText("K2: %d, Sigma2: %f" % (self.k2, self.sigma2))
        self.blur_image2 = cv2.GaussianBlur(self.image, (self.k2, self.k2), self.sigma2)
        self.grey2 = cv2.cvtColor(self.blur_image2, cv2.COLOR_BGR2GRAY)

        self.view1.set_overlay(painter.numpy2qimage(self.blur_image1))
        self.view2.set_overlay(painter.numpy2qimage(self.blur_image2))

    
        # find edges on the blurred image
        self.laplace1 = cv2.Laplacian(self.grey1, cv2.CV_64F)
        self.laplace1 -= np.min(self.laplace1)
        self.laplace1 /= np.max(self.laplace1)
        self.laplace1 *= 255
        self.laplace1 = np.asarray(self.laplace1, dtype=np.uint8)

        self.laplace2 = cv2.Laplacian(self.grey2, cv2.CV_64F)
        self.laplace2 -= np.min(self.laplace2)
        self.laplace2 /= np.max(self.laplace2)
        self.laplace2 *= 255
        self.laplace2 = np.asarray(self.laplace2, dtype=np.uint8)

    def show_edges(self):

        #dif = self.laplace1 - self.laplace2
        
        dif = np.asarray(self.grey1, dtype=np.int32) - np.asarray(self.grey2, dtype=np.int32)
    
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

        self.setLayout(QtGui.QHBoxLayout())
        self.layout().setAlignment(QtCore.Qt.AlignBottom)

        # drawing area
        self.view1 = painter.Painter(self.image, paint_name="PINK", paint_r=255, paint_g=0, paint_b=238, paint_a=255)
        self.view2 = painter.Painter(self.image, paint_name="PINK", paint_r=255, paint_g=0, paint_b=238, paint_a=255)

        # left panel widget
        self.left_panel = QtGui.QWidget()
        self.left_panel.setLayout(QtGui.QVBoxLayout())
        self.left_panel.layout().setAlignment(QtCore.Qt.AlignTop)
        # set left panel widget width to 300px
        self.left_panel.setMaximumWidth(300)
        self.left_panel.setMinimumWidth(300)

        self.slider1 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider1.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider1.setGeometry(30, 40, 50, 30)
        self.slider1.setRange(0, 50)
        self.slider1.setTickInterval(2)
        self.slider1.setValue(self.k1)
        self.slider1.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider1.valueChanged[int].connect(self.slide1)
        self.slider1.setVisible(True)
        self.left_panel.layout().addWidget(self.slider1)

        self.slider2 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider2.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider2.setGeometry(30, 40, 50, 30)
        self.slider2.setRange(0, 50)
        self.slider2.setTickInterval(2)
        self.slider2.setValue(self.k2)
        self.slider2.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider2.valueChanged[int].connect(self.slide2)
        self.slider2.setVisible(True)
        self.left_panel.layout().addWidget(self.slider2)

        self.slider3 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider3.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider3.setRange(1, 100)
        self.slider3.setTickInterval(1)
        self.slider3.setValue(self.sigma1)
        self.slider3.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider3.valueChanged[int].connect(self.slide3)
        self.slider3.setVisible(True)
        self.left_panel.layout().addWidget(self.slider3)

        self.slider4 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider4.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider4.setRange(1, 100)
        self.slider4.setTickInterval(1)
        self.slider4.setValue(self.sigma2)
        self.slider4.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider4.valueChanged[int].connect(self.slide4)
        self.slider4.setVisible(True)
        self.left_panel.layout().addWidget(self.slider4)

        self.label1 = QtGui.QLabel()
        self.label1.setWordWrap(True)
        self.label1.setText("")
        self.left_panel.layout().addWidget(self.label1)

        self.label2 = QtGui.QLabel()
        self.label2.setWordWrap(True)
        self.label2.setText("")
        self.left_panel.layout().addWidget(self.label2)

        self.edge_button = QtGui.QPushButton("Show edges")
        self.edge_button.clicked.connect(self.show_edges)
        self.left_panel.layout().addWidget(self.edge_button)

        # complete the gui
        self.layout().addWidget(self.left_panel)
        self.layout().addWidget(self.view1)
        self.layout().addWidget(self.view2)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    ex = LaplacianTest('/home/dita/img_67.png')
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()

