__author__ = 'flipajs'

import cv2
from gui.img_controls import img_controls
import sys, os
from PyQt4 import QtGui

app = QtGui.QApplication(sys.argv)
ex = img_controls.ImgControls()

app.exec_()
app.deleteLater()
sys.exit()