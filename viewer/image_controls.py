from utils import video_manager

__author__ = 'flipajs'

import cv2
from gui.img_controls import img_controls
import sys
from PyQt4 import QtGui

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = img_controls.ImgControls()

    app.exec_()
    app.deleteLater()
    sys.exit()