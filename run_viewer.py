__author__ = 'flipajs'

from viewer.gui.img_controls import img_controls
import sys
from PyQt4 import QtGui
import video_manager


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = img_controls.ImgControls()

    app.exec_()
    app.deleteLater()
    sys.exit()
