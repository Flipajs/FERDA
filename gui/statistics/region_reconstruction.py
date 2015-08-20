__author__ = 'flipajs'


import sys
import cv2
import numpy as np

from PyQt4 import QtGui
from PyQt4 import QtCore
from skimage.transform import resize
from core.region.mser import get_msers_
from core.project.project import Project
from gui.img_controls.utils import cvimg2qtpixmap
from scripts.region_graph3 import visualize_nodes


class RegionReconstruction(QtGui.QWidget):
    def __init__(self, project, query):
        pass



if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    p = Project()
    p.load('')
    p.mser_parameters.min_area = 30
    p.mser_parameters.min_margin = 5

    ex = RegionReconstruction(p)
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()