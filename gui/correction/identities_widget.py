from PyQt4 import QtGui, QtCore
from gui.img_controls.my_view import MyView
from utils.video_manager import get_auto_video_manager
from gui.img_controls.utils import cvimg2qtpixmap
import math
import cv2
from viewer.gui.img_controls import markers
from core.animal import colors_
from core.settings import Settings as S_
from core.graph.region_chunk import RegionChunk
import numpy as np
import sys
from core.animal import Animal
from gui import gui_utils

class AnimalVisu(QtGui.QWidget):
    def __init__(self, animal):
        super(AnimalVisu, self).__init__()

        self.hbox = QtGui.QHBoxLayout()
        self.setLayout(self.hbox)

        cimg = np.zeros((15, 10, 3), dtype=np.uint8)
        cimg = np.asarray(cimg+animal.color_, dtype=np.uint8)
        self.color_label = gui_utils.get_image_label(cimg)

        self.hbox.addWidget(self.color_label)
        self.hbox.addWidget(QtGui.QLabel(animal.name))

        self.orig_img = None
        self.adjusted_img = None

    def update_visu(self, img, region):
        if region is None:
            # set gray images
            pass


class IdentitiesWidget(QtGui.QWidget):
    def __init__(self, project):
        super(IdentitiesWidget, self).__init__()

        self.p = project

        # TOOD: remove in future
        self.p.animals = [
            # BGR colors
            Animal(0, name='red', color=(0, 0, 255)),
            Animal(1, 'light blue', color=(255, 100, 70)),
            Animal(2, 'dark blue', color=(230, 0, 0)),
            Animal(3, 'yellow', color=(0, 255, 255)),
            Animal(4, 'green', color=(0, 255, 0)),
            Animal(5, 'silver', color=(230, 230, 230))
        ]

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.animal_widgets = []
        for a in self.p.animals:
            w_ = AnimalVisu(a)
            self.animal_widgets.append(w_)
            self.vbox.addWidget(w_)


    def update(self, frame):
        regions = [None] * len(self.p.animals)

        # TODO: assign regions to ids..

        for i in range(len(self.p.animals)):
            self.animal_widgets[i].update_visu(img, regions[i])




if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    from core.project.project import Project

    project = Project()

    name = 'Cam1_orig'
    wd = '/Users/flipajs/Documents/wd/GT/'
    project.load(wd+name+'/cam1.fproj')

    ex = IdentitiesWidget(project)
    ex.show()

    app.exec_()
    app.deleteLater()
    sys.exit()

