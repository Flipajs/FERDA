from __future__ import division
from __future__ import unicode_literals
from past.utils import old_div
__author__ = 'fnaiser'

import numpy as np
from PyQt4 import QtGui, QtCore

from gui import gui_utils
from gui.settings import Settings as S_
from utils.img import get_safe_selection


class AssignmentRowWidget(QtGui.QWidget):
    def __init__(self, animal, img, regions, show_img, show_regions, show_colormarks, show_color_label, show_stats):
        super(AssignmentRowWidget, self).__init__()

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.setAlignment(QtCore.Qt.AlignLeft)
        self.setLayout(self.hbox)

        row_height = 50

        # TODO: remove constants
        cimg = np.zeros((row_height, 5, 3), dtype=np.uint8)
        cimg = np.asarray(cimg+animal.color_, dtype=np.uint8)

        self.color_label = gui_utils.get_image_label(cimg)
        self.info = QtGui.QLabel(animal.name)
        self.info.setFixedWidth(60)

        a_ = 100
        im_crop = get_safe_selection(img, animal.init_pos_center_[0] - old_div(a_,2), animal.init_pos_center_[1] - old_div(a_,2), a_, a_)
        self.im_label = gui_utils.get_image_label(im_crop)
        self.im_label.setFixedWidth(row_height)
        self.im_label.setFixedHeight(row_height)

        self.hbox.addWidget(self.info)
        self.hbox.addWidget(self.color_label)
        self.hbox.addWidget(self.im_label)


class AssignmentWidget(QtGui.QWidget):
    def __init__(self, animals, img=None, regions=None):
        super(AssignmentWidget, self).__init__()

        self.img = img
        self.regions = regions

        self.animals = animals
        self.show_img = True
        if img is None:
            self.show_img = False

        self.show_region = True
        if regions is None:
            self.show_region = False

        self.show_colormark = True if S_.colormarks.use else False

        self.show_color_label = True
        self.show_name = True
        self.show_stats = True

        self.vbox = QtGui.QVBoxLayout()

        self.setLayout(self.vbox)

        self.fill_rows()

    def fill_rows(self):
        for a in self.animals:
            w = AssignmentRowWidget(a, self.img, self.regions, self.show_img, self.show_region, self.show_colormark, self.show_color_label, self.show_stats)
            self.vbox.addWidget(w)