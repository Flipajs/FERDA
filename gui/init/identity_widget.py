__author__ = 'fnaiser'

import sys

from PyQt4 import QtGui, QtCore
from gui import gui_utils
from utils.img import get_safe_selection
from core.animal import Animal
from skimage.transform import resize, rescale
import numpy as np

class IdentityWidget(QtGui.QWidget):
    def __init__(self, img, animal, pos, radius):
        super(IdentityWidget, self).__init__()

        preview_size = 50
        color_stripe_width = 12
        self.animal = animal
        self.hbox = QtGui.QHBoxLayout()
        self.setLayout(self.hbox)

        self.img = img
        self.pos = pos
        self.radius = radius

        x_ = pos.x() - radius
        y_ = pos.y() - radius

        crop = get_safe_selection(img, y_, x_, radius*2, radius*2)
        crop = np.asarray(resize(crop, (preview_size, preview_size))*255, dtype=np.uint8)

        cimg = np.zeros((preview_size, color_stripe_width, 3), dtype=np.uint8)
        cimg = np.asarray(cimg+self.animal.color_, dtype=np.uint8)

        self.color_label = gui_utils.get_image_label(cimg)
        self.hbox.addWidget(self.color_label)

        self.img_label = gui_utils.get_image_label(crop)
        self.hbox.addWidget(self.img_label)

        self.name = QtGui.QLineEdit()
        self.name.setText(self.animal.name)
        self.hbox.addWidget(self.name)



        self.show()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = IdentityWidget()

    app.exec_()
    app.deleteLater()
    sys.exit()