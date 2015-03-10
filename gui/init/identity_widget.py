__author__ = 'fnaiser'

import sys

from PyQt4 import QtGui, QtCore
from gui import gui_utils
from utils.img import get_safe_selection
from core.animal import Animal
from skimage.transform import resize, rescale
import numpy as np

class IdentityWidget(QtGui.QWidget):
    update_identity = QtCore.pyqtSignal(int)

    def __init__(self, img, animal, pos, radius, classes=None, groups=None):
        super(IdentityWidget, self).__init__()

        self.preview_size = 50
        self.color_stripe_width = 12
        self.animal = animal
        self.hbox = QtGui.QHBoxLayout()
        self.hbox.setSpacing(0)

        self.setLayout(self.hbox)
        self.classes = classes
        self.groups = groups

        self.img = img
        self.pos = pos
        self.radius = radius

        x_ = pos.x() - radius
        y_ = pos.y() - radius

        crop = get_safe_selection(img, y_, x_, radius*2, radius*2)
        crop = np.asarray(resize(crop, (self.preview_size, self.preview_size))*255, dtype=np.uint8)

        cimg = np.zeros((self.preview_size, self.color_stripe_width, 3), dtype=np.uint8)
        cimg = np.asarray(cimg+self.animal.color_, dtype=np.uint8)

        self.color_label = gui_utils.get_image_label(cimg)
        self.color_label.mouseReleaseEvent = self.color_label_clicked

        self.hbox.addWidget(self.color_label)

        self.img_label = gui_utils.get_image_label(crop)
        self.hbox.addWidget(self.img_label)

        self.name_class_group_layout = QtGui.QVBoxLayout()
        self.name_class_group_layout.setSpacing(0)
        self.name_class_group_layout.setMargin(0)
        self.name_class_group_widget = QtGui.QWidget()
        self.name_class_group_widget.setLayout(self.name_class_group_layout)

        self.name = QtGui.QLineEdit()
        self.name.setText(self.animal.name)
        self.name_class_group_layout.addWidget(self.name)

        self.classes_groups_layout = QtGui.QHBoxLayout()
        self.classes_groups_layout.setSpacing(0)
        self.classes_groups_layout.setMargin(0)
        self.classes_groups_widget = QtGui.QWidget()
        self.classes_groups_widget.setLayout(self.classes_groups_layout)

        if self.classes:
            self.classes_box = QtGui.QComboBox()
            id = 0
            for c in self.classes:
                self.classes_box.addItem(c.name, id)
                id += 1

        if self.groups:
            self.groups_box = QtGui.QComboBox()
            id = 0
            for g in self.groups:
                self.groups_box.addItem(g.name, id)
                id += 1

        self.classes_groups_layout.addWidget(self.classes_box)
        self.classes_groups_layout.addWidget(self.groups_box)

        self.name_class_group_layout.addWidget(self.classes_groups_widget)

        self.hbox.addWidget(self.name_class_group_widget)

        self.delete_button = QtGui.QPushButton('x')
        self.delete_button.setFixedWidth(30)
        self.delete_button.setFixedHeight(50)
        self.hbox.addWidget(self.delete_button)

    def color_label_clicked(self, event):
        col = QtGui.QColorDialog.getColor()
        col = np.array([col.blue(), col.green(), col.red()])

        cimg = np.zeros((self.preview_size, self.color_stripe_width, 3), dtype=np.uint8)
        cimg = np.asarray(cimg+col, dtype=np.uint8)

        self.hbox.removeItem(self.hbox.itemAt(0))
        self.color_label = gui_utils.get_image_label(cimg)
        self.color_label.mouseReleaseEvent = self.color_label_clicked
        self.hbox.insertWidget(0, self.color_label)

        self.animal.color_ = col
        self.update_identity.emit(self.animal.id)

    def update_classes(self, classes):
        self.classes = classes

        val = self.classes_box.currentText()
        self.classes_box.setParent(None)

        if self.classes:
            self.classes_box = QtGui.QComboBox()
            id = 0
            for c in self.classes:
                self.classes_box.addItem(c.name, id)
                if c.name == val:
                    self.classes_box.setCurrentIndex(id)

                id += 1

        self.classes_groups_layout.insertWidget(0, self.classes_box)

    def update_groups(self, groups):
        self.groups = groups

        val = self.groups_box.currentText()
        self.groups_box.setParent(None)

        if self.groups:
            self.groups_box = QtGui.QComboBox()
            id = 0
            for c in self.groups:
                self.groups_box.addItem(c.name, id)
                if c.name == val:
                    self.groups_box.setCurrentIndex(id)

                id += 1

        self.classes_groups_layout.insertWidget(1, self.groups_box)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = IdentityWidget()

    app.exec_()
    app.deleteLater()
    sys.exit()