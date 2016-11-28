__author__ = 'naiser'

from PyQt4 import QtGui, QtCore
from gui.settings.default import get_tooltip
from gui import gui_utils
from core.settings import Settings as S_


class VisualisationTab(QtGui.QWidget):
    def __init__(self):
        super(VisualisationTab, self).__init__()

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.default_color_ = None

        self.default_color_b = QtGui.QPushButton('default region color')
        self.default_color_b.clicked.connect(self.color_picker)
        self.vbox.addWidget(self.default_color_b)


        self.segmentation_alpha = QtGui.QSpinBox()
        self.segmentation_alpha.setMinimum(0)
        self.segmentation_alpha.setMaximum(255)
        self.segmentation_alpha.setValue(230)

        self.vbox.addWidget(self.segmentation_alpha)

        self.populate()

    def color_picker(self):
        color = QtGui.QColorDialog.getColor()
        self.default_color_ = color

        self.default_color_b.setStyleSheet("QWidget { background-color: %s}" % color.name())

    def populate(self):
        # self.colormarks_box.setChecked(S_.colormarks.use)

        # self.igbr_i_weight.setValue(S_.colormarks.igbr_i_weight)

        self.segmentation_alpha.setValue(S_.visualization.segmentation_alpha)
        pass

    def restore_defaults(self):
        # TODO
        return

    def harvest(self):
        # TODO:
        if self.default_color_ is not None:
            S_.visualization.default_region_color = self.default_color_

        S_.visualization.segmentation_alpha = int(self.segmentation_alpha.value())