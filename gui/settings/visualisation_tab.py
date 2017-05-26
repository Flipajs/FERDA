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

        self.no_single_id_filled = QtGui.QCheckBox('fill noise, multi-id, parts regions')
        self.no_single_id_filled.setChecked(True)
        self.vbox.addWidget(self.no_single_id_filled)

        self.trajectory_history = QtGui.QCheckBox('show trajectory history')
        self.trajectory_history.setChecked(True)
        self.vbox.addWidget(self.trajectory_history)

        self.history_depth = QtGui.QSpinBox()
        self.history_depth.setMinimum(0)
        self.history_depth.setMaximum(500)
        self.history_depth.setValue(10)
        self.vbox.addWidget(self.history_depth)

        self.history_depth_step = QtGui.QSpinBox()
        self.history_depth_step.setMinimum(1)
        self.history_depth_step.setMaximum(10)
        self.history_depth_step.setValue(1)
        self.vbox.addWidget(self.history_depth_step)

        self.history_alpha = QtGui.QDoubleSpinBox()
        self.history_alpha.setMinimum(0)
        self.history_alpha.setMaximum(10.0)
        self.history_alpha.setSingleStep(0.02)
        self.history_alpha.setValue(2.0)
        self.vbox.addWidget(self.history_alpha)

        self.populate()

    def color_picker(self):
        color = QtGui.QColorDialog.getColor()
        self.default_color_ = color

        self.default_color_b.setStyleSheet("QWidget { background-color: %s}" % color.name())

    def populate(self):
        # self.colormarks_box.setChecked(S_.colormarks.use)

        # self.igbr_i_weight.setValue(S_.colormarks.igbr_i_weight)

        self.segmentation_alpha.setValue(S_.visualization.segmentation_alpha)
        self.no_single_id_filled.setChecked(S_.visualization.no_single_id_filled)
        self.trajectory_history.setChecked(S_.visualization.trajectory_history)
        self.history_depth.setValue(S_.visualization.history_depth)
        self.history_depth_step.setValue(S_.visualization.history_depth_step)
        self.history_alpha.setValue(S_.visualization.history_alpha)

        pass

    def restore_defaults(self):
        # TODO
        return

    def harvest(self):
        # TODO:
        if self.default_color_ is not None:
            S_.visualization.default_region_color = self.default_color_

        S_.visualization.segmentation_alpha = int(self.segmentation_alpha.value())

        S_.visualization.no_single_id_filled = self.no_single_id_filled.isChecked()
        S_.visualization.trajectory_history = self.trajectory_history.isChecked()
        S_.visualization.history_depth = self.history_depth.value()
        S_.visualization.history_depth_step = self.history_depth_step.value()
        S_.visualization.history_alpha = self.history_alpha.value()