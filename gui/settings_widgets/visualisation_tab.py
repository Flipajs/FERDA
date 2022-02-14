__author__ = 'naiser'

from PyQt5 import QtGui, QtWidgets

from gui.settings import Settings as S_


class VisualisationTab(QtWidgets.QWidget):
    def __init__(self):
        super(VisualisationTab, self).__init__()

        self.fbox = QtWidgets.QFormLayout()
        self.setLayout(self.fbox)

        self.default_color_ = None

        self.default_color_b = QtWidgets.QPushButton('default region color')
        self.default_color_b.clicked.connect(self.color_picker)
        self.fbox.addRow(self.default_color_b)


        self.segmentation_alpha = QtWidgets.QSpinBox()
        self.segmentation_alpha.setMinimum(0)
        self.segmentation_alpha.setMaximum(255)
        self.segmentation_alpha.setValue(230)
        self.fbox.addRow(self.segmentation_alpha)

        self.no_single_id_filled = QtWidgets.QCheckBox('fill noise, multi-id, parts regions')
        self.no_single_id_filled.setChecked(True)
        self.fbox.addRow(self.no_single_id_filled)

        self.trajectory_history = QtWidgets.QCheckBox('show trajectory history')
        self.trajectory_history.setChecked(True)
        self.fbox.addRow(self.trajectory_history)

        self.history_depth = QtWidgets.QSpinBox()
        self.history_depth.setMinimum(0)
        self.history_depth.setMaximum(500)
        self.history_depth.setValue(10)
        self.fbox.addRow(self.history_depth)

        self.history_depth_step = QtWidgets.QSpinBox()
        self.history_depth_step.setMinimum(1)
        self.history_depth_step.setMaximum(10)
        self.history_depth_step.setValue(1)
        self.fbox.addRow(self.history_depth_step)

        self.history_alpha = QtWidgets.QDoubleSpinBox()
        self.history_alpha.setMinimum(0)
        self.history_alpha.setMaximum(10.0)
        self.history_alpha.setSingleStep(0.02)
        self.history_alpha.setValue(2.0)
        self.fbox.addRow(self.history_alpha)

        self.tracklet_len_per_px_sb = QtWidgets.QSpinBox()
        self.tracklet_len_per_px_sb.setMinimum(0)
        self.tracklet_len_per_px_sb.setValue(1)
        self.tracklet_len_per_px_sb.setMaximum(99)

        self.fbox.addRow('tracklet len per pixel (0 switch off)', self.tracklet_len_per_px_sb)

        self.populate()

    def color_picker(self):
        color = QtWidgets.QColorDialog.getColor()
        self.default_color_ = color

        self.default_color_b.setStyleSheet("QWidget { background-color: %s}" % color.name())

    def populate(self):
        self.segmentation_alpha.setValue(S_.visualization.segmentation_alpha)
        self.no_single_id_filled.setChecked(S_.visualization.no_single_id_filled)
        self.trajectory_history.setChecked(S_.visualization.trajectory_history)
        self.history_depth.setValue(S_.visualization.history_depth)
        self.history_depth_step.setValue(S_.visualization.history_depth_step)
        self.history_alpha.setValue(S_.visualization.history_alpha)
        self.tracklet_len_per_px_sb.setValue(S_.visualization.tracklet_len_per_px)

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
        S_.visualization.tracklet_len_per_px = self.tracklet_len_per_px_sb.value()
