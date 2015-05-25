__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore
from gui.settings.default import get_tooltip
from gui import gui_utils
from core.settings import Settings as S_


class ParametersTab(QtGui.QWidget):
    def __init__(self):
        super(ParametersTab, self).__init__()

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        # COLORMARKS
        self.colormarks_box = QtGui.QGroupBox('Colormarks')
        self.colormarks_box.setCheckable(True)
        self.colormarks_box.setChecked(S_.colormarks.use)
        self.colormarks_box.toggled.connect(lambda : gui_utils.gbox_collapse_expand(self.colormarks_box))
        if not self.colormarks_box.isChecked():
            gui_utils.gbox_collapse_expand(self.colormarks_box)

        self.colormarks_box.setAlignment(QtCore.Qt.AlignLeft)
        self.colormarks_box.setLayout(QtGui.QFormLayout())

        self.igbr_i_weight = gui_utils.get_double_spin_box(0, 10, 0.01, 'igbr_i_weight')
        self.colormarks_mser_max_area = gui_utils.get_spin_box(0, 1000000, 1, 'colormarks_mser_max_area')
        self.colormarks_mser_min_area = gui_utils.get_spin_box(0, 1000000, 1, 'colormarks_mser_min_area')
        self.colormarks_mser_min_margin = gui_utils.get_spin_box(0, 255, 1, 'colormarks_mser_min_margin')
        self.colormarks_avg_radius = gui_utils.get_spin_box(0, 1000, 1, 'colormarks_avg_radius')
        self.colormarks_avg_radius.setStyleSheet("QSpinBox { background-color: #E64016; }");
        self.colormarks_debug = gui_utils.get_checkbox('', 'colormarks_debug')

        self.colormarks_box.layout().addRow('mser max area', self.colormarks_mser_max_area)
        self.colormarks_box.layout().addRow('mser min area', self.colormarks_mser_min_area)
        self.colormarks_box.layout().addRow('mser min margin', self.colormarks_mser_min_margin)
        self.colormarks_box.layout().addRow('average radius', self.colormarks_avg_radius)
        self.colormarks_box.layout().addRow('Igbr I-weight', self.igbr_i_weight)
        self.colormarks_box.layout().addRow('Store debug info', self.colormarks_debug)

        self.vbox.addWidget(self.colormarks_box)
        self.frame_layout = QtGui.QFormLayout()
        self.vbox.addLayout(self.frame_layout)

        self.mser_max_area = QtGui.QDoubleSpinBox()
        self.mser_max_area.setMinimum(0.0001)
        self.mser_max_area.setSingleStep(0.0001)
        self.mser_max_area.setMaximum(1.0)
        self.frame_layout.addRow('MSER Max relative area', self.mser_max_area)

        self.mser_min_area = QtGui.QSpinBox()
        self.mser_min_area.setMinimum(0)
        self.mser_min_area.setMaximum(1000)
        self.frame_layout.addRow('MSER Min area', self.mser_min_area)

        self.mser_min_margin = QtGui.QSpinBox()
        self.mser_min_margin.setMinimum(1)
        self.mser_min_margin.setMaximum(100)
        self.frame_layout.addRow('MSER Min margin', self.mser_min_margin)

        self.populate()

    def populate(self):
        self.colormarks_box.setChecked(S_.colormarks.use)
        self.mser_max_area.setValue(S_.mser.max_area)
        self.mser_min_area.setValue(S_.mser.min_area)
        self.mser_min_margin.setValue(S_.mser.min_margin)

        # self.igbr_i_weight.setValue(S_.colormarks.igbr_i_weight)

    def restore_defaults(self):
        # TODO
        return

    def harvest(self):
        # TODO:
        S_.colormarks.use = self.colormarks_box.isChecked()
        S_.mser.max_area = self.mser_max_area.value()
        S_.mser.min_area = self.mser_max_area.value()
        S_.mser.min_margin = self.mser_min_margin.value()
