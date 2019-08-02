__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore

from gui import gui_utils
from gui.settings import Settings as S_


class ParametersTab(QtGui.QWidget):
    def __init__(self):
        super(ParametersTab, self).__init__()

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.frame_layout = QtGui.QFormLayout()
        self.vbox.addLayout(self.frame_layout)

        self.blur_kernel_size = QtGui.QDoubleSpinBox()
        self.blur_kernel_size.setMinimum(0.0)
        self.blur_kernel_size.setMaximum(5.0)
        self.blur_kernel_size.setSingleStep(0.1)
        self.blur_kernel_size.setValue(0)
        self.frame_layout.addRow('Gblur kernel size', self.blur_kernel_size)

        self.populate()

    def populate(self):
        pass

    def restore_defaults(self):
        # TODO
        return

    def harvest(self):
        # TODO:
        pass
