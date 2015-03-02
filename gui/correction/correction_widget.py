__author__ = 'fnaiser'

from PyQt4 import QtGui


class CorrectionWidget(QtGui.QWidget):
    def __init__(self):
        super(CorrectionWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        b = QtGui.QPushButton('corrector')

        self.setLayout(self.vbox)
        self.vbox.addWidget(b)
