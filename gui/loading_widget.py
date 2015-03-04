__author__ = 'fnaiser'

from PyQt4 import QtGui

class LoadingWidget(QtGui.QWidget):
    def __init__(self):
        super(LoadingWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        self.progress_bar = QtGui.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.vbox.addWidget(self.progress_bar)
