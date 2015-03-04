__author__ = 'fnaiser'

from PyQt4 import QtGui

class InitHowWidget(QtGui.QWidget):
    def __init__(self, finish_callback):
        super(InitHowWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
