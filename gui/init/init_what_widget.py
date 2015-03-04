__author__ = 'fnaiser'

from PyQt4 import QtGui

class InitWhatWidget(QtGui.QWidget):
    def __init__(self, finish_callback):
        super(InitWhatWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        self.finish_callback = finish_callback

