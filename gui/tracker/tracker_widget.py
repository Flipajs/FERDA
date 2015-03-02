__author__ = 'fnaiser'

from PyQt4 import QtGui


class TrackerWidget(QtGui.QWidget):
    def __init__(self):
        super(TrackerWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        b = QtGui.QPushButton('tracker')

        self.setLayout(self.vbox)
        self.vbox.addWidget(b)
