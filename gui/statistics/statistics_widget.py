__author__ = 'fnaiser'

from PyQt4 import QtGui


class StatisticsWidget(QtGui.QWidget):
    def __init__(self):
        super(StatisticsWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        b = QtGui.QPushButton('statistics')

        self.setLayout(self.vbox)
        self.vbox.addWidget(b)
