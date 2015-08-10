__author__ = 'fnaiser'

from PyQt4 import QtGui

class LoadingWidget(QtGui.QWidget):
    def __init__(self, max_range=0):
        super(LoadingWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        self.vbox.addWidget(QtGui.QLabel('processing...'))
        self.progress_bar = QtGui.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setMaximum(max_range)
        self.vbox.addWidget(self.progress_bar)

    def update_progress(self, val):
        print val
        self.progress_bar.setValue(val)

    def finished(self):
        print "FINISHED"