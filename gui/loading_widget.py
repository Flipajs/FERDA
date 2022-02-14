__author__ = 'fnaiser'
from PyQt6 import QtGui, QtWidgets


class LoadingWidget(QtWidgets.QWidget):
    def __init__(self, max_range=100, text='processing...'):
        super(LoadingWidget, self).__init__()
        self.vbox = QtWidgets.QVBoxLayout()
        self.setLayout(self.vbox)
        self.text_l = QtWidgets.QLabel(text)
        self.vbox.addWidget(self.text_l)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setMaximum(max_range)
        self.vbox.addWidget(self.progress_bar)

    def update_progress(self, val):
        self.progress_bar.setValue(val*100)
        QtWidgets.QApplication.processEvents()

    def update_text(self, text):
        self.text_l.setText(text)

    def finished(self):
        print("FINISHED")
