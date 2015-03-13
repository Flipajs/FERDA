__author__ = 'fnaiser'

from PyQt4 import QtGui
from gui.img_grid.img_grid_widget import ImgGridWidget

class InitHowWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project):
        super(InitHowWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)


        self.image_grid_widget = ImgGridWidget()
        self.vbox.addWidget(self.image_grid_widget)

        self.b = QtGui.QPushButton('reshape')
        self.b.clicked.connect(lambda: self.image_grid_widget.reshape(15))
        self.vbox.addWidget(self.b)

