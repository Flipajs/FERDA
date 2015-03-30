__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore
from gui.img_grid.img_grid_widget import ImgGridWidget


class ImgGridDialog(QtGui.QDialog):
    confirmed = QtCore.pyqtSignal("PyQt_PyObject")

    def __init__(self, parent=None, items=None):
        super(ImgGridDialog, self).__init__(parent)
        self.setLayout(QtGui.QVBoxLayout())

        self.img_grid = ImgGridWidget()
        self.set_items(items)

        self.cancel = QtGui.QPushButton('cancel')
        self.cancel.clicked.connect(self.close)
        self.confirm = QtGui.QPushButton('confirm')
        self.confirm.clicked.connect(self.confirm_clicked)

        self.layout().addWidget(self.img_grid)

        self.blayout = QtGui.QHBoxLayout()
        self.layout().addLayout(self.blayout)
        self.blayout.addWidget(self.cancel)
        self.blayout.addWidget(self.confirm)

    def set_items(self, items):
        self.img_grid.items = items
        self.img_grid.reshape(5)


    def confirm_clicked(self):
        ids = self.img_grid.get_selected()

        self.confirmed.emit(ids)
        self.close()
