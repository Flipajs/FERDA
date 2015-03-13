__author__ = 'fnaiser'

from PyQt4 import QtGui

class ImgGridWidget(QtGui.QWidget):
    def __init__(self):
        super(ImgGridWidget, self).__init__()

        self.cols = 10
        self.id = 0

        self.grid = QtGui.QGridLayout()
        self.items = []
        cols = 10
        for i in range(100):
            self.items.append(QtGui.QLabel('test' + str(i)))
            row = i/cols
            col = i%cols
            self.grid.addWidget(self.items[i], row, col)


        self.setLayout(self.grid)

    def reshape(self, cols):
        self.grid2 = QtGui.QGridLayout()
        for i in range(self.grid.count()):
            row = i/cols
            col = i%cols

            self.grid2.addWidget(self.items[i], row, col)

        QtGui.QWidget().setLayout(self.layout())
        self.setLayout(self.grid2)

    def add_item(self, item):
        self.items.append(item)