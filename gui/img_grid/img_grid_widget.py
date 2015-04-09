__author__ = 'fnaiser'

from PyQt4 import QtGui

class ImgGridWidget(QtGui.QWidget):
    def __init__(self):
        super(ImgGridWidget, self).__init__()

        self.element_width = 300
        self.cols = 5
        self.id = 0


        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(1)
        self.grid.setMargin(0)

        self.grid_widget = QtGui.QWidget()
        self.grid_widget.setLayout(self.grid)

        self.scroll_ = QtGui.QScrollArea()
        self.scroll_.setWidgetResizable(True)
        self.scroll_.setWidget(self.grid_widget)
        self.set_width_()

        self.items = []

        self.setLayout(QtGui.QVBoxLayout())
        self.layout().setSpacing(0)
        self.layout().setMargin(0)
        self.layout().addWidget(self.scroll_)


    def reshape(self, cols, element_width=100):
        self.cols = cols
        self.element_width = element_width

        grid2 = QtGui.QGridLayout()

        for i in range(len(self.items)):
            row = i/cols
            col = i%cols

            grid2.addWidget(self.items[i], row, col)

        QtGui.QWidget().setLayout(self.grid)
        self.grid_widget.setLayout(grid2)
        self.grid = grid2
        self.grid.setSpacing(1)
        self.grid.setMargin(0)

        self.set_width_()

    def add_item(self, item):
        self.items.append(item)

        row = self.id/self.cols
        col = self.id%self.cols

        self.grid.addWidget(self.items[self.id], row, col)

        self.set_width_()

        self.id += 1

    def set_width_(self):
        vscroll_shint = self.scroll_.verticalScrollBar().sizeHint()
        self.scroll_.setFixedWidth(self.cols*self.element_width + (self.cols + 1) + vscroll_shint.width())

    def get_selected(self):
        ids = []

        for i in range(len(self.items)):
            if self.items[i].selected:
                ids.append(self.items[i].id)

        return ids