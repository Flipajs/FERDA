import sys
from PyQt4 import QtGui

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import random

class EigenWidget(QtGui.QDialog):
    def __init__(self, pca, eigens, ant):
        super(EigenWidget, self).__init__()

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.eigens = QtGui.QHBoxLayout()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(self.eigens)
        self.setLayout(layout)

        self.eigens_layouts = []
        self.prepare_eigens(eigens)
        for l in self.eigens_layouts:
            self.eigens.addLayout(l)

        # Just some button connected to `plot` method
        self.button = QtGui.QPushButton('Plot')
        self.button.clicked.connect(self.plot)

    def prepare_eigens(self, eigens):
        for eigen in eigens:
            layout = QtGui.QVBoxLayout()
            figure = plt.figure()
            ax = figure.add_subplot(111)
            ax.plot(eigen[::2], eigen[1::2])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # ax.set_aspect('equal')
            canvas = FigureCanvas(figure)
            canvas.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                       QtGui.QSizePolicy.Expanding)
            canvas.updateGeometry()
            canvas.draw()
            slider = QtGui.QSlider()
            layout.addWidget(canvas)
            layout.addWidget(slider)
            self.eigens_layouts.append(layout)


    def plot(self):
        ''' plot some random stuff '''
        # random data
        data = [random.random() for i in range(10)]

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.hold(False)

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()

