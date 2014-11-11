__author__ = 'filip@naiser.cz'

import sys
from PyQt4 import QtGui

import sys
import os
import random
from matplotlib.backends import qt4_compat
use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

from numpy import arange, sin, pi, cos
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class TestWidget(QtGui.QMainWindow):
    def __init__(self):
        super(TestWidget, self).__init__()

        self.central_widget = QtGui.QWidget()

        self.lines_layout = QtGui.QVBoxLayout()
        self.lines_layout.setSpacing(0)
        self.lines_layout.setMargin(0)
        self.lines_layout.setContentsMargins(0, 0, 0, 0)

        self.main_line_layout = QtGui.QHBoxLayout()
        self.main_line_layout.setSpacing(0)
        self.main_line_layout.setMargin(0)
        self.main_line_layout.setContentsMargins(0, 0, 0, 0)
        self.main_line_widget = QtGui.QWidget()
        self.main_line_widget.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.main_line_widget.setLayout(self.main_line_layout)

        self.bottom_line_layout = QtGui.QVBoxLayout()
        self.bottom_line_layout.setSpacing(0)
        self.bottom_line_layout.setMargin(0)
        self.bottom_line_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_line_widget = QtGui.QWidget()
        self.bottom_line_widget.setLayout(self.bottom_line_layout)

        self.lines_layout.addWidget(self.main_line_widget)
        self.lines_layout.addWidget(self.bottom_line_widget)

        self.central_widget.setLayout(self.lines_layout)

        self.setCentralWidget(self.central_widget)
        self.centralWidget()

        self.b1 = QtGui.QPushButton("test 1234")
        self.b1.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.b1.setMaximumHeight(400)

        self.b2 = QtGui.QPushButton("test2 1231 123 123 1")

        self.b3 = QtGui.QPushButton("test 1234")
        self.b3.setMinimumWidth(200)
        self.b3.setMaximumWidth(200)

        self.main_line_layout.addWidget(self.b1)
        self.main_line_layout.addWidget(self.b2)
        self.main_line_layout.addWidget(self.b3)

        self.b4 = MyStaticMplCanvas(self.bottom_line_widget, width=5, height=4, dpi=100)
        x = arange(0.0, 3.0, 0.01)
        y = sin(2*pi*x)
        y2 = cos(2*pi*x)
        self.b4.process_data(y, x)
        self.b4.add_data(y2, x)
        # self.b4.setMinimumHeight(150)
        # self.b4.setMaximumHeight(150)
        self.bottom_line_layout.addWidget(self.b4)
        self.b4.setFixedHeight(1)


        # self.central_widget = QtGui.Q
        # self.centralWidget()
        self.update()
        self.show()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = TestWidget()

    app.exec_()
    app.deleteLater()
    sys.exit()
