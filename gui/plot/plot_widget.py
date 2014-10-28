__author__ = 'flipajs'

from PyQt4 import QtGui
from numpy import arange, sin, pi, cos
from my_mpl_canvas import *
import sys

class PlotWidget(QtGui.QWidget):
    def __init__(self):
        super(PlotWidget, self).__init__()

        self.main_layout = QtGui.QVBoxLayout()
        self.setLayout(self.main_layout)

        self.central_widget = QtGui.QWidget()
        self.b4 = MyStaticMplCanvas(self.central_widget, width=5, height=4, dpi=100)
        x = arange(0.0, 3.0, 0.01)
        y = sin(2*pi*x)
        y2 = cos(2*pi*x)
        self.b4.process_data(y, x)
        self.b4.add_data(y2, x)

        self.main_layout.addWidget(self.b4)

        self.update()
        self.show()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = PlotWidget()

    app.exec_()
    app.deleteLater()
    sys.exit()