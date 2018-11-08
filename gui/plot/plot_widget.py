from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
__author__ = 'filip@naiser.cz'

from PyQt4 import QtGui
from numpy import arange, sin, pi, cos
from .my_mpl_canvas import *
import sys
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png
import numpy as np


class PlotWidget(QtGui.QWidget):
    def __init__(self):
        super(PlotWidget, self).__init__()

        self.main_layout = QtGui.QVBoxLayout()
        self.setLayout(self.main_layout)

        self.central_widget = QtGui.QWidget()

        self.p3 = MyMplCanvas3D(self.central_widget, width=5, height=4, dpi=100)

        self.z = 0
        self.draw_plane()
        self.main_layout.addWidget(self.p3)

        self.b = QtGui.QPushButton('draw plane')
        self.b.clicked.connect(self.draw_plane)
        self.main_layout.addWidget(self.b)

        self.update()
        self.show()

    def draw_plane(self):
        try:
            self.plane.remove()
            del self.plane
        except:
            pass

        img = np.zeros((1000, 1000, 4))
        img[:,:,3] = 0.2

        self.z += 300
        print(self.z)
        x, y = ogrid[0:img.shape[0], 0:img.shape[1]]
        self.plane = self.p3.axes.plot_surface(x, y, self.z, rstride=1000, cstride=1000, facecolors=img)

        self.p3.draw()

    def new_data(self, x, y):
        self.b4.process_data(x, y)

    def add_data(self, x, y, c=None):
        self.b4.add_data(x, y, c)

    def grid(self, status=True):
        self.b4.turn_grid(status)

    def set_onclick_callback(self, method):
        self.b4.onclick_callback = method

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = PlotWidget()

    app.exec_()
    app.deleteLater()
    sys.exit()