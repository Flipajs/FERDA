__author__ = 'filip@naiser.cz'

from PyQt4 import QtGui
from numpy import arange, sin, pi, cos
from my_mpl_canvas import *
import sys
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png
import numpy as np
from gui.plot.plot_utils import line_picker


class PlotChunks(QtGui.QWidget):
    def __init__(self):
        super(PlotChunks, self).__init__()

        self.main_layout = QtGui.QVBoxLayout()
        self.setLayout(self.main_layout)

        self.central_widget = QtGui.QWidget()

        self.p3 = MyMplCanvas3D(self.central_widget, width=5, height=4, dpi=100)
        self.z = 0
        self.main_layout.addWidget(self.p3)

        self.b = QtGui.QPushButton('draw plane')
        self.b.clicked.connect(self.draw_plane)
        self.main_layout.addWidget(self.b)

        self.z = 0
        self.p3.figure.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0.0)
        self.update()
        # self.show()

    def plot_chunks(self, chunks, start_t=-1, end_t=-1):
        self.lines = []
        for ch in chunks:
            time = []
            x = []
            y = []

            if start_t < ch.start_t():
                time = [ch.start_t()]
                x = [ch.start_n.centroid()[1]]
                y = [ch.start_n.centroid()[0]]

            for t in range(max(start_t, ch.start_t()+1), min(end_t, ch.end_t()-1)):
                time.append(t)
                r = ch.get_reduced_at(t)
                x.append(r.centroid()[1])
                y.append(r.centroid()[0])

            if start_t > ch.end_t():
                time.append(ch.end_t())
                x.append(ch.end_n.centroid()[1])
                y.append(ch.end_n.centroid()[0])

            self.lines.append(self.p3.axes.plot(x, y, time, picker=line_picker)[0])
            self.p3.axes.hold(True)

            if start_t < ch.start_t():
                self.p3.axes.scatter(ch.start_n.centroid()[1], ch.start_n.centroid()[0], zs=ch.start_t())

            if end_t > ch.end_t():
                self.p3.axes.scatter(ch.end_n.centroid()[1], ch.end_n.centroid()[0], zs=ch.end_t())

            self.p3.axes.hold(True)

        self.p3.axes.hold(False)
        self.p3.axes.grid(False)

        self.z = start_t
        self.draw_plane()
        self.p3.draw()

    def draw_plane(self, level=-1):
        try:
            self.plane.remove()
            del self.plane
        except:
            pass

        level = self.z if level < 0 else level

        img = np.zeros((1000, 1000, 4), dtype=np.float)
        img[:,:,0] = 1
        img[:,:,3] = 0.2

        self.z += 1
        x, y = ogrid[0:img.shape[0], 0:img.shape[1]]
        self.plane = self.p3.axes.plot_surface(x, y, level, rstride=1000, cstride=1000, facecolors=img)

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