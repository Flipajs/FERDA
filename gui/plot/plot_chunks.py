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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.img import get_cmap


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
        self.chunks = None
        self.p3.figure.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0.0)
        self.update()
        # self.show()

    def plot_chunks(self, chunks, start_t=-1, end_t=-1):
        self.chunks = chunks

        self.lines = []
        for ch in chunks:
            time = []
            x = []
            y = []

            if start_t < ch.start_t():
                time = [ch.start_t()]
                x = [ch.start_n.centroid()[1]]
                y = [ch.start_n.centroid()[0]]

            for t in range(max(start_t, ch.start_t()+1), min(end_t, ch.end_t()1)):
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
                self.p3.axes.scatter(ch.start_n.centroid()[1], ch.start_n.centroid()[0], zs=ch.start_t(), marker='^')

            if end_t > ch.end_t():
                self.p3.axes.scatter(ch.end_n.centroid()[1], ch.end_n.centroid()[0], zs=ch.end_t(), marker='v')

            self.p3.axes.hold(True)

        self.p3.axes.hold(False)
        # self.p3.axes.grid(False)

        self.intersection_items = []
        self.intersection_positions = []

        self.z = start_t
        self.draw_plane()
        self.p3.draw()

    def draw_intersections(self, frame):
        for it in self.intersection_items:
            try:
                it.remove()
                del it
            except:
                pass

        self.intersection_items = []
        self.intersection_positions = []

        self.p3.axes.hold(True)
        i = 0
        for ch in self.chunks:
            if ch.start_t() <= frame <= ch.end_t():
                c = ch.get_centroid_in_time(frame)
                color_ = get_cmap(len(self.chunks))(i)
                self.intersection_items.append(self.p3.axes.scatter(c[1], c[0], zs=frame, color=color_))
                self.intersection_positions.append((c[0], c[1], color_))

            i += 1

    def draw_plane(self, level=-1):
        level = self.z if level < 0 else level
        for o in self.p3.figure.findobj(lambda x: isinstance(x, Poly3DCollection)):
            try:
                o.remove()
                del o
            except:
                pass

        x = [0, 1000, 1000, 0]
        y = [0, 0, 1000, 1000]
        z = [level, level,level,level]
        verts = [zip(x, y,z)]
        self.plane = Poly3DCollection(verts, alpha=0.1, facecolors='r')
        self.p3.axes.add_collection3d(self.plane)

        self.draw_intersections(level)
        self.p3.draw()

    def new_data(self, x, y):
        self.b4.process_data(x, y)

    def add_data(self, x, y, c=None):
        self.b4.add_data(x, y, c)

    def grid(self, status=True):
        self.b4.turn_grid(status)

    def set_onclick_callback(self, method):
        self.b4.onclick_callback = method
