from __future__ import unicode_literals
__author__ = 'filip@naiser.cz'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt4 import QtGui, QtCore
from numpy import arange, sin, pi, cos
from mpl_toolkits.mplot3d import axes3d, Axes3D

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        cid = self.mpl_connect('button_press_event', self.onclick)

        self.onclick_callback = None

        # self.mpl_disconnect(cid)

    def compute_initial_figure(self):
        pass

    def onclick(self, event):
        if self.onclick_callback:
            self.onclick_callback(event.xdata, event.ydata)
            # print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)

class MyMplCanvas3D(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)

        self.axes = fig.add_subplot(111, projection='3d')

        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # cid = self.mpl_connect('button_press_event', self.onclick)

        self.onclick_callback = None
        self.axes.mouse_init()

        # self.mpl_disconnect(cid)

    def compute_initial_figure(self):
        pass

    def onclick(self, event):
        if self.onclick_callback:
            self.onclick_callback(event.xdata, event.ydata)
            # print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)



class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    def compute_initial_figure(self):
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axes.plot(t, s)

    def process_data(self, x, y):
        self.axes.plot(x, y)
        self.draw()

    def add_data(self, x, y, c=None):
        self.axes.hold(True)
        if c:
            self.axes.scatter(x, y, color='r')
        else:
            self.axes.plot(x, y)

        self.axes.hold(False)

    def turn_grid(self, status):
        self.axes.grid(status)
        self.draw()