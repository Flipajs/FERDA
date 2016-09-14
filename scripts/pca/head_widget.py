import matplotlib.pyplot as plt
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

import numpy as np
from skimage.viewer.utils import FigureCanvas
import pca

BACK_UP_CONSTANT = 50


class HeadWidget(QtGui.QWidget):
    def __init__(self, project, trainer):
        super(HeadWidget, self).__init__()
        self.project = project
        self.trainer = trainer

        self.setLayout(QtGui.QVBoxLayout())
        self.label = QtGui.QLabel()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.buttons_l = QtGui.QHBoxLayout()
        self.no = QtGui.QPushButton('no (N)', self)
        self.yes = QtGui.QPushButton('yes (M)', self)
        self.dont_know = QtGui.QPushButton('dont know (P)', self)
        self.skip = QtGui.QPushButton('skip (Space)', self)
        self.quit = QtGui.QPushButton('save and quit', self)

        self._prepare_layouts()
        self._prepare_buttons()

        self.setWindowTitle('Head GT Widget')

        self.results = {}
        self.regions = None

        self.last = -1

    def set_data(self, regions):
        self.regions = regions
        self._next()

    def _next(self):
        if self.regions and len(self.regions) > 0:
            self.current = self.regions.pop()
            self._add_region(self.current)
            if len(self.regions) % BACK_UP_CONSTANT == 0:
                self.trainer.accept_results(self.results)
                self.results = {}
        else:
            self.buttons_l.addWidget(QtGui.QLabel("Every region from input already marked"))
            self.no.setDisabled(True)
            self.yes.setDisabled(True)
            self.dont_know.setDisabled(True)
            self.skip.setDisabled(True)
            self.canvas.hide()
            self.toolbar.hide()
            # self.close()

    def get_results(self):
        return self.results

    def _add_region(self, region):
        ant = pca.get_feature_vector(region, len(region.contour_without_holes()) / 2, True)
        ax = self.figure.add_subplot(111)
        ax.plot(np.append(ant[::2], ant[0]), np.append(ant[1::2], ant[1]))
        ax.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        ax.hold(False)
        self.canvas.draw()
        self.label.setText('Last id: ' + str(self.last))
        self.last = region.id()

    def _prepare_layouts(self):
        self.layout().addLayout(self.buttons_l)
        self.layout().addWidget(self.canvas)
        # self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.label)
        self.buttons_l.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)

    def _prepare_buttons(self):
        self.buttons_l.addWidget(self.no)
        self.buttons_l.addWidget(self.yes)
        self.buttons_l.addWidget(self.dont_know)
        self.buttons_l.addWidget(self.skip)
        self.buttons_l.addWidget(self.quit)
        self.connect(self.no, QtCore.SIGNAL('clicked()'), self.no_function)
        self.connect(self.yes, QtCore.SIGNAL('clicked()'), self.yes_function)
        self.connect(self.dont_know, QtCore.SIGNAL('clicked()'), self.dont_know_function)
        self.connect(self.skip, QtCore.SIGNAL('clicked()'), self._next)
        self.connect(self.quit, QtCore.SIGNAL('clicked()'), self.close)
        self.connect(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_N), self), QtCore.SIGNAL('activated()'),
                     self.no_function)
        self.connect(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_M), self), QtCore.SIGNAL('activated()'),
                     self.yes_function)
        self.connect(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_P), self), QtCore.SIGNAL('activated()'),
                     self.dont_know_function)
        self.connect(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self), QtCore.SIGNAL('activated()'),
                     self._next)

        self.yes.setFixedWidth(100)
        self.no.setFixedWidth(100)
        self.dont_know.setFixedWidth(100)
        self.skip.setFixedWidth(100)

    def no_function(self):
        self.results[self.current.id()] = False
        self._next()

    def yes_function(self):
        self.results[self.current.id()] = True
        self._next()

    def dont_know_function(self):
        if self.current.id() in self.trainer.results:
            print "deleting", self.current.id()
            del self.trainer.results[self.current.id()]
        self._next()

    def closeEvent(self, QCloseEvent):
        super(HeadWidget, self).closeEvent(QCloseEvent)
        self.trainer.accept_results(self.results)
