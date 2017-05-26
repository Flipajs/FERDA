__author__ = 'flipajs'

from PyQt4 import QtGui, QtCore
from utils.video_manager import get_auto_video_manager
from gui.img_controls.gui_utils import cvimg2qtpixmap
from skimage.transform import resize
import numpy as np
from gui.img_controls import markers
from core.settings import Settings as S_


class ChunksOnFrame(QtGui.QWidget):
    def __init__(self, project, plot_w, start_t, end_t, close_callback):
        super(ChunksOnFrame, self).__init__()

        self.close_callback = close_callback
        self.plot_w = plot_w
        self.setLayout(QtGui.QHBoxLayout())
        self.setFixedWidth(450)
        self.project = project
        self.slider = QtGui.QSlider()
        self.slider.setMinimum(start_t)
        self.slider.setMaximum(end_t)
        self.slider.valueChanged.connect(self.frame_slider_changed)

        self.layout().addWidget(self.slider)

        self.view = QtGui.QGraphicsView()
        self.scene = QtGui.QGraphicsScene()
        self.view.setScene(self.scene)

        self.vid = get_auto_video_manager(self.project)
        self.frame_it = None

        self.next_action = QtGui.QAction('next', self)
        self.next_action.triggered.connect(self.next_frame)
        self.next_action.setShortcut(S_.controls.video_next)
        self.addAction(self.next_action)

        self.prev_action = QtGui.QAction('prev', self)
        self.prev_action.triggered.connect(self.prev_frame)
        self.prev_action.setShortcut(S_.controls.video_prev)
        self.addAction(self.prev_action)

        self.marker_its = []

        self.vbox = QtGui.QVBoxLayout()
        self.layout().addLayout(self.vbox)

        self.close_b = QtGui.QPushButton('close')
        self.close_b.clicked.connect(self.close_callback)
        self.vbox.addWidget(self.close_b)
        self.vbox.addWidget(self.view)

    def next_frame(self):
        f = self.slider.value()
        self.slider.setValue(f+1)

    def prev_frame(self):
        f = self.slider.value()
        self.slider.setValue(f-1)

    def frame_slider_changed(self):
        f = self.slider.value()
        self.plot_w.draw_plane(f)

        im = self.vid.get_frame(f)
        # im = self.vid.seek_frame(f)

        w = 420
        scale = (w / float(im.shape[1]))
        h = int(im.shape[0] * scale)

        for it in self.marker_its:
            it.setPos(-19, -10)
            self.scene.removeItem(it)

        self.marker_its = []

        im = np.asarray(resize(im, (h, w)) * 255, dtype=np.uint8)
        for y, x, c in self.plot_w.intersection_positions:
            r = int(c[0]*255)
            g = int(c[1]*255)
            b = int(c[2]*255)
            self.marker_its.append(markers.CenterMarker(0, 0, 3, QtGui.QColor(r, g, b), 0, None))
            self.marker_its[-1].setZValue(0.5)
            self.marker_its[-1].setPos(x*scale, y*scale)
            self.scene.addItem(self.marker_its[-1])

        if self.frame_it:
            self.scene.removeItem(self.frame_it)

        pm = cvimg2qtpixmap(im)
        self.frame_it = self.scene.addPixmap(pm)
