__author__ = 'flipajs'

from PyQt4 import QtGui, QtCore
from utils.video_manager import get_auto_video_manager
from gui.img_controls.utils import cvimg2qtpixmap
from skimage.transform import resize
import numpy as np


class ChunksOnFrame(QtGui.QWidget):
    def __init__(self, project, plot_w, start_t, end_t):
        super(ChunksOnFrame, self).__init__()

        self.plot_w = plot_w
        self.setLayout(QtGui.QHBoxLayout())
        self.setFixedWidth(320)
        self.project = project
        self.slider = QtGui.QSlider()
        self.slider.setMinimum(start_t)
        self.slider.setMaximum(end_t)
        self.slider.valueChanged.connect(self.frame_slider_changed)

        self.layout().addWidget(self.slider)

        self.view = QtGui.QGraphicsView()
        self.scene = QtGui.QGraphicsScene()
        self.view.setScene(self.scene)

        self.vid = get_auto_video_manager(self.project.video_paths)
        self.frame_it = None

        self.layout().addWidget(self.view)

    def frame_slider_changed(self):
        f = self.slider.value()
        self.plot_w.draw_plane(f)

        self.vid.get_frame(f, auto=True)
        im = self.vid.seek_frame(f)

        w = 250
        h = int(im.shape[0] * (250 / float(im.shape[1])))

        im = np.asarray(resize(im, (h, w)) * 255, dtype=np.uint8)

        if self.frame_it:
            self.scene.removeItem(self.frame_it)

        pm = cvimg2qtpixmap(im)
        self.frame_it = self.scene.addPixmap(pm)
