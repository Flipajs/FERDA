import logging
from PyQt4 import QtCore
from PyQt4 import QtGui

import sys

from core.graph.region_chunk import RegionChunk
from core.project.project import Project
from gui.gui_utils import cvimg2qtpixmap
from gui.segmentation.segmentation import SegmentationPicker
from utils.video_manager import get_auto_video_manager
import cPickle

__author__='simon'


class GTWidget(QtGui.QWidget):

    width = 1000
    height = 1000

    def __init__(self, project, cluster_tracklets_id, threshold=12):
        QtGui.QWidget.__init__(self)
        self.project = project
        self.threshold = threshold
        self.img_viewer = None
        self.region_generator = self._regions_gen(cluster_tracklets_id)
        self._init_gui()


#         for t_id in cluster_tracklets_id:
#             tracklet = project.chm[t_id]
#             for region in RegionChunk(tracklet, project.gm, project.rm).regions_gen():
#
#                 app = QtGui.QApplication(sys.argv)
#                 ex = SegmentationPicker(self.im_manager.get_crop(region.frame(), region,  width=self.width, height=self.height, default_color=(255,255,255,0))
# )
#                 ex.show()
#                 ex.move(-500, -500)
#                 ex.showMaximized()
#                 ex.setFocus()
#
#                 app.exec_()
#         sys.exit()

        # self._next()

    def _next_region(self):
        pass

    def _next(self):
        self.img_viewer.set_next(self.region_generator.next())
        self.roi_tickbox.setChecked(True)

    def _toggle_roi(self):
        if self.roi_tickbox.isChecked():
            self.img_viewer.show_roi()
        else:
            self.img_viewer.show_img()

    def _regions_gen(self, tracklets_id):
        for t_id in tracklets_id:
            tracklet = self.project.chm[t_id]
            for region in RegionChunk(tracklet, self.project.gm, self.project.rm).regions_gen():
                yield region


    def _set_threshold(self, value):
        self.threshold = value

    def _init_gui(self):
        self.showMaximized()

        self.layout = QtGui.QHBoxLayout()
        self.left_part = QtGui.QWidget()
        self.left_part.setLayout(QtGui.QVBoxLayout())
        self.left_part.layout().setAlignment(QtCore.Qt.AlignCenter)

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        # self.slider.setGeometry(30, 40, 50, 30)
        self.slider.setRange(0, 100)
        self.slider.setTickInterval(10)
        self.slider.setValue(self.threshold)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider.valueChanged[int].connect(self._set_threshold)

        self.roi_tickbox = QtGui.QCheckBox("Roi")
        self.roi_tickbox.clicked.connect(self._toggle_roi)

        self.next_button = QtGui.QPushButton('Next')
        self.next_button.clicked.connect(self._next)

        self.next_region_button = QtGui.QPushButton('Next Region')
        self.next_region_button.clicked.connect(self._next_region)


        self.left_part.layout().addWidget(self.slider)
        self.left_part.layout().addWidget(self.roi_tickbox)
        self.left_part.layout().addWidget(self.next_button)
        self.left_part.layout().addWidget(self.next_region_button)

        self.img_viewer = ImgViewer(self.project.img_manager)

        self.layout.addWidget(self.left_part)
        self.layout.addWidget(self.img_viewer)

        self.setLayout(self.layout)


class ImgViewer(QtGui.QLabel):

    img = None
    img_roi = None

    def __init__(self, img_manager):
        QtGui.QLabel.__init__(self)
        self.img_manager = img_manager
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

    def show_img(self):
        self.setPixmap(cvimg2qtpixmap(self.img))

    def show_roi(self):
        self.setPixmap(cvimg2qtpixmap(self.img_roi))

    def set_next(self, region):
        print self.height(), self.width()
        self.img = self.img_manager.get_crop(region.frame(), region, width=self.width(), height=self.height(), default_color=(255,255,255,0))
        self.img_roi = self.img_manager.get_crop(region.frame(), region)
        self.setPixmap(cvimg2qtpixmap(self.img_roi))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")
    chunks = project.gm.chunk_list()

    chunks_with_clusters = [6, 10, 12, 13, 17, 18, 26, 28, 29, 32, 37, 39, 40, 41, 43, 47, 51, 54, 57, 58, 60, 61, 65,
                            67, 69, 73, 75, 78, 81, 84, 87, 90, 93, 94, 96, 99, 102, 105]
    chunks_with_clusters = map(lambda x: chunks[x], chunks_with_clusters)

    app = QtGui.QApplication(sys.argv)

    gt = GTWidget(project, chunks_with_clusters)
    gt.show()

    app.exec_()
