import logging
from PyQt4 import QtCore
from PyQt4 import QtGui

import sys

from core.graph.region_chunk import RegionChunk
from core.project.project import Project
from gui.arena.my_view import MyView
from gui.gui_utils import cvimg2qtpixmap
from gui.segmentation.my_scene import MyScene
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
        self.img_viewer.update_selection()

    def _init_gui(self):
        self.showMaximized()

        self.layout = QtGui.QHBoxLayout()
        self.left_part = QtGui.QWidget()
        self.left_part.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        self.left_part.setLayout(QtGui.QVBoxLayout())
        self.left_part.layout().setAlignment(QtCore.Qt.AlignTop)

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, 100)
        self.slider.setTickInterval(10)
        self.slider.setValue(self.threshold)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider.valueChanged[int].connect(self._set_threshold)

        self.roi_tickbox = QtGui.QCheckBox("Roi")
        self.roi_tickbox.clicked.connect(self._toggle_roi)


        self.buttons = QtGui.QWidget()
        self.buttons.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        self.buttons.setLayout(QtGui.QVBoxLayout())
        self.buttons.layout().setAlignment(QtCore.Qt.AlignBottom)
        self.next_button = QtGui.QPushButton('Next')
        self.next_button.clicked.connect(self._next)

        self.next_region_button = QtGui.QPushButton('Next Region')
        self.next_region_button.clicked.connect(self._next_region)

        self.buttons.layout().addWidget(self.next_button)
        self.buttons.layout().addWidget(self.next_region_button)

        self.left_part.layout().addWidget(self.slider)
        self.left_part.layout().addWidget(self.roi_tickbox)
        self.left_part.layout().addWidget(self.buttons)

        self.img_viewer = ImgViewer(self.project.img_manager, self.slider)

        self.layout.addWidget(self.left_part)
        self.layout.addWidget(self.img_viewer)

        self.setLayout(self.layout)


class ImgViewer(MyView):

    img = None
    img_roi = None

    def __init__(self, img_manager, slider):
        MyView.__init__(self)
        # self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.img_manager = img_manager
        self.scene = MyScene(update_callback_release=self.mouse_release)
        self.setScene(self.scene)
        self.slider = slider

    def show_img(self):
        self.scene.addPixmap(cvimg2qtpixmap(self.img))

    def show_roi(self):
        self.scene.addPixmap(cvimg2qtpixmap(self.img_roi))

    def set_next(self, region):
        print self.height(), self.width()
        # self.img = self.img_manager.get_crop(region.frame(), region, width=self.width(), height=self.height(), default_color=(255,255,255,0))
        # self.img_roi = self.img_manager.get_crop(region.frame(), region, width=self.width(), height=self.height())
        self.img = self.img_manager.get_crop(region.frame(), region, width=1000, height=1000,
                                             default_color=(255, 255, 255, 0))
        self.img_roi = self.img_manager.get_crop(region.frame(), region, width=1000, height=1000)
        self.scene.addPixmap(cvimg2qtpixmap(self.img_roi))

    def mouse_release(self, event):
        point = self.mapToScene(event.pos())
        if self.is_in_scene(point):
            self.draw(point)

    def update_selection(self):
        threshold = self.slider.getValue()
        print "Updating"
        print threshold
        print self.x, self.y


    def draw(self, point):
        if type(point) == QtCore.QPointF:
            point = point.toPoint()

        self.x = point.x()
        self.y = point.y()

        # self.colors[self.color_name][0][fromy: toy, fromx: tox] = self.eraser
        self.draw_mask(self.color_name)

    # def draw_mask(self, name):
    #
    #     qimg = mask2qimage(self.colors[name][0], self.colors[name][1])
    #
    #     # add pixmap to scene and move it to the foreground
    #     # delete old pixmap
    #     self.scene.removeItem(self.colors[name][2])
    #     self.colors[name][2] = self.scene.addPixmap(QtGui.QPixmap.fromImage(qimg))
    #     self.colors[name][2].setZValue(20)


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
