import logging
from PyQt4 import QtCore
from PyQt4 import QtGui

import sys

from core.graph.region_chunk import RegionChunk
from core.project.project import Project
from gui.arena.my_view import MyView
import numpy as np
from gui.gui_utils import cvimg2qtpixmap
from gui.segmentation.my_scene import MyScene
from gui.segmentation.painter import mask2qimage
from gui.segmentation.segmentation import SegmentationPicker
from utils.video_manager import get_auto_video_manager
import cPickle

__author__ = 'simon'


class GTWidget(QtGui.QWidget):
    width = 1000
    height = 1000

    def __init__(self, project, cluster_tracklets_id, threshold=12):
        QtGui.QWidget.__init__(self)
        self.project = project
        self.threshold = threshold
        self.img_viewer = ImgPainter(self.project.img_manager, threshold)
        self.region_generator = self.regions_gen(cluster_tracklets_id)
        self.init_gui()

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

    def next_ant(self):
        pass

    def next_region(self):
        self.img_viewer.set_next(self.region_generator.next())
        self.roi_tickbox.setChecked(True)

    def toggle_roi(self):
        if self.roi_tickbox.isChecked():
            self.img_viewer.show_roi()
        else:
            self.img_viewer.show_img()

    def regions_gen(self, tracklets_id):
        for t_id in tracklets_id:
            tracklet = self.project.chm[t_id]
            for region in RegionChunk(tracklet, self.project.gm, self.project.rm).regions_gen():
                yield region

    def set_threshold(self, value):
        self.threshold = value
        self.img_viewer.set_threshold(value)

    def init_gui(self):
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
        self.slider.valueChanged[int].connect(self.set_threshold)
        self.slider.sliderReleased.connect(self.img_viewer.update_selection)

        self.roi_tickbox = QtGui.QCheckBox("Roi")
        self.roi_tickbox.clicked.connect(self.toggle_roi)

        self.buttons = QtGui.QWidget()
        self.buttons.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        self.buttons.setLayout(QtGui.QVBoxLayout())
        self.buttons.layout().setAlignment(QtCore.Qt.AlignBottom)
        self.next_button = QtGui.QPushButton('Next')
        self.next_button.clicked.connect(self.next_region)

        self.next_region_button = QtGui.QPushButton('Next Ant')
        self.next_region_button.clicked.connect(self.next_ant)

        self.reset_button = QtGui.QPushButton('Reset')
        self.reset_button.clicked.connect(self.img_viewer.reset)

        self.buttons.layout().addWidget(self.next_button)
        self.buttons.layout().addWidget(self.next_region_button)
        self.buttons.layout().addWidget(self.reset_button)

        self.help = QtGui.QLabel("Ctrl + Scroll to zoom, Shift + click for multi-selection")

        self.left_part.layout().addWidget(self.slider)
        self.left_part.layout().addWidget(self.roi_tickbox)
        self.left_part.layout().addWidget(self.buttons)
        self.left_part.layout().addWidget(self.help)

        self.layout.addWidget(self.left_part)
        self.layout.addWidget(self.img_viewer)

        self.setLayout(self.layout)


class ImgPainter(MyView):
    img = None
    img_roi = None

    def __init__(self, img_manager, threshold=12):
        MyView.__init__(self)
        # self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.img_manager = img_manager
        self.scene = MyScene()
        self.setScene(self.scene)

        self.threshold = threshold
        self.last_x = 0
        self.last_y = 0
        self.x = []
        self.y = []

        self.visited = np.array((0,0))
        self.selected = np.array((0,0))
        self.bitmask_pixmap = np.array((0,0))

    def set_threshold(self, threshold):
        self.threshold = threshold
        self.visited.fill(False)
        self.selected.fill(False)
        self.update_selection()
        self.draw()

    def show_img(self):
        self.scene.addPixmap(cvimg2qtpixmap(self.img))

    def show_roi(self):
        self.scene.addPixmap(cvimg2qtpixmap(self.img_roi))

    def set_next(self, region):
        self.img = self.img_manager.get_crop(region.frame(), region, width=self.width(), height=self.height(),
                                             default_color=(255, 255, 255, 0))
        self.img_roi = self.img_manager.get_crop(region.frame(), region, width=self.width(), height=self.height())
        self.visited = np.zeros(self.img.shape[:2], dtype=bool)
        self.selected = np.zeros(self.img.shape[:2], dtype=bool)
        self.scene.addPixmap(cvimg2qtpixmap(self.img_roi))

    def mousePressEvent(self, e):
        super(MyView, self).mousePressEvent(e)
        point = self.mapToScene(e.pos())
        if self.scene.itemsBoundingRect().contains(point):
            self.add_point(point)

    def update_selection(self):
        for x, y in zip(self.x, self.y):
            self.floodfill(self.img[x, y], x, y)

        # np.set_printoptions(threshold=np.inf)
        # print self.selected.ndim
        # print self.selected.shape
        # print self.selected[range(self.last_x-3, self.last_x+3), :]
        import matplotlib.pyplot as plt
        fig = plt.figure()
        # ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        # ax4 = fig.add_subplot(224)

        # ax1.spy(self.selected, markersize=5)
        # ax2.spy(self.selected, precision=0.1, markersize=5)

        ax3.spy(self.selected)
        # ax4.spy(self.selected, precision=0.1)

        plt.show()


    def floodfill(self, color, x, y):
        stack = [self.find_line_segment(color, x, y)]
        while len(stack) > 0:
            segment = stack.pop()
            x = segment[0]
            for y in range(segment[1], segment[2] + 1):
                if x - 1 >= 0:
                    if self.is_similar(color, x - 1, y):
                        stack.append(self.find_line_segment(color, x - 1, y))
                    else:
                        self.visited[x - 1, y] = True
                if x + 1 < self.img.shape[0]:
                    if self.is_similar(color, x + 1, y):
                        stack.append(self.find_line_segment(color, x + 1, y))
                    else:
                        self.visited[x + 1, y] = True

    def find_line_segment(self, color, x, y):
        y1 = y2 = y
        self.visited[x, y] = True
        self.selected[x, y] = True
        while y1 - 1 >= 0:
            self.visited[x, y1] = True
            if self.is_similar(color, x, y1 - 1):
                self.selected[x, y1] = True
                y1 -= 1
            else:
                break
        while y2 + 1 < self.img.shape[1]:
            self.visited[x, y2] = True
            if self.is_similar(color, x, y2 + 1):
                self.selected[x, y2] = True
                y2 += 1
            else:
                break
        return x, y1, y2

    def is_similar(self, color, x, y):
        return not self.visited[x, y] and self.are_colors_similar(color, self.img[x, y])

    def are_colors_similar(self, color1, color2):
        for c1, c2 in zip(color1, color2):
            if abs(int(c1) - int(c2)) > self.threshold:
                return False
        return True

    def reset(self):
        self.x = []
        self.y = []
        self.visited.fill(False)
        self.selected.fill(False)
        self.draw()

    def add_point(self, point):
        if type(point) == QtCore.QPointF:
            point = point.toPoint()
        self.last_x = point.x()
        self.last_y = point.y()
        self.x.append(point.x())
        self.y.append(point.y())

        print self.last_x, self.last_y
        # TODO uncomment!!!
        # self.update_selection()

    def draw_bitmask(self, mask, color=(20, 20, 20)):
        qimg = mask2qimage(mask, color)
        pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(qimg))
        pixmap.setZValue(20)
        return pixmap


    def draw(self):
        # qimg = mask2qimage(self.selected, (10, 10, 10, 10))
        #
        # self.colors[name][2] = self.scene.addPixmap(QtGui.QPixmap.fromImage(qimg))
        # self.colors[name][2].setZValue(20)
        if self.bitmask_pixmap is not None:
            self.scene.removeItem(self.bitmask_pixmap)
        self.bitmask_pixmap = self.draw_bitmask(self.selected)

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
