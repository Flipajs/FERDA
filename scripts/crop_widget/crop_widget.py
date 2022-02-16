import threading
from PyQt6 import QtCore, QtGui, QtWidgets
import cv2, sys
import time
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
import numpy as np

from core.project.project import Project
from gui.gui_utils import cvimg2qtpixmap
from utils.img_manager import ImgManager

__author__ = 'Simon Mandlik'

RELATIVE_MARGIN = 0.1
NUMBER_OF_COLUMNS = 4
TEXT = "Press Space to proceed\nS for change of step\nD for change of ROI\n\n\n\n" \
       "Frame: {0}\nStep: {1}\nRoi: {2}\n\n\n\nChunk:\nId: {3}\nLength: {4}\n\n\n\n" \
       "Region:\nId: {5}\nCentroid: {6}"

class CropWidget(QtWidgets.QWidget):

    def __init__(self, project, ft=None, roi=RELATIVE_MARGIN, start_frame=0, end_frame=200, step=1):
        super(CropWidget, self).__init__()
        self.project = project
        self.graph_manager = project.gm
        self.image_manager = ImgManager(project)

        self.ft = ft

        self.margin = self.frameGeometry().height() * 2
        self.start_frame = start_frame
        if end_frame is None:
            self.end_frame = self.project.vm.total_frame_count()
        else:
            self.end_frame = end_frame
        self.step = step
        self.roi = roi

        self.h_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.h_layout)

        self.info_widget = QtWidgets.QWidget()
        self.info_widget.setLayout(QtWidgets.QVBoxLayout())

        self.info_label = QtWidgets.QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stylesheet = "font: 30px"
        self.info_label.setStyleSheet(stylesheet)
        self.button_start = QtWidgets.QPushButton('Start (I)', self)
        self.button_start.clicked.connect(self.handle_start)
        self.button_start.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_I))
        self.button_stop = QtWidgets.QPushButton('Stop (O)', self)
        self.button_stop.clicked.connect(self.handle_stop)
        self.button_stop.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_O))
        self._running = False

        self.info_widget.layout().addWidget(self.info_label)
        self.info_widget.layout().addWidget(self.button_start)
        self.info_widget.layout().addWidget(self.button_stop)

        self.layout().addWidget(self.info_widget)

        self.img_label = QtWidgets.QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout().addWidget(self.img_label)

        self.next_frame_action = QtGui.QAction('next_frame', self)
        self.next_frame_action.triggered.connect(self.next_frame)
        self.next_frame_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space))
        self.addAction(self.next_frame_action)

        self.change_step_action = QtGui.QAction('change_step', self)
        self.change_step_action.triggered.connect(self.change_step)
        self.change_step_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_S))
        self.addAction(self.change_step_action)

        self.change_roi_action = QtGui.QAction('change_roi', self)
        self.change_roi_action.triggered.connect(self.change_roi)
        self.change_roi_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_D))
        self.addAction(self.change_roi_action)

        self.chunk_generator = iter(self.project.chm.get_tracklets_in_interval(self.start_frame, self.end_frame))
        self.chunk = None
        self.chunk_id = None
        self.chunk_length = 0
        self.region_generator = None

        self.next_frame()

    def handle_start(self):
        self.button_start.setDisabled(True)
        self._running = True
        old_step = self.step
        self.step = 1
        while self._running:
            self.next_frame()
        self.step = old_step
        self.button_start.setDisabled(False)

    def handle_stop(self):
        self._running = False

    def change_step(self):
        step, none = QtWidgets.QInputDialog.getInt(self, 'Change of step', 'Enter new step:', value=1)
        self.step = step

    def change_roi(self):
        roi, none = QtWidgets.QInputDialog.getDouble(self, 'Change of ROI', 'Enter new ROI:', value=0.1)
        self.roi = roi

    def next_frame(self):
        if self.chunk is None:
            try:
                self.chunk = next(self.chunk_generator)
                self.chunk_id = self.chunk.id()
                self.chunk_length = self.chunk.length()
            except StopIteration:
                return
        if self.region_generator is None:
            self.region_generator = self.generate_next_region(self.chunk)
        try:
            region, frame = next(self.region_generator)
            self.img_label.setPixmap(self.prepare_pixmap(region, frame))
            self.info_label.setText(TEXT.format(frame, self.step, self.roi, self.chunk_id, self.chunk_length, region.id(), region.centroid()))
            QApplication.processEvents()
        except StopIteration:
            self.region_generator = None
            self.chunk = None
            self.next_frame()

    def prepare_pixmap(self, region, frame):
        from utils.img import get_safe_selection
        from skimage.transform import rescale

        img = self.image_manager.get_whole_img(frame)
        img_ = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        border = 5
        img_ = cv2.copyMakeBorder(img_, border, border, border, border, cv2.BORDER_REPLICATE )
        ft.getCharSegmentations(img_, '', 'base')


        keypoints = ft.getLastDetectionKeypoints()
        strokes = []

        sizes_ = [1, 3, 6]

        # TODO: solve scale
        scales = ft.getImageScales()
        scs_ = []
        for kp, i in zip(keypoints, list(range(len(keypoints)))):
            s_ = ft.getKeypointStrokes(i) * (1.0 / scales[kp[2]])
            scs_.append(kp[2])
            strokes.append(s_)


        scale_factor = 2.5
        img = np.asarray(rescale(img, scale_factor) * 255, dtype=np.uint8)

        for i in [2, 1, 0]:
            for s_, sc_ in zip(strokes, scs_):
                if sc_ != i:
                    continue

                c = tuple(numpy.asarray(numpy.random.rand(3,) * 255, dtype=numpy.uint8))
                c = (int(c[0]), int(c[1]), int(c[2]))

                for pt in s_:
                    cv2.circle(img, (int(scale_factor * (pt[0] - border)), int(scale_factor * (pt[1]-border))), sizes_[i], c, -1)

        border = 700
        img = get_safe_selection(img, scale_factor*region.centroid()[0] - border/2, scale_factor* region.centroid()[1] - border/2, border, border)

        pixmap = cvimg2qtpixmap(img)
        return pixmap

    def generate_next_region(self, chunk):
        chunk_start = chunk.start_frame()
        i = chunk_start
        if i < self.start_frame:
            i = self.start_frame
        end = chunk.end_frame()
        while i <= self.end_frame and i <= end:
            yield self.graph_manager.region(chunk[i - chunk_start]), i
            i += self.step

if __name__ == '__main__':
    import sys
    sys.path.append('/Users/flipajs/Documents/dev/fastext/toolbox')
    from ft import FASTex
    import math
    import cv2
    from utils.video_manager import get_auto_video_manager
    from core.project.project import Project

    import matplotlib.pyplot as plt
    import numpy

    project = Project()
    project.load("/Users/flipajs/Documents/wd/GT/Cam1/cam1.fproj")

    vm = get_auto_video_manager(project)

    scaleFactor = 1.4
    nlevels = 3
    edgeThreshold = 13
    keypointTypes = 2
    kMin = 9
    kMax = 11
    erode = 0
    segmentGrad = 0
    minCompSize = 4
    process_color = 0
    segmDeltaInt = 1
    min_tupple_top_bottom_angle = math.pi / 2
    maxSpaceHeightRatio = -1
    createKeypointSegmenter = 1

    params_dict = {
        'scaleFactor': scaleFactor,
        'nlevels': nlevels,
        'edgeThreshold': edgeThreshold,
        'keypointTypes': keypointTypes,
        'kMin': kMin,
        'kMax': kMax,
        'erode': erode,
        'segmentGrad': segmentGrad,
        'minCompSize': minCompSize,
        'process_color': process_color,
        'segmDeltaInt': segmDeltaInt,
        'min_tupple_top_bottom_angle': min_tupple_top_bottom_angle,
        'maxSpaceHeightRatio': maxSpaceHeightRatio,
        'createKeypointSegmenter': createKeypointSegmenter
    }

    process_color = 0

    imgName = '/Users/flipajs/Pictures/vlcsnap-2016-02-12-13h43m29s676.png'
    border = 5

    ft = FASTex(edgeThreshold=edgeThreshold,
                createKeypointSegmenter=createKeypointSegmenter,
                nlevels=nlevels,
                minCompSize=minCompSize,
                keypointTypes=keypointTypes)

    app = QtWidgets.QApplication(sys.argv)
    widget = CropWidget(project, ft)
    widget.showMaximized()
    app.exec()
    cv2.waitKey(0)

