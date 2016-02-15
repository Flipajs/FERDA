import threading
from PyQt4 import QtGui, QtCore
import cv2, sys
import time
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication

from core.project.project import Project
from gui.gui_utils import cvimg2qtpixmap
from utils.img_manager import ImgManager

__author__ = 'Simon Mandlik'

RELATIVE_MARGIN = 0.1
NUMBER_OF_COLUMNS = 4
TEXT = "Press Space to proceed\nS for change of step\nD for change of ROI\n\n\n\n" \
       "Frame: {0}\nStep: {1}\nRoi: {2}\n\n\n\nChunk:\nId: {3}\nLength: {4}\n\n\n\n" \
       "Region:\nId: {5}\nCentroid: {6}"

class CropWidget(QtGui.QWidget):

    def __init__(self, project, roi=RELATIVE_MARGIN, start_frame=0, end_frame=200, step=1):
        super(CropWidget, self).__init__()
        self.project = project
        self.graph_manager = project.gm
        self.image_manager = ImgManager(project)

        self.margin = self.frameGeometry().height() * 2
        self.start_frame = start_frame
        if end_frame is None:
            self.end_frame = self.project.vm.total_frame_count()
        else:
            self.end_frame = end_frame
        self.step = step
        self.roi = roi

        self.h_layout = QtGui.QHBoxLayout()
        self.setLayout(self.h_layout)

        self.info_widget = QtGui.QWidget()
        self.info_widget.setLayout(QtGui.QVBoxLayout())

        self.info_label = QtGui.QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        stylesheet = "font: 30px"
        self.info_label.setStyleSheet(stylesheet)
        self.button_start = QtGui.QPushButton('Start (I)', self)
        self.button_start.clicked.connect(self.handle_start)
        self.button_start.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_I))
        self.button_stop = QtGui.QPushButton('Stop (O)', self)
        self.button_stop.clicked.connect(self.handle_stop)
        self.button_stop.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_O))
        self._running = False

        self.info_widget.layout().addWidget(self.info_label)
        self.info_widget.layout().addWidget(self.button_start)
        self.info_widget.layout().addWidget(self.button_stop)

        self.layout().addWidget(self.info_widget)

        self.img_label = QtGui.QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(self.img_label)

        self.next_frame_action = QtGui.QAction('next_frame', self)
        self.next_frame_action.triggered.connect(self.next_frame)
        self.next_frame_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space))
        self.addAction(self.next_frame_action)

        self.change_step_action = QtGui.QAction('change_step', self)
        self.change_step_action.triggered.connect(self.change_step)
        self.change_step_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_S))
        self.addAction(self.change_step_action)

        self.change_roi_action = QtGui.QAction('change_roi', self)
        self.change_roi_action.triggered.connect(self.change_roi)
        self.change_roi_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_D))
        self.addAction(self.change_roi_action)

        self.chunk_generator = self.graph_manager.chunks_in_frame_generator(self.start_frame, self.end_frame)
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
        step, none = QtGui.QInputDialog.getInt(self, 'Change of step', 'Enter new step:', value=1)
        self.step = step

    def change_roi(self):
        roi, none = QtGui.QInputDialog.getDouble(self, 'Change of ROI', 'Enter new ROI:', value=0.1)
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
        img = self.image_manager.get_crop(frame, region, width=self.margin, height=self.margin, relative_margin=self.roi)
        pixmap = cvimg2qtpixmap(img)
        return pixmap

    def generate_next_region(self, chunk):
        chunk_start = chunk.start_frame(self.graph_manager)
        i = chunk_start
        if i < self.start_frame:
            i = self.start_frame
        end = chunk.end_frame(self.graph_manager)
        while i <= self.end_frame and i <= end:
            yield self.graph_manager.region(chunk[i - chunk_start]), i
            i += self.step

if __name__ == '__main__':
    project = Project()
    project.load("/home/sheemon/FERDA/projects/Cam1_/cam1.fproj")

    app = QtGui.QApplication(sys.argv)
    widget = CropWidget(project)
    widget.showMaximized()
    app.exec_()
    cv2.waitKey(0)

