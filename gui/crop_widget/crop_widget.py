from PyQt4 import QtGui, QtCore

import cv2
import sys
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication

from core.project.project import Project
from gui.gui_utils import cvimg2qtpixmap
from utils.img_manager import ImgManager

__author__ = 'Simon Mandlik'

RELATIVE_MARGIN = 0.1
NUMBER_OF_COLUMNS = 4
TEXT = "Press Space to proceed\nShift + Space for one step backwards\nS for change of step\nD for change of ROI\n\n\n\n" \
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
        stylesheet = "border-style: outset; border-width: 2px; border-radius: 10px; border-color: black; font: bold 14px; min-width: 10em; padding: 6px; font: 30px"
        self.info_label.setStyleSheet(stylesheet)
        self.button_start = QtGui.QPushButton('Start (I)', self)
        self.button_start.clicked.connect(self.handle_start)
        self.button_start.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_I))
        self.button_stop = QtGui.QPushButton('Stop (O)', self)
        self.button_stop.clicked.connect(self.handle_stop)
        self.button_stop.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_O))
        self._running = False

        self.chunk_combobox = QtGui.QComboBox()
        self.chunk_combobox.setEditable(True)
        self.chunk_combobox.lineEdit().setReadOnly(True)
        self.chunk_combobox.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.chunk_combobox.activated.connect(self.combobox_signal)

        self.info_widget.layout().addWidget(self.chunk_combobox)
        self.info_widget.layout().addWidget(self.info_label)
        self.info_widget.layout().addWidget(self.button_start)
        self.info_widget.layout().addWidget(self.button_stop)

        self.layout().addWidget(self.info_widget)

        self.img_label = QtGui.QLabel()
        stylesheet += "; background-color: black"
        self.img_label.setStyleSheet(stylesheet)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(self.img_label)

        self.reverse_frame_action = QtGui.QAction('reverse_frame', self)
        self.reverse_frame_action.triggered.connect(self.reverse)
        self.reverse_frame_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_Space))
        self.addAction(self.reverse_frame_action)

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

        self.chunks = self.graph_manager.chunks_in_frame(self.start_frame, self.end_frame)
        self.number_of_chunks = len(self.chunks)
        self.actual_chunk_position = 0
        self.actual_chunk = None
        self.chunk_id = None
        self.chunk_length = 0
        self.region_generator = None

        # self.chunk_combobox.addItem("Select chunk by its id")
        self.chunk_combobox.addItems([str(x.id()) for x in self.chunks])

        self.change_chunk()
        self.next_frame()

    def combobox_signal(self, index):
        self.actual_chunk_position = index - 1
        self.actual_chunk = None
        self.region_generator = None
        self.refresh()

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
        self.refresh()

    def refresh(self):
        old_step = self.step
        self.step = 0
        self.next_frame()
        self.step = old_step

    def reverse(self):
        self.step = -self.step;
        self.next_frame()
        self.step = -self.step

    def next_frame(self):
        if self.actual_chunk is None:
            if self.step > 0 and self.actual_chunk_position + 1 < self.number_of_chunks:
                self.actual_chunk_position += 1
            elif self.step < 0 < self.actual_chunk_position:
                self.actual_chunk_position -= 1
            if self.actual_chunk_position in range(0, self.number_of_chunks):
                self.change_chunk()
            else:
                return
        if self.region_generator is None:
            self.region_generator = self.generate_next_region(self.actual_chunk)
        try:
            region, frame = next(self.region_generator)
            self.img_label.setPixmap(self.prepare_pixmap(region, frame))
            self.info_label.setText(
                TEXT.format(frame, self.step, self.roi, self.chunk_id, self.chunk_length, region.id(),
                            region.centroid()))
            QApplication.processEvents()
        except StopIteration:
            self.region_generator = None
            self.actual_chunk = None
            self.next_frame()

    def change_chunk(self):
        self.actual_chunk = self.chunks[self.actual_chunk_position]
        self.chunk_combobox.setCurrentIndex(self.actual_chunk_position + 1)
        self.chunk_id = self.actual_chunk.id()
        self.chunk_length = self.actual_chunk.length()

    def prepare_pixmap(self, region, frame):
        img = self.image_manager.get_crop(frame, region, width=self.margin, height=self.margin,
                                          relative_margin=self.roi)
        pixmap = cvimg2qtpixmap(img)
        return pixmap

    def generate_next_region(self, chunk):
        chunk_start = chunk.start_frame(self.graph_manager)
        chunk_end = chunk.end_frame(self.graph_manager)
        start = chunk_start if chunk_start > self.start_frame else self.start_frame
        end = chunk_end if chunk_end < self.end_frame else self.end_frame
        if self.step >= 0:
            if self.actual_chunk_position + 1 < self.number_of_chunks:
                i = start
            else:
                i = end
        if self.step < 0:
            if self.actual_chunk_position > 0:
                i = end
            else:
                i = start

        while i in range(start, end + 1):
            yield self.graph_manager.region(chunk[i - chunk_start]), i
            i += self.step


if __name__ == '__main__':
    project = Project()
    project.load("/home/sheemon/FERDA/projects/Cam1_/cam1.fproj")

    app = QtGui.QApplication(sys.argv)
    widget = CropWidget(project, start_frame=0, end_frame=10)
    widget.showMaximized()
    app.exec_()
    cv2.waitKey(0)
