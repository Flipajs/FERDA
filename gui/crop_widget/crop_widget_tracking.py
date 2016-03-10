import threading
from PyQt4 import QtGui, QtCore

import cv2
import sys
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication, QLabel

from core.project.project import Project
from gui.gui_utils import cvimg2qtpixmap
from utils.img_manager import ImgManager

__author__ = 'Simon Mandlik'

RELATIVE_MARGIN = 0.1
NUMBER_OF_COLUMNS = 4
PICTURE_SIZE = 70
INSTRUCTIONS = "Press Space to proceed\nShift + Space for one step backwards\nS for change of step\nD for change of ROI"
TEXT = "Frame: {0}\nStep: {1}\nRoi: {2}\n\n\n\nChunk:\nId: {3}\nLength: {4}\n\n\n\n" \
       "Region:\nId: {5}\nCentroid:\n{6}"
LABEL_STYLESHEET = "border-style: outset; border-width: 2px; border-radius: 10px;" \
                   " border-color: black; min-width: 10em; padding: 6px"
PICTURE_STYLESHEET = LABEL_STYLESHEET + "; background-color: black"
CHUNK_DETAIL_STYLESHEET = "background-color: black"


class CropWidgetTracker(QtGui.QWidget):
    def __init__(self, project, roi=RELATIVE_MARGIN, start_frame=0, end_frame=200, step=1):
        super(CropWidgetTracker, self).__init__()
        self.project = project
        self.graph_manager = project.gm
        self.image_manager = ImgManager(project)

        self.size_of_pix = self.frameGeometry().height() * 2
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
        self.picture_widget = QtGui.QWidget()
        self.picture_widget.setLayout(QtGui.QVBoxLayout())

        self.info_label = QtGui.QLabel()
        self.instruction_label = QtGui.QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet(LABEL_STYLESHEET)
        self.instruction_label.setText(INSTRUCTIONS)

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
        self.info_widget.layout().addWidget(self.instruction_label)
        self.info_widget.layout().addWidget(self.info_label)
        self.info_widget.layout().addWidget(self.button_start)
        self.info_widget.layout().addWidget(self.button_stop)

        self.img_label = QtGui.QLabel()
        self.img_label.setStyleSheet(PICTURE_STYLESHEET)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.picture_widget.layout().addWidget(self.img_label)

        self.chunk_detail_widget = QtGui.QWidget()
        self.chunk_detail_widget.setLayout(QtGui.QHBoxLayout())
        self.chunk_detail_widget.layout().setContentsMargins(0,0,0,0)
        self.chunk_detail_scroll = QtGui.QScrollArea()
        self.chunk_detail_scroll.setWidgetResizable(True)
        self.chunk_detail_scroll.setFixedHeight(1.5 * PICTURE_SIZE)
        self.chunk_detail_scroll.setWidget(self.chunk_detail_widget)
        self.picture_widget.layout().addWidget(self.chunk_detail_scroll)

        self.layout().addWidget(self.picture_widget)
        self.layout().addWidget(self.info_widget)

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

        self.chunks = self.graph_manager.chunks_in_frame_range(self.start_frame, self.end_frame)
        self.number_of_chunks = len(self.chunks)
        self.actual_chunk_position = -1
        self.actual_chunk = None
        self.chunk_id = None
        self.chunk_length = 0
        self.region_generator = None
        self.chunk_labels = None

        self.chunk_combobox.addItem("Select chunk by its id")
        self.chunk_combobox.addItems([str(x.id()) for x in self.chunks])

        self.showMaximized()
        self.next_frame()

    def combobox_signal(self, index):
        self.actual_chunk_position = index - 1
        self.actual_chunk = None
        self.region_generator = None
        self.actual_chunk = None
        self.step = abs(self.step)
        self.next_frame()

    def handle_start(self):
        self.button_start.setDisabled(True)
        self._running = True
        old_step = self.step
        self.step = 1 if self.step > 0 else -1
        while self._running:
            self.next_frame()
        self.step = old_step
        self.button_start.setDisabled(False)

    def handle_stop(self):
        self._running = False

    def change_step(self):
        step, none = QtGui.QInputDialog.getInt(self, 'Change of step', 'Enter new step:', value=self.step)
        self.step = step

    def change_roi(self):
        roi, none = QtGui.QInputDialog.getDouble(self, 'Change of ROI', 'Enter new ROI:', value=self.roi)
        self.roi = roi

    def reverse(self):
        self.step = -self.step
        self.next_frame()
        self.step = -self.step

    def next_frame(self):
        QApplication.processEvents()
        if self.actual_chunk is None:
            if self.step > 0 and self.actual_chunk_position < self.number_of_chunks:
                self.actual_chunk_position += 1
            elif self.step < 0 <= self.actual_chunk_position:
                self.actual_chunk_position -= 1
        if self.region_generator is None:
            if self.actual_chunk_position in range(0, self.number_of_chunks):
                self.change_chunk()
            else:
                self._running = False
                return
            self.region_generator = self.generate_next_region(self.actual_chunk, mark=True)
        try:
            region, frame = next(self.region_generator)
            self.img_label.setPixmap(self.prepare_pixmap(region, frame, self.size_of_pix - 2 * PICTURE_SIZE, self.roi))
            self.info_label.setText(
                TEXT.format(frame, self.step, self.roi, self.chunk_id, self.chunk_length, region.id(),
                            region.centroid()))
        except StopIteration:
            self.region_generator = None
            self.actual_chunk = None
            self.next_frame()

    def change_chunk(self):
        self.actual_chunk = self.chunks[self.actual_chunk_position]
        self.chunk_combobox.setCurrentIndex(self.actual_chunk_position + 1)
        self.chunk_id = self.actual_chunk.id()
        self.chunk_length = self.actual_chunk.length()
        # thread = threading.Thread(group=None, target=self.load_pictures)
        # thread.start()
        self.load_pictures()

    def load_pictures(self):
        self.chunk_labels = []
        for child in self.chunk_detail_widget.findChildren(QLabel):
            self.chunk_detail_widget.layout().removeWidget(child)
            child.hide()
        if self.step < 0:
            self.chunk_detail_scroll.horizontalScrollBar().setValue(self.chunk_detail_scroll.horizontalScrollBar().maximum())
        else:
            self.chunk_detail_scroll.horizontalScrollBar().setValue(0)
        old_step = self.step
        self.step = 1
        region_generator = self.generate_next_region(self.actual_chunk)
        for region, frame in region_generator:
            pixmap = self.prepare_pixmap(region, frame, PICTURE_SIZE, RELATIVE_MARGIN)
            label = QtGui.QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setPixmap(pixmap)
            self.chunk_labels.append(label)
        self.step = old_step
        if self.step < 0:
            self.chunk_labels.reverse()
        for label in self.chunk_labels:
            self.chunk_detail_widget.layout().addWidget(label)

    def prepare_pixmap(self, region, frame, size, relative_margin):
        img = self.image_manager.get_crop(frame, region, width=size, height=size,
                                          relative_margin=relative_margin)
        pixmap = cvimg2qtpixmap(img)
        return pixmap

    def generate_next_region(self, chunk, mark=False):
        chunk_start = chunk.start_frame(self.graph_manager)
        chunk_end = chunk.end_frame(self.graph_manager)
        start = chunk_start if chunk_start > self.start_frame else self.start_frame
        end = chunk_end if chunk_end < self.end_frame else self.end_frame
        if self.step >= 0:
            if self.actual_chunk_position < self.number_of_chunks:
                i = start
            else:
                i = end
        if self.step < 0:
            if self.actual_chunk_position >= 0:
                i = end
            else:
                i = start

        j = 0 - self.step if self.step > 0 else len(self.chunk_labels) - 1 - self.step
        while i in range(start, end + 1):
            if mark:
                j += self.step
                if j in range(len(self.chunk_labels)):
                    self.chunk_labels[j].setStyleSheet(CHUNK_DETAIL_STYLESHEET)
                if j - self.step in range(len(self.chunk_labels)):
                    self.chunk_labels[j - self.step].setStyleSheet("")
                    if self.step > 0 and self.chunk_labels[j].pos().x() > self.picture_widget.width() / 2 or \
                                            self.step < 0 and self.chunk_labels[j].pos().x() < self.chunk_detail_widget.width() - self.picture_widget.width()/ 2:
                        self.chunk_detail_scroll.horizontalScrollBar().setValue(self.chunk_detail_scroll.horizontalScrollBar().value() +
                                                                                (PICTURE_SIZE + self.chunk_detail_widget.layout().spacing()) * self.step)
            yield self.graph_manager.region(chunk[i - chunk_start]), i
            i += self.step


if __name__ == '__main__':
    project = Project()
    project.load("/home/sheemon/FERDA/projects/Cam1_/cam1.fproj")
    app = QtGui.QApplication(sys.argv)
    widget = CropWidgetTracker(project, start_frame=0, end_frame=30)
    app.exec_()
    cv2.waitKey(0)
