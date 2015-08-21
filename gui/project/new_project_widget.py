__author__ = 'fnaiser'

import threading
import os

from PyQt4 import QtGui, QtCore
import numpy as np
import skimage.transform

import core.project.project
import gui.gui_utils
import utils.video_manager
import utils.misc
import utils.img
from utils.video_manager import VideoType
from core.bg_model.max_intensity import MaxIntensity
from core.settings import Settings as S_
from gui.project.import_widget import ImportWidget
from gui.init.set_msers import SetMSERs
from core.project.project import Project
from gui.init.crop_video_widget import CropVideoWidget
from functools import partial


class NewProjectWidget(QtGui.QWidget):
    def __init__(self, finish_callback):
        super(NewProjectWidget, self).__init__()
        self.finish_callback = finish_callback

        self.hbox = QtGui.QHBoxLayout()
        self.setLayout(self.hbox)

        self.back_button = QtGui.QPushButton('Back')
        self.back_button.clicked.connect(self.back_button_clicked)
        self.hbox.addWidget(self.back_button)

        self.form_layout = QtGui.QFormLayout()

        label = QtGui.QLabel('Video files')
        self.select_video_files = QtGui.QPushButton('Browse')
        self.select_video_files.clicked.connect(self.select_video_files_clicked)
        self.form_layout.addRow(label, self.select_video_files)

        self.video_bounds_b = QtGui.QPushButton('Set video bounds')
        self.video_bounds_b.clicked.connect(self.set_video_bounds)
        self.form_layout.addRow('', self.video_bounds_b)

        label = QtGui.QLabel('Working directory')
        self.select_working_directory = QtGui.QPushButton('Browse')
        self.select_working_directory.clicked.connect(self.select_working_directory_clicked)
        self.form_layout.addRow(label, self.select_working_directory)

        label = QtGui.QLabel('Project name')
        self.project_name = QtGui.QLineEdit()
        self.form_layout.addRow(label, self.project_name)

        label = QtGui.QLabel('Project description')
        self.project_description = QtGui.QPlainTextEdit(self)
        self.form_layout.addRow(label, self.project_description)

        self.set_msers_button = QtGui.QPushButton('Set MSERs')
        self.set_msers_button.clicked.connect(self.set_msers)
        self.form_layout.addRow('', self.set_msers_button)

        self.left_vbox = QtGui.QVBoxLayout()
        self.import_templates = QtGui.QPushButton('Import templates')
        self.import_templates.clicked.connect(self.import_templates_clicked)

        self.import_widget = ImportWidget()
        self.import_widget.import_button.clicked.connect(self.finish_import)
        self.import_widget.hide()

        self.certainty_slider = QtGui.QDoubleSpinBox()
        self.certainty_slider.setMinimum(0)
        self.certainty_slider.setMaximum(1)
        self.certainty_slider.setSingleStep(0.01)
        self.certainty_slider.setValue(0.5)
        self.form_layout.addRow('min certainty: ', self.certainty_slider)
        self.form_layout.addRow(QtGui.QLabel('0 means try to solve everything...'))

        self.create_project_button = QtGui.QPushButton('Create new project', self)
        self.create_project_button.clicked.connect(self.create_project)

        self.hbox.addLayout(self.left_vbox)
        self.left_vbox.addLayout(self.form_layout)
        self.left_vbox.addWidget(self.import_widget)
        self.left_vbox.addWidget(self.import_templates)
        self.left_vbox.addWidget(self.create_project_button)

        self.bg_progress_bar = QtGui.QProgressBar()
        self.bg_progress_bar.setRange(0, 100)

        self.bg_computation = None

        self.video_preview_layout = QtGui.QFormLayout()
        self.hbox.addLayout(self.video_preview_layout)

        self.project = Project()

    def select_video_files_clicked(self):
        path = ''
        if os.path.isdir(S_.temp.last_vid_path):
            path = S_.temp.last_vid_path
        self.project.video_paths = gui.gui_utils.file_names_dialog(self, 'Select video files', filter_="Videos (*.avi *.mkv *.mp4 *.m4v)", path=path)
        if self.project.video_paths:
            S_.temp.last_vid_path = os.path.dirname(self.project.video_paths[0])

        self.select_working_directory.setFocus()

    def back_button_clicked(self):
        self.finish_callback('new_project_back')

    def update_progress_label(self, val):
        self.bg_progress_bar.setValue(val)

    def select_working_directory_clicked(self):
        path = ''
        if os.path.isdir(S_.temp.last_wd_path):
            path = S_.temp.last_wd_path

        working_directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select working directory", path, QtGui.QFileDialog.ShowDirsOnly))

        S_.temp.last_wd_path = os.path.dirname(working_directory)

        if os.path.isdir(working_directory):
            filenames = os.listdir(working_directory)
            for f in filenames:
                if os.path.isfile(working_directory+'/'+f) and f.endswith('.fproj'):
                    QtGui.QMessageBox.information(None, '', 'This folder is already used for FERDA project, choose different one, please')
                    self.select_working_directory_clicked()
                    return

        if not self.project.video_paths:
            QtGui.QMessageBox.warning(self, "Warning", "Choose video path first", QtGui.QMessageBox.Ok)
            return

        self.bg_computation = MaxIntensity(self.project)
        self.connect(self.bg_computation, QtCore.SIGNAL("update(int)"), self.update_progress_label)
        self.bg_computation.start()
        self.activateWindow()

        vid = utils.video_manager.get_auto_video_manager(self.project)
        im = vid.random_frame()
        h, w, _ = im.shape
        im = np.asarray(skimage.transform.resize(im, (100, 100))*255, dtype=np.uint8)

        img_label = QtGui.QLabel()
        img_label.setPixmap(utils.img.get_pixmap_from_np_bgr(im))
        layout = QtGui.QLabel('preview: ')
        self.video_preview_layout.addRow(layout, img_label)
        layout = QtGui.QLabel('#frames: ')
        value_layout = QtGui.QLabel(str(vid.total_frame_count()))
        self.video_preview_layout.addRow(layout, value_layout)

        layout = QtGui.QLabel('resolution: ')
        value_layout = QtGui.QLabel(str(w)+'x'+str(h)+'px')
        self.video_preview_layout.addRow(layout, value_layout)
        self.video_preview_layout.addRow(None, self.bg_progress_bar)
        layout = QtGui.QLabel('Video pre-processing in progress running in background... But don\'t worry, you can continue with your project creation and initialization meanwhile it will be finished.')
        layout.setWordWrap(True)
        self.video_preview_layout.addRow(None, layout)

        self.project.working_directory = working_directory

        self.project_name.setFocus()

    def import_templates_clicked(self):
        self.project_name.setDisabled(True)
        self.project_description.setDisabled(True)
        self.select_working_directory.setDisabled(True)
        self.select_video_files.setDisabled(True)

        self.import_templates.hide()
        self.create_project_button.hide()

        self.import_widget.show()

    def finish_import(self):
        project = self.get_project()
        self.import_widget.finish_import(project)

        if self.finish_callback:
            self.finish_callback('project_created', project)

    def update_project(self):
        self.project.name = self.project_name.text()
        if not len(self.project.name):
            self.project.name = "untitled"

        self.project.description = str(self.project_description.toPlainText())

        self.project.bg_model = self.bg_computation

        import time
        self.project.date_created = time.time()
        self.project.date_last_modifiaction = time.time()
        self.project.solver_parameters.certainty_threshold = self.certainty_slider.value()

    def create_project(self):
        if self.project.working_directory == '':
            QtGui.QMessageBox.warning(self, "Warning", "Please choose working directory", QtGui.QMessageBox.Ok)
            return

        self.update_project()

        if self.finish_callback:
            self.finish_callback('project_created', self.project)

    def set_msers(self):
        if self.project.video_paths:
            self.d_ = QtGui.QDialog()
            self.d_.setLayout(QtGui.QVBoxLayout())
            sm = SetMSERs(self.project)
            self.d_.layout().addWidget(sm)
            self.d_.showMaximized()
            self.d_.exec_()
        else:
            QtGui.QMessageBox.warning(self, "Warning", "Choose video path first", QtGui.QMessageBox.Ok)

    def video_boundaries_confirmed(self, w):
        self.project.video_start_t = w.start_frame + 1
        self.project.video_end_t = w.end_frame

        w.hide()
        w.setParent(None)

    def set_video_bounds(self):
        if not self.project.video_paths:
            QtGui.QMessageBox.warning(self, "Warning", "Choose video path first", QtGui.QMessageBox.Ok)
            return

        w = CropVideoWidget(self.project)
        button = QtGui.QPushButton('confirm boundaries')
        button.clicked.connect(partial(self.video_boundaries_confirmed, w))
        w.layout().addWidget(button)
        self.layout().addWidget(w)
