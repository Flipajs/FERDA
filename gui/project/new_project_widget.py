__author__ = 'fnaiser'

import threading
import os

from PyQt4 import QtGui, QtCore
import numpy as np
import skimage.transform
import time

import core.project.project
import gui.gui_utils
import utils.video_manager
import utils.misc
import utils.img
from core.bg_model.max_intensity import MaxIntensity
from gui.project.import_widget import ImportWidget
from gui.init.set_msers import SetMSERs
from core.project.project import Project
from gui.init.crop_video_widget import CropVideoWidget
from functools import partial
from core.settings import Settings as S_
import cPickle as pickle

class NewProjectWidget(QtGui.QWidget):
    def __init__(self, finish_callback):
        super(NewProjectWidget, self).__init__()
        self.finish_callback = finish_callback

        self.hbox = QtGui.QHBoxLayout()
        self.setLayout(self.hbox)

        # self.back_button = QtGui.QPushButton('return')
        # self.back_button.clicked.connect(self.back_button_clicked)
        # self.hbox.addWidget(self.back_button)

        self.step1_w = QtGui.QWidget()
        self.step2_w = QtGui.QWidget()
        self.step3_w = QtGui.QWidget()
        self.step4_w = QtGui.QWidget()

        self.form_layout = QtGui.QFormLayout()
        self.step1_w.setLayout(self.form_layout)

        label = QtGui.QLabel('Video files')
        self.select_video_files = QtGui.QPushButton('Browse')
        self.select_video_files.clicked.connect(self.select_video_files_clicked)
        self.form_layout.addRow(label, self.select_video_files)

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

        # self.set_msers_button = QtGui.QPushButton('Set MSERs')
        # self.set_msers_button.clicked.connect(self.set_msers)
        # self.form_layout.addRow('', self.set_msers_button)

        self.left_vbox = QtGui.QVBoxLayout()
        # self.import_templates = QtGui.QPushButton('Import templates')
        # self.import_templates.clicked.connect(self.import_templates_clicked)

        self.import_widget = ImportWidget()
        self.import_widget.import_button.clicked.connect(self.finish_import)
        self.import_widget.hide()
        self.import_widget.setDisabled(True)

        # self.certainty_slider = QtGui.QDoubleSpinBox()
        # self.certainty_slider.setMinimum(0)
        # self.certainty_slider.setMaximum(1)
        # self.certainty_slider.setSingleStep(0.01)
        # self.certainty_slider.setValue(0.5)
        # self.form_layout.addRow('min certainty: ', self.certainty_slider)
        # self.form_layout.addRow(QtGui.QLabel('0 means try to solve everything...'))

        # self.max_edge_distance = QtGui.QDoubleSpinBox()
        # self.max_edge_distance.setMinimum(0.1)
        # self.max_edge_distance.setMaximum(10)
        # self.max_edge_distance.setValue(2.5)
        # self.max_edge_distance.setSingleStep(0.05)
        # self.form_layout.addRow('max edge distance (in ant body length)', self.max_edge_distance)

        # self.use_colormarks_ch = gui.gui_utils.get_checkbox('Use colormarks', 'colormarks_use')
        # self.form_layout.addRow('use colormarks', self.use_colormarks_ch)

        self.create_project_button = QtGui.QPushButton('continue', self)
        self.create_project_button.clicked.connect(self.go_to_video_config)

        self.hbox.addLayout(self.left_vbox)

        self.left_vbox.addWidget(self.step1_w)

        self.left_vbox.addWidget(self.import_widget)
        # self.left_vbox.addWidget(self.import_templates)
        self.step1_w.layout().addWidget(self.create_project_button)

        self.bg_progress_bar = QtGui.QProgressBar()
        self.bg_progress_bar.setRange(0, 100)

        self.bg_computation = None

        self.video_preview_layout = QtGui.QFormLayout()
        self.hbox.addLayout(self.video_preview_layout)

        self.activateWindow()
        self.create_project_button.setFocus()

        self.project = Project()
        self.project.working_directory = '/Users/flipajs/Documents/wd/FERDA/test/'
        self.__go_to_3()

    def __go_to_3(self):
        self.project.video_paths = ['/Users/flipajs/Dropbox/FERDA/Cam1_clip.avi']

        setattr(self.step2_w, 'start_frame', 1)
        setattr(self.step2_w, 'end_frame', 4000)

        self.go_to_video_config()
        self.video_boundaries_confirmed()


    def select_video_files_clicked(self):
        path = ''
        if os.path.isdir(S_.temp.last_vid_path):
            path = S_.temp.last_vid_path
        self.project.video_paths = gui.gui_utils.file_names_dialog(self, 'Select video files', filter_="Videos (*.avi *.mkv *.mp4 *.m4v)", path=path)
        if self.project.video_paths:
            S_.temp.last_vid_path = os.path.dirname(self.project.video_paths[0])

        # self.set_video_bounds()
        self.activateWindow()
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

        tentative_name = working_directory.split('/')[-1]
        self.project_name.setText(tentative_name)

        self.activateWindow()

        self.project.working_directory = working_directory

        # self.project_name.setFocus()
        self.create_project_button.setFocus()

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

    def update_project_step1(self):
        self.project.name = self.project_name.text()
        if not len(self.project.name):
            self.project.name = "untitled"

        self.project.description = str(self.project_description.toPlainText())

        self.project.date_created = time.time()
        self.project.date_last_modifiaction = time.time()

    def go_to_video_config(self):
        self.showMaximized()
        self.step1_w.hide()

        w = CropVideoWidget(self.project)
        button = QtGui.QPushButton('confirm and continue')
        button.clicked.connect(self.video_boundaries_confirmed)
        w.layout().addWidget(button)

        self.step2_w = w
        self.left_vbox.addWidget(self.step2_w)

    def create_project(self):
        if self.project.working_directory == '':
            QtGui.QMessageBox.warning(self, "Warning", "Please choose working directory", QtGui.QMessageBox.Ok)
            return

        self.update_project_step1()

        from utils.img_manager import ImgManager
        self.project.img_manager = ImgManager(self.project, max_size_mb=S_.cache.img_manager_size_MB)

        if self.finish_callback:
            self.finish_callback('project_created', self.project)

    # def set_msers(self):
        # if self.project.video_paths:
        #     self.d_ = QtGui.QDialog()
        #     self.d_.setLayout(QtGui.QVBoxLayout())
        #     sm = SetMSERs(self.project)
        #     self.d_.layout().addWidget(sm)
        #     self.d_.showMaximized()
        #     self.d_.exec_()
        #
        #     button = QtGui.QPushButton('confirm and continue')
        #     button.clicked.connect(self.segmentation_confirmed)
        #     sm.left_panel.layout().addWidget(self.button_done)
        # else:
        #     QtGui.QMessageBox.warning(self, "Warning", "Choose video path first", QtGui.QMessageBox.Ok)

    def segmentation_confirmed(self):
        print "segmentation_confirmed"

        with open(self.project.working_directory+'/segmentation_model.pkl', 'wb') as f:
            pickle.dump(self.step4_w.helper, f, -1)

        self.project.segmentation_model = self.step4_w.helper

        self.step4_w.hide()
        pass

    def video_boundaries_confirmed(self):
        self.project.video_start_t = self.step2_w.start_frame + 1
        self.project.video_end_t = self.step2_w.end_frame

        self.step2_w.hide()

        from gui.init.init_where_widget import InitWhereWidget
        self.step3_w = InitWhereWidget(self.roi_finished, self.project)
        self.left_vbox.addWidget(self.step3_w)

    def roi_finished(self):
        # self.project.video_crop_model = {'y1': w.sc_y1.value(),
        #                                  'y2': w.sc_y2.value(),
        #                                  'x1': w.sc_x1.value(),
        #                                  'x2': w.sc_x2.value()}

        # TODO: deal with advanced arena editor
        c = np.array([self.step3_w.arena_ellipse.c.pos().y(), self.step3_w.arena_ellipse.c.pos().x()])
        r = np.array([self.step3_w.arena_ellipse.a.pos().y(), self.step3_w.arena_ellipse.a.pos().x()])
        r = np.linalg.norm(c - r)

        from utils.video_manager import get_auto_video_manager
        vm = get_auto_video_manager(self.project)
        im = vm.next_frame()

        from math import ceil, floor
        from core.arena.circle import Circle

        video_crop_model = {}
        video_crop_model['y1'] = int(max(0, floor(c[0]-r)))
        video_crop_model['x1'] = int(max(0, floor(c[1]-r)))
        video_crop_model['y2'] = int(min(im.shape[0], ceil(c[0]+r)))
        video_crop_model['x2'] = int(min(im.shape[1], ceil(c[1]+r)))
        self.project.video_crop_model = video_crop_model

        c = np.array([c[0] - video_crop_model['y1'], c[1] - video_crop_model['x1']])
        self.project.arena_model = Circle(video_crop_model['y2'] - video_crop_model['y1'],
                                          video_crop_model['x2'] - video_crop_model['x2'])
        self.project.arena_model.set_circle(c, r)

        from gui.init.set_msers import SetMSERs
        self.step3_w.hide()
        self.step4_w = SetMSERs(self.project)
        button = QtGui.QPushButton('confirm and continue')
        button.clicked.connect(self.segmentation_confirmed)
        self.step4_w.left_panel.layout().addWidget(button)

        self.left_vbox.addWidget(self.step4_w)

    def set_video_bounds(self):
        if not self.project.video_paths:
            QtGui.QMessageBox.warning(self, "Warning", "Choose video path first", QtGui.QMessageBox.Ok)
            return

        w = CropVideoWidget(self.project)
        button = QtGui.QPushButton('confirm boundaries')
        button.clicked.connect(partial(self.video_boundaries_confirmed, w))
        w.layout().addWidget(button)
        self.layout().addWidget(w)
