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
from gui.settings import Settings as S_
from core.config import config
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
        self.step5_w = QtGui.QWidget()

        self.postpone_parallelisation = False

        self.form_layout = QtGui.QFormLayout()
        self.step1_w.setLayout(self.form_layout)

        self.form_layout.addRow(QtGui.QLabel('<i>Entries in </i><b>bold</b><i> are obligatory. When in doubt, use tooltips displayed on hover over given entries.</i>'))
        label = QtGui.QLabel('<b>Video files:</b> ')
        self.select_video_files = QtGui.QPushButton('Browse')
        self.select_video_files.setToolTip('Select two video files only in case when FERDA\'s video compression is used (find more in documentation).')
        self.select_video_files.clicked.connect(self.select_video_files_clicked)
        self.form_layout.addRow(label, self.select_video_files)

        label = QtGui.QLabel('<b>Working directory: </b>')
        self.select_working_directory = QtGui.QPushButton('Browse')
        self.select_working_directory.setToolTip('Select working directory for project. Best practice is to use empty directory.')
        self.select_working_directory.clicked.connect(self.select_working_directory_clicked)
        self.form_layout.addRow(label, self.select_working_directory)

        label = QtGui.QLabel('<b>Project name:</b> ')
        self.project_name = QtGui.QLineEdit()
        self.form_layout.addRow(label, self.project_name)

        label = QtGui.QLabel('Project description')
        self.project_description = QtGui.QPlainTextEdit(self)
        self.form_layout.addRow(label, self.project_description)

        self.postpone_parallelisation_ch = QtGui.QCheckBox('')
        self.postpone_parallelisation_ch.setChecked(False)

        self.postpone_parallelisation_ch.setToolTip("Check in a case when the segmentation will be computed on a cluster")
        self.form_layout.addRow('postpone segmentation parallelisation', self.postpone_parallelisation_ch)

        self.left_vbox = QtGui.QVBoxLayout()

        self.import_widget = ImportWidget()
        self.import_widget.import_button.clicked.connect(self.finish_import)
        self.import_widget.hide()
        self.import_widget.setDisabled(True)

        self.create_project_button = QtGui.QPushButton('continue', self)
        self.create_project_button.clicked.connect(self.create_project)

        self.hbox.addLayout(self.left_vbox)

        self.left_vbox.addWidget(self.step1_w)

        self.left_vbox.addWidget(self.import_widget)
        self.step1_w.layout().addWidget(self.create_project_button)

        self.bg_progress_bar = QtGui.QProgressBar()
        self.bg_progress_bar.setRange(0, 100)

        self.bg_computation = None

        self.video_preview_layout = QtGui.QFormLayout()
        self.hbox.addLayout(self.video_preview_layout)

        self.activateWindow()
        self.create_project_button.setFocus()

        self.project = Project()
        # self.project.working_directory = '/Users/flipajs/Documents/wd/FERDA/test/'
        # self.__go_to_3()

    def __go_to_3(self):
        self.project.video_paths = ['/Users/flipajs/Dropbox/FERDA/S9T95min.avi']

        setattr(self.step2_w, 'start_frame', 1)
        setattr(self.step2_w, 'end_frame', 4000)

        self.go_to_video_config()
        self.video_boundaries_confirmed()


    def select_video_files_clicked(self):
        path = ''
        if os.path.isdir(S_.temp.last_vid_path):
            path = S_.temp.last_vid_path
        self.project.video_paths = gui.gui_utils.file_names_dialog(self, 'Select video files', filter_="Videos (*.avi *.mkv *.mp4 *.m4v *.dav)", path=path)
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

        self.postpone_parallelisation = self.postpone_parallelisation_ch.isChecked()

        self.update_project_step1()

        from utils.img_manager import ImgManager
        self.project.img_manager = ImgManager(self.project, max_size_mb=config['cache']['img_manager_size_MB'])

        # if self.finish_callback:
        #     self.finish_callback('project_created', self.project)

        self.go_to_video_config()

    def segmentation_confirmed(self):
        from core.classes_stats import dummy_classes_stats
        print "segmentation_confirmed"

        if self.step4_w.gb_pixel_classifier.isChecked():
            with open(self.project.working_directory+'/segmentation_model.pkl', 'wb') as f:
                pickle.dump(self.step4_w.helper, f, -1)

            self.project.segmentation_model = self.step4_w.helper

        self.project.other_parameters.segmentation_use_roi_prediction_optimisation = self.step4_w.use_roi_prediction_optimisation_ch.isChecked()
        self.project.other_parameters.segmentation_prediction_optimisation_border = self.step4_w.prediction_optimisation_border_spin.value()
        self.project.other_parameters.full_segmentation_refresh_in_spin = self.step4_w.full_segmentation_refresh_in_spin.value()

        self.project.stats = dummy_classes_stats()

        self.project.stats.major_axis_median = self.step4_w.major_axis_median.value()
        self.project.solver_parameters.max_edge_distance_in_ant_length = self.step4_w.max_dist_object_length.value()
        self.step4_w.hide()

        w = self.step5_w
        w.setLayout(QtGui.QHBoxLayout())

        # TODO: clustering_tool

        button = QtGui.QPushButton('confirm and continue')
        button.clicked.connect(self.finish_initialisation)
        w.layout().addWidget(button)

        self.left_vbox.addWidget(self.step5_w)
        self.finish_initialisation(self.step4_w.num_animals_sb.value())

    def finish_initialisation(self, num_animals):
        from core.region.region_manager import RegionManager
        from core.graph.graph_manager import GraphManager
        from core.graph.solver import Solver
        from core.graph.chunk_manager import ChunkManager
        from core.animal import Animal

        self.project.rm = RegionManager(self.project.working_directory)
        self.project.solver = Solver(self.project)
        self.project.gm = GraphManager(self.project, self.project.solver)
        self.project.chm = ChunkManager()

        self.project.animals = []
        for i in range(num_animals):
            self.project.animals.append(Animal(i))

        self.project.solver_parameters.certainty_threshold = .01

        self.project.save()

        if self.finish_callback:
            self.finish_callback('initialization_finished', [self.project, self.postpone_parallelisation])

    def video_boundaries_confirmed(self):
        self.project.video_start_t = self.step2_w.start_frame + 1
        self.project.video_end_t = self.step2_w.end_frame

        self.step2_w.hide()

        from gui.init.init_where_widget import InitWhereWidget
        self.step3_w = InitWhereWidget(self.roi_finished, self.project)
        self.left_vbox.addWidget(self.step3_w)

    def roi_finished(self, mask_already_prepared=False):
        if not mask_already_prepared:
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
                                              video_crop_model['x2'] - video_crop_model['x1'])
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
