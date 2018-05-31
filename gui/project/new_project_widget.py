__author__ = 'fnaiser'
import os
from PyQt4 import QtGui, QtCore
import numpy as np
import time
import cPickle as pickle
from functools import partial
from core.region.region_manager import RegionManager
from core.graph.graph_manager import GraphManager
from core.graph.solver import Solver
from core.graph.chunk_manager import ChunkManager
from gui.project.import_widget import ImportWidget
from core.animal import Animal
from core.project.project import Project
from core.config import config
import gui.gui_utils
from gui.init.crop_video_widget import CropVideoWidget
from gui.region_classifier_tool import RegionClassifierTool
from gui.init.circle_arena_editor_widget import CircleArenaEditorWidget
from gui.init.set_msers import SetMSERs
from gui.settings import Settings as S_


class NewProjectWidget(QtGui.QWidget):
    def __init__(self, finish_callback):
        super(NewProjectWidget, self).__init__()
        self.finish_callback = finish_callback

        self.hbox = QtGui.QHBoxLayout()
        self.setLayout(self.hbox)

        # self.back_button = QtGui.QPushButton('return')
        # self.back_button.clicked.connect(self.back_button_clicked)
        # self.hbox.addWidget(self.back_button)

        self.widget1_create_project = QtGui.QWidget()
        self.widget2_crop_video = QtGui.QWidget()
        self.widget3_arena_editor = QtGui.QWidget()
        self.widget4_mser_setup = QtGui.QWidget()
        self.widget5_cardinality_classification = QtGui.QWidget()

        self.form_layout = QtGui.QFormLayout()
        self.widget1_create_project.setLayout(self.form_layout)

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

        self.left_vbox = QtGui.QVBoxLayout()

        self.import_widget = ImportWidget()
        self.import_widget.import_button.clicked.connect(self.finish_import)
        self.import_widget.hide()
        self.import_widget.setDisabled(True)

        self.hbox.addLayout(self.left_vbox)

        self.left_vbox.addWidget(self.widget1_create_project)

        self.left_vbox.addWidget(self.import_widget)

        self.bg_progress_bar = QtGui.QProgressBar()
        self.bg_progress_bar.setRange(0, 100)

        self.bg_computation = None

        self.video_preview_layout = QtGui.QFormLayout()
        self.hbox.addLayout(self.video_preview_layout)

        self.activateWindow()

        self.create_project_button = QtGui.QPushButton('continue', self)
        self.create_project_button.clicked.connect(self.create_project)
        self.widget1_create_project.layout().addWidget(self.create_project_button)
        self.create_project_button.setFocus()

        self.project = Project()
        # self.project.working_directory = '/Users/flipajs/Documents/wd/FERDA/test/'
        # self.__go_to_3()

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

    def setup_crop_video_widget(self):
        self.showMaximized()
        self.widget1_create_project.hide()

        w = CropVideoWidget(self.project)
        button = QtGui.QPushButton('confirm and continue')
        button.clicked.connect(self.crop_video_confirmed)
        w.layout().addWidget(button)

        self.widget2_crop_video = w
        self.left_vbox.addWidget(self.widget2_crop_video)

    def create_project(self):
        if self.project.working_directory == '':
            QtGui.QMessageBox.warning(self, "Warning", "Please choose working directory", QtGui.QMessageBox.Ok)
            return

        self.project.name = str(self.project_name.text())
        if not len(self.project.name):
            self.project.name = "untitled"
        self.project.description = str(self.project_description.toPlainText())
        self.project.date_created = time.time()
        self.project.date_last_modification = time.time()

        from utils.img_manager import ImgManager
        self.project.img_manager = ImgManager(self.project, max_size_mb=config['cache']['img_manager_size_MB'])

        self.setup_crop_video_widget()

    def mser_setup_confirmed(self):
        from core.classes_stats import dummy_classes_stats
        print "mser setup confirmed"

        if self.widget4_mser_setup.gb_pixel_classifier.isChecked():
            with open(self.project.working_directory+'/segmentation_model.pkl', 'wb') as f:
                pickle.dump(self.widget4_mser_setup.helper, f, -1)

            self.project.segmentation_model = self.widget4_mser_setup.helper

        self.project.other_parameters.segmentation_use_roi_prediction_optimisation = \
            self.widget4_mser_setup.use_roi_prediction_optimisation_ch.isChecked()
        self.project.other_parameters.segmentation_prediction_optimisation_border = \
            self.widget4_mser_setup.prediction_optimisation_border_spin.value()
        self.project.other_parameters.full_segmentation_refresh_in_spin = \
            self.widget4_mser_setup.full_segmentation_refresh_in_spin.value()

        self.project.stats = dummy_classes_stats()

        self.project.stats.major_axis_median = self.widget4_mser_setup.major_axis_median.value()
        self.project.solver_parameters.max_edge_distance_in_ant_length = self.widget4_mser_setup.max_dist_object_length.value()

        self.project.rm = RegionManager(self.project.working_directory)
        self.project.solver = Solver(self.project)
        self.project.gm = GraphManager(self.project, self.project.solver)
        self.project.chm = ChunkManager()
        self.project.animals = []
        for i in range(self.widget4_mser_setup.num_animals_sb.value()):
            self.project.animals.append(Animal(i))

        self.project.solver_parameters.certainty_threshold = .01

        self.project.save()

        self.setup_cardinality_classification_widget()

    def setup_cardinality_classification_widget(self):
        self.widget4_mser_setup.hide()
        self.widget5_cardinality_classification = RegionClassifierTool(self.project)
        # w.setLayout(QtGui.QHBoxLayout())
        # button = QtGui.QPushButton('confirm and continue')
        # button.clicked.connect(self.finish_initialisation)
        # w.layout().addWidget(button)
        self.widget5_cardinality_classification.on_finished.connect(self.cardinality_classification_finished)
        self.left_vbox.addWidget(self.widget5_cardinality_classification)

    def cardinality_classification_finished(self):
        self.finish_initialisation()

    def finish_initialisation(self):
        if self.finish_callback:
            self.finish_callback('initialization_finished', self.project)

    def crop_video_confirmed(self):
        self.project.video_start_t = self.widget2_crop_video.start_frame + 1
        self.project.video_end_t = self.widget2_crop_video.end_frame

        self.setup_arena_editor_widget()

    def setup_arena_editor_widget(self):
        self.widget2_crop_video.hide()

        self.widget3_arena_editor = CircleArenaEditorWidget(finish_callback=self.arena_editor_confirmed, project=self.project)
        self.left_vbox.addWidget(self.widget3_arena_editor)

    def arena_editor_confirmed(self, mask_already_prepared=False):
        if not mask_already_prepared:
            # TODO: deal with advanced arena editor
            c = np.array([self.widget3_arena_editor.arena_ellipse.c.pos().y(), self.widget3_arena_editor.arena_ellipse.c.pos().x()])
            r = np.array([self.widget3_arena_editor.arena_ellipse.a.pos().y(), self.widget3_arena_editor.arena_ellipse.a.pos().x()])
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

        self.setup_mser_setup_widget()

    def setup_mser_setup_widget(self):
        self.widget3_arena_editor.hide()
        self.widget4_mser_setup = SetMSERs(self.project, finish_callback=self.mser_setup_confirmed)
        self.left_vbox.addWidget(self.widget4_mser_setup)

    def set_video_bounds(self):
        if not self.project.video_paths:
            QtGui.QMessageBox.warning(self, "Warning", "Choose video path first", QtGui.QMessageBox.Ok)
            return

        w = CropVideoWidget(self.project)
        button = QtGui.QPushButton('confirm boundaries')
        button.clicked.connect(partial(self.crop_video_confirmed, w))
        w.layout().addWidget(button)
        self.layout().addWidget(w)

