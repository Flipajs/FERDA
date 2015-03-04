import os

__author__ = 'fnaiser'

import sys
from PyQt4 import QtGui
import numpy as np
import core.project
import utils.gui
import utils.video_manager
import utils.misc
import utils.img
import skimage.transform
import threading
from utils.video_manager import VideoType
from methods.bg_model.max_intensity import MaxIntensity

class BGSub(threading.Thread):
    def __init__(self, vid, update_callback):
        super(BGSub, self).__init__()
        self.running = False
        self.vid = vid
        self.update_callback = update_callback


    def run(self):
        print "COMPUTING BG...."
        num_of_steps = 50
        for i in range(num_of_steps):
            self.update_callback(int(100*(i+1)/float(num_of_steps)))

            im = self.vid.random_frame()


class NewProjectWidget(QtGui.QWidget):
    def __init__(self, finish_callback):
        super(NewProjectWidget, self).__init__()
        self.finish_callback = finish_callback

        self.video_files = None
        self.working_directory = ''

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

        label = QtGui.QLabel('Working directory')
        self.select_working_directory = QtGui.QPushButton('Browse')
        self.select_working_directory.clicked.connect(self.select_working_directory_clicked)
        self.form_layout.addRow(label, self.select_working_directory)

        label = QtGui.QLabel('Project name')
        self.project_name = QtGui.QLineEdit()
        self.form_layout.addRow(label, self.project_name)

        label = QtGui.QLabel('Project description', self)
        self.project_description = QtGui.QPlainTextEdit(self)
        self.form_layout.addRow(label, self.project_description)

        self.create_project_button = QtGui.QPushButton('Create new project', self)
        self.form_layout.addWidget(self.create_project_button)
        self.create_project_button.clicked.connect(self.create_project)

        self.hbox.addLayout(self.form_layout)

        self.bg_progress_bar = QtGui.QProgressBar()
        self.bg_progress_bar.setRange(0, 100)

        self.bg_computation = None

        self.video_preview_layout = QtGui.QFormLayout()
        self.hbox.addLayout(self.video_preview_layout)
        self.update()
        self.show()

    def select_video_files_clicked(self):
        self.video_files = utils.gui.file_names_dialog(self, 'Select video files', '*.avi; *.mkv; *.mp4') #, 'AVI (*.avi);MKV (*.mkv); MP4 (*.mp4)')
        try:
            vid = utils.video_manager.get_auto_video_manager(self.video_files)
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
            layout.setWordWrap(True);
            self.video_preview_layout.addRow(None, layout)

            # self.bg_computation = BGSub(vid, self.update_progress_label)
            self.bg_computation = MaxIntensity(self.video_files, update_callback=self.update_progress_label)
            self.bg_computation.start()


        except Exception as e:
            utils.misc.print_exception(e)

    def back_button_clicked(self):
        self.finish_callback('new_project_back')

    def update_progress_label(self, val):
        self.bg_progress_bar.setValue(val)

    def select_working_directory_clicked(self):
        self.working_directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select working directory"))

        if os.path.isdir(self.working_directory):
            filenames = os.listdir(self.working_directory)
            for f in filenames:
                if os.path.isfile(self.working_directory+'/'+f) and f.endswith('.fproj'):
                    QtGui.QMessageBox.information(None, '', 'This folder is already used for FERDA project, choose different one, please')
                    self.select_working_directory_clicked()


    def create_project(self):
        project = core.project.Project()
        project.name = self.project_name.text()
        project.description = str(self.project_description.toPlainText())
        project.video_paths = self.video_files
        project.working_directory = self.working_directory

        if self.finish_callback:
            self.finish_callback('project_created', {'project': project, 'bg_model': self.bg_computation})