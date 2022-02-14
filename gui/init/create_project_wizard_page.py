from PyQt6 import QtCore, QtGui, QtWidgets
import os
import time

import gui.gui_utils
from gui.settings import Settings as S_
from gui.generated.ui_create_project_page import Ui_createProjectPage
from utils.video_manager import get_auto_video_manager


class CreateProjectPage(QtWidgets.QWizardPage):
    def __init__(self):
        super(CreateProjectPage, self).__init__()

        self.ui = Ui_createProjectPage()
        self.ui.setupUi(self)
        self.registerField('project_folder*', self.ui.projectFolderEdit)
        self.registerField('video_file*', self.ui.videoFileEdit)
        self.registerField('project_name', self.ui.projectNameEdit)
        self.registerField('project_description', self.ui.projectDescriptionEdit)

        self.ui.videoFileButton.clicked.connect(self.select_video_file)
        self.ui.projectFolderButton.clicked.connect(self.select_project_folder)
        self.ui.videoFileEdit.textChanged.connect(self.reset_video_file_warning)
        self.ui.projectFolderEdit.textChanged.connect(self.check_project_folder)

        self.warning_default_stylesheet = self.ui.videoFileWarning.styleSheet()

    def reset_video_file_warning(self):
        self.show_warning(self.ui.videoFileWarning, None)

    def show_warning(self, label_widget, text=None):
        if text is not None:
            label_widget.setStyleSheet('QLabel { background-color : red; color : white; }')
            label_widget.setText(text)
        else:
            label_widget.setStyleSheet(self.warning_default_stylesheet)
            label_widget.setText('')

    def check_project_folder(self):
        project_folder = self.ui.projectFolderEdit.text()

        if os.path.isdir(project_folder):
            filenames = os.listdir(project_folder)
            for f in filenames:
                if os.path.isfile(project_folder + '/' + f) and (f.endswith('.fproj') or f.endswith('project.json')):
                    self.show_warning(self.ui.projectFolderWarning, 'this will overwrite existing project')
                    return True  # overwriting a project is allowed
        else:
            self.show_warning(self.ui.projectFolderWarning, 'project folder doesn\'t exist')
            return False

        self.show_warning(self.ui.projectFolderWarning, None)
        return True

    def select_video_file(self):
        path = ''
        if os.path.isdir(S_.temp.last_vid_path):
            path = S_.temp.last_vid_path
        video_path = gui.gui_utils.file_name_dialog(self, 'Select video file',
                                                    filter_="Videos (*.avi *.mkv *.mp4 *.m4v *.dav)",
                                                    path=path)
        if video_path:
            S_.temp.last_vid_path = os.path.dirname(video_path)
            self.ui.videoFileEdit.setText(video_path)

    def select_project_folder(self):
        if os.path.isdir(S_.temp.last_wd_path):
            path = S_.temp.last_wd_path
        else:
            path = ''
        working_directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select project folder",
                                                                       path, QtWidgets.QFileDialog.Option.ShowDirsOnly))
        if working_directory:
            S_.temp.last_wd_path = os.path.dirname(working_directory)
            tentative_name = working_directory.split('/')[-1]
            self.ui.projectFolderEdit.setText(working_directory)
            self.ui.projectNameEdit.setText(tentative_name)

    def validatePage(self):
        project = self.wizard().project
        project.working_directory = str(self.field('project_folder').toString())
        project.video_path = str(self.field('video_file').toString())
        project.name = str(self.field('project_name').toString())
        if not len(project.name):
            project.name = 'untitled'
        project.description = str(self.field('project_description').toString())
        project.date_created = time.time()
        project.date_last_modification = time.time()
        # from utils.img_manager import ImgManager
        # self.project.img_manager = ImgManager(self.project, max_size_mb=config['cache']['img_manager_size_MB'])

        if not self.check_project_folder():
            return False
        try:
            get_auto_video_manager(project)
            self.show_warning(self.ui.videoFileWarning, '')
        except OSError:
            self.show_warning(self.ui.videoFileWarning, 'can\'t open the video file')
            return False

        return True
