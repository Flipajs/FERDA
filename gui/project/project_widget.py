__author__ = 'fnaiser'

import sys
from functools import partial

import os
from PyQt4 import QtGui, QtCore

import core.project.project
import gui.gui_utils
from core.project.compatibility_solver import CompatibilitySolver
from gui.loading_widget import LoadingWidget
from gui.settings import Settings as S_
from core.config import config
from gui.settings_widgets.settings_dialog import SettingsDialog


class ProjectLoader(QtCore.QThread):
    proc_done = QtCore.pyqtSignal(object)
    part_done = QtCore.pyqtSignal(float)

    def __init__(self, project, path):
        super(ProjectLoader, self).__init__()

        self.project = project
        self.path = path

    def run(self):
        CompatibilitySolver(self.project)
        self.proc_done.emit(self.project)


class TestLoader(QtCore.QThread):
    proc_done = QtCore.pyqtSignal(object)
    part_done = QtCore.pyqtSignal(float)

    def __init__(self, r):
        super(TestLoader, self).__init__()

        self.r = r

    def run(self):
        list = []
        for i in range(self.r):
            list.append(i)

        self.proc_done.emit(self.project)


class ProjectWidget(QtGui.QWidget):
    def __init__(self, finish_callback=None, progress_callback=None):
        super(ProjectWidget, self).__init__()
        self.setLayout(QtGui.QVBoxLayout())
        self.finish_callback = finish_callback

        self.progress_callback = progress_callback

        self.settings_button = QtGui.QPushButton('Settings')
        self.settings_button.clicked.connect(self.show_settings)
        self.layout().addWidget(self.settings_button)

        self.new_project_button = QtGui.QPushButton('New Project', self)
        self.layout().addWidget(self.new_project_button)
        self.new_project_button.clicked.connect(self.new_project)

        self.load_project_button = QtGui.QPushButton('LoadProject', self)
        self.layout().addWidget(self.load_project_button)
        self.load_project_button.clicked.connect(self.load_project)

        self.new_project_button.setFixedHeight(100)
        self.load_project_button.setFixedHeight(100)

        self.b = QtGui.QPushButton('test')
        self.b.clicked.connect(self.test)
        # self.layout().addWidget(self.b)


        # loading speed in bytes per second
        self.loading_speed = 20000000
        # current loading status (0 on the beginning)
        self.status = 0
        self.timer_step = 0

        self.loading_thread = None
        self.update()

    def test(self):
        from functools import partial
        partial(self.progress_callback, True)

        self.tl = TestLoader(100000000)
        self.tl.start()

        partial(self.progress_callback, False)

    def show_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec_()

    def new_project(self):
        if self.finish_callback:
            self.finish_callback('new_project')

    def pick_new_video(self):
        reply = QtGui.QMessageBox.question(
            self, "Video file not found",
            "A video file for this project wasn't found. Please select the new position of the video in your filesystem.",
            "Cancel", "Choose video")
        if reply == 1:
            return str(QtGui.QFileDialog.getOpenFileName(self, 'Select new video location', '.'))
        else:
            return None

    def load_project(self):
        # pick .fproj location
        path = ''
        if os.path.isdir(S_.temp.last_wd_path):
            path = S_.temp.last_wd_path

        project_dir = QtGui.QFileDialog.getExistingDirectory(self, 'Select FERDA project folder', directory=path)
        project_dir = str(project_dir)
        if not project_dir:
            return

        project = core.project.project.Project()
        S_.temp.last_wd_path = project_dir

        # load project - this doesn't take as much time and is needed in the main thread to run popup windows
        if not core.project.project.project_video_file_exists(project_dir):
            path = self.pick_new_video()
            if path is None:
                return
            else:
                project.video_paths = path

        # disable all buttons, so another project can't be loaded/created at the same time
        self.load_project_button.setEnabled(False)
        self.new_project_button.setEnabled(False)
        self.settings_button.setEnabled(False)

        # add progress bar
        self.loading_w = LoadingWidget(text='Loading project... ')
        self.layout().addWidget(self.loading_w)
        QtGui.QApplication.processEvents()

        project.load(project_dir)

        # setup loading thread
        self.loading_thread = ProjectLoader(project, project_dir)
        self.loading_thread.proc_done.connect(partial(self.loading_finished, project))
        self.loading_thread.part_done.connect(self.loading_w.update_progress)

        # start loading thread and timer
        self.start_timer(project_dir)
        self.loading_thread.start()

    def start_timer(self, path):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.timer_done)
        self.timer.start(1000)
        # count timer_step
        self.timer_step = (self.loading_speed + 0.0)/self.get_size(path)

    def loading_finished(self, project):
        # stop timer and fill the progress bar
        project.rm.con.close()
        self.loading_w.update_progress(1)
        self.timer.stop()

        from core.region.region_manager import RegionManager
        project.rm = RegionManager(db_wd=project.working_directory, cache_size_limit=config['cache']['region_manager_num_of_instances'])

        self.finish_callback('load_project', project)

    def timer_done(self):
        # increase status by step, update the progress bar
        self.status += self.timer_step
        self.loading_w.update_progress(self.status)

    def get_size(self, project_dir):
        # gets size in bytes from all files that should be loaded
        size = 0
        # file = path+'/bg_model.pkl'
        # size += os.path.getsize(file)
        file = project_dir + '/arena_model.pkl'
        size += os.path.getsize(file)
        # file = path+'/classes.pkl'
        # size += os.path.getsize(file)
        # file = path+'/groups.pkl'
        # size += os.path.getsize(file)
        file = project_dir + '/animals.pkl'
        size += os.path.getsize(file)
        file = project_dir + '/stats.pkl'
        size += os.path.getsize(file)

        try:
            file = project_dir + '/progress_save.pkl'
            size += os.path.getsize(file)
        except OSError:
            pass

        return size


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    ex = ProjectWidget()
    ex.show()

    app.exec_()
    app.deleteLater()
    sys.exit()
