__author__ = 'fnaiser'

import sys
import os
import time

from PyQt4 import QtGui, QtCore
import core.project.project
import gui.gui_utils
from core.settings import Settings as S_
from gui.loading_widget import LoadingWidget
from functools import partial
from gui.settings.settings_dialog import SettingsDialog
from core.project.compatibility_solver import CompatibilitySolver


class ProjectLoader(QtCore.QThread):
    proc_done = QtCore.pyqtSignal(object)
    part_done = QtCore.pyqtSignal(float)

    def __init__(self, project, path):
        super(ProjectLoader, self).__init__()

        self.project = project
        self.path = path

    def run(self):
        self.project.load(self.path)
        CompatibilitySolver(self.project)
        self.project.rm.con.close()
        self.proc_done.emit(self.project)

class ProjectWidget(QtGui.QWidget):
    def __init__(self, finish_callback=None):
        super(ProjectWidget, self).__init__()
        self.setLayout(QtGui.QVBoxLayout())
        self.finish_callback = finish_callback

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

        # loading speed in bytes per second
        self.loading_speed = 20000000
        # current loading status (0 on the beginning)
        self.status = 0
        self.timer_step = 0

        self.loading_thread = None
        self.update()

    def show_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec_()

    def new_project(self):
        if self.finish_callback:
            self.finish_callback('new_project')

    def load_project(self):
        # pick .fproj location
        if os.path.isdir(S_.temp.last_wd_path):
            path = S_.temp.last_vid_path
            files = gui.gui_utils.file_names_dialog(self, 'Select FERDA project', filter_="Project (*.fproj)", path=path)
        if len(files) == 1:
            f = files[0]
            project = core.project.project.Project()

        # disable all buttons, so another project can't be loaded/created at the same time
        self.load_project_button.setEnabled(False)
        self.new_project_button.setEnabled(False)
        self.settings_button.setEnabled(False)

        # add progress bar
        self.loading_w = LoadingWidget(text='Loading project... ')
        self.layout().addWidget(self.loading_w)
        QtGui.QApplication.processEvents()

        # setup loading thread
        self.loading_thread = ProjectLoader(project, f)
        self.loading_thread.proc_done.connect(partial(self.loading_finished, project))
        self.loading_thread.part_done.connect(self.loading_w.update_progress)

        # start loading thread and timer
        self.start_timer(f)
        self.loading_thread.start()

    def start_timer(self, path):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.timer_done)
        self.timer.start(1000)
        # count timer_step
        self.timer_step = (self.loading_speed + 0.0)/self.get_size(path)

    def loading_finished(self, project):
        # stop timer and fill the progress bar
        self.loading_w.update_progress(1)
        self.timer.stop()

        from core.region.region_manager import RegionManager
        project.rm = RegionManager(db_wd=project.working_directory, cache_size_limit=S_.cache.region_manager_num_of_instances)

        self.finish_callback('load_project', project)

    def timer_done(self):
        # increase status by step, update the progress bar
        self.status += self.timer_step
        self.loading_w.update_progress(self.status)

    def get_size(self, path):
        # gets size in bytes from all files that should be loaded
        path = os.path.dirname(path)
        size = 0
        file = path+'/bg_model.pkl'
        size += os.path.getsize(file)
        file = path+'/arena_model.pkl'
        size += os.path.getsize(file)
        file = path+'/classes.pkl'
        size += os.path.getsize(file)
        file = path+'/groups.pkl'
        size += os.path.getsize(file)
        file = path+'/animals.pkl'
        size += os.path.getsize(file)
        file = path+'/stats.pkl'
        size += os.path.getsize(file)

        try:
            file = path+'/progress_save.pkl'
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