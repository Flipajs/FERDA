__author__ = 'fnaiser'

import sys
import os

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
        cs = CompatibilitySolver(self.project)
        # self.project.load(self.path)
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

        self.loading_thread = None
        self.update()

    def show_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec_()

    def new_project(self):
        if self.finish_callback:
            self.finish_callback('new_project')

    def loading_finished(self, project):
        # w.hide()
        self.finish_callback('load_project', project)

    def load_project(self):
        path = ''
        if os.path.isdir(S_.temp.last_wd_path):
            path = S_.temp.last_vid_path
        files = gui.gui_utils.file_names_dialog(self, 'Select FERDA project', filter_="Project (*.fproj)", path=path)
        if len(files) == 1:
            f = files[0]
            project = core.project.project.Project()

            loading_w = LoadingWidget(text='Loading project... Probably the progress bar won\'t move... But at least this will prevent window freezing')
            self.layout().addWidget(loading_w)
            QtGui.QApplication.processEvents()

            self.loading_thread = ProjectLoader(project, f)
            # self.loading_thread.proc_done.connect(partial(self.loading_finished, project, loading_w))
            self.loading_thread.proc_done.connect(self.loading_finished)
            self.loading_thread.part_done.connect(loading_w.update_progress)
            self.loading_thread.start()



if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    ex = ProjectWidget()

    app.exec_()
    app.deleteLater()
    sys.exit()