__author__ = 'fnaiser'
import sys
import os
from PyQt4 import QtGui, QtCore

# from gui.settings.dialogs import SettingsDialog
from gui import ferda_window_qt
from gui.init_window import init_window
from gui.init.init_widget import InitWidget
from gui import control_window
from gui.project import project_widget, new_project_widget
from gui.main_tab_widget import MainTabWidget
import core.project
from methods.bg_model.max_intensity import MaxIntensity
from gui.loading_widget import LoadingWidget
from gui.settings.settings_dialog import SettingsDialog
import pickle
import cv2
from utils.video_manager import get_auto_video_manager


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # IMPORTANT CLASSES INSTANCES
        self.project = None
        self.bg_model = None
        self.loading_widget = None

        # INIT WIDGETS
        self.new_project_widget = None
        self.main_tab_widget = None
        self.init_widget = None

        self.central_widget = QtGui.QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.project_widget = project_widget.ProjectWidget(self.widget_control)
        self.central_widget.addWidget(self.project_widget)

        self.setWindowIcon(QtGui.QIcon('imgs/ferda.ico'))
        self.setWindowTitle('FERDA')
        self.setGeometry(100, 100, 700, 400)

        # self.menu_bar = QtGui.QMenuBar(self)
        # self.toolbar = self.addToolBar('')

        self.settings_action = QtGui.QAction('&Settings', self.centralWidget())
        self.settings_action.triggered.connect(self.show_settings)
        self.settings_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_Comma))
        self.addAction(self.settings_action)

        # self.new_project_action = QtGui.QAction("New project", self.centralWidget())

        # self.load_project_action = QtGui.QAction("Load project", self.centralWidget())
        # self.load_project_action.triggered.connect(self.show_load_project_dialog)

        # self.show_correction_tool_action = QtGui.QAction("Correction tool", self.centralWidget())
        # self.settings_action.triggered.connect(self.show_correction_tool)

        # self.file_menu = self.menu_bar.addMenu('&File')
        # self.file_menu.addAction(self.settings_action)
        # self.file_menu.addAction(self.new_project_action)
        # self.file_menu.addAction(self.load_project_action)

        self.update()
        self.showMaximized()


    def closeEvent(self, event):
        print "exiting"
        #
        # if self.control_widget is not None:
        #     self.control_widget.close()

        event.accept()

    def widget_control(self, state, values=None):
        if state == 'new_project':
            self.new_project_widget = new_project_widget.NewProjectWidget(self.widget_control)
            self.central_widget.addWidget(self.new_project_widget)
            self.central_widget.setCurrentWidget(self.new_project_widget)

        if state == 'load_project':
            if isinstance(values, core.project.Project):
                self.project = values
                self.statusBar().showMessage("The project was successfully loaded.")
                self.setWindowTitle('FERDA - '+self.project.name)

                self.main_tab_widget = MainTabWidget(self.widget_control, self.project)
                self.central_widget.addWidget(self.main_tab_widget)
                self.central_widget.setCurrentWidget(self.main_tab_widget)
            else:
                self.statusBar().showMessage("Something went wrong during project loading!")

        if state == 'project_created':
            if isinstance(values, core.project.Project):
                self.project = values
                self.project.save()
                self.statusBar().showMessage("The project was successfully created.")
                self.setWindowTitle('FERDA - '+self.project.name)

                self.init_widget = InitWidget(self.widget_control, self.project)
                self.central_widget.addWidget(self.init_widget)
                self.central_widget.setCurrentWidget(self.init_widget)
            else:
                self.statusBar().showMessage("Something went wrong during project creation!")

        if state == 'new_project_back':
            self.central_widget.setCurrentWidget(self.project_widget)

        if state == 'initialization_finished':
            self.project.save()
            self.project.load(self.project.working_directory+'/'+self.project.name+'.fproj')
            self.main_tab_widget = MainTabWidget(self.widget_control, self.project)
            self.central_widget.addWidget(self.main_tab_widget)
            self.central_widget.setCurrentWidget(self.main_tab_widget)

    def start_ferda(self):
        self.control_widget = control_window.ControlWindow(self.init_widget.params, self.init_widget.ants, self.init_widget.video_manager)
        self.control_widget.set_exit_callback(self.control_widget_exit)

        self.central_widget.addWidget(self.control_widget)
        self.central_widget.setCurrentWidget(self.control_widget)

        x = self.x()
        y = self.y()
        w = 350
        h = 299

        self.setGeometry(QtCore.QRect(x, y, w, h))

    def show_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec_()

    def control_widget_exit(self):
        self.close()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    ex.raise_()
    ex.activateWindow()
    from core.project import Project
    proj = Project()
    # proj.load('/Volumes/Seagate Expansion Drive/working_dir/eight1/eight1.fproj')
    proj.load('/Users/fnaiser/Documents/work_dir/eight/eight.fproj')
    # proj.load('/Users/fnaiser/Documents/work_dir/big_lenses1/bl1.fproj')

    ex.widget_control('load_project', proj)

    # vid = get_auto_video_manager(proj.video_paths)
    # im = vid.move2_next()

    app.exec_()
    app.deleteLater()
    sys.exit()

