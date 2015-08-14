__author__ = 'fnaiser'

import sys

from PyQt4 import QtGui, QtCore

from gui.init.init_widget import InitWidget
from gui.project import project_widget, new_project_widget
from gui.main_tab_widget import MainTabWidget
import core.project.project
from gui.settings.settings_dialog import SettingsDialog
from core.settings import Settings as S_


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

        self.settings_button = QtGui.QPushButton('Settings', self)
        self.settings_button.clicked.connect(self.show_settings)
        self.layout().addWidget(self.settings_button)

        self.setWindowIcon(QtGui.QIcon('imgs/ferda.ico'))
        self.setWindowTitle('FERDA')
        self.setGeometry(100, 100, 700, 400)

        self.settings_action = QtGui.QAction('&Settings', self.centralWidget())
        self.settings_action.triggered.connect(self.show_settings)
        self.settings_action.setShortcut(S_.controls.show_settings)
        self.addAction(self.settings_action)

        self.update()

        self.show()

        # TOOD: remove this hack...
        self.move(-500, -500)
        self.showMaximized()

    def closeEvent(self, event):
        event.accept()

    def widget_control(self, state, values=None):
        self.settings_button.hide()
        if state == 'new_project':
            self.new_project_widget = new_project_widget.NewProjectWidget(self.widget_control)
            self.central_widget.addWidget(self.new_project_widget)
            self.central_widget.setCurrentWidget(self.new_project_widget)
            self.new_project_widget.select_video_files.setFocus(True)

        if state == 'load_project':
            if isinstance(values, core.project.project.Project):
                self.project = values
                self.statusBar().showMessage("The project was successfully loaded.")
                self.setWindowTitle('FERDA - '+self.project.name)

                self.main_tab_widget = MainTabWidget(self.widget_control, self.project)
                self.central_widget.addWidget(self.main_tab_widget)
                self.central_widget.setCurrentWidget(self.main_tab_widget)
            else:
                self.statusBar().showMessage("Something went wrong during project loading!")

        if state == 'project_created':
            if isinstance(values, core.project.project.Project):
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
            self.settings_button.show()

        if state == 'initialization_finished':
            self.project.save()

            self.main_tab_widget = MainTabWidget(self.widget_control, self.project)
            self.central_widget.addWidget(self.main_tab_widget)
            self.central_widget.setCurrentWidget(self.main_tab_widget)

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
    from core.project.project import Project
    proj = Project()
    proj.load('/Users/fnaiser/Documents/work_dir/eight/eight.fproj')
    ex.widget_control('load_project', proj)

    app.exec_()
    app.deleteLater()
    sys.exit()

