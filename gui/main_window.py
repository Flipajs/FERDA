__author__ = 'fnaiser'

import sys
from PyQt4 import QtGui, QtCore

# from gui.settings.dialogs import SettingsDialog
from gui import ferda_window_qt
from gui.init_window import init_window
from gui import control_window
from gui.project import project_widget, new_project_widget
import core.project


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # IMPORTANT CLASSES INSTANCES
        self.project = None


        # INIT WIDGETS
        self.new_project_widget = None

        self.central_widget = QtGui.QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.project_widget = project_widget.ProjectWidget(self.widget_control)
        self.central_widget.addWidget(self.project_widget)

        self.setWindowIcon(QtGui.QIcon('imgs/ferda.ico'))
        self.setWindowTitle('FERDA')
        self.setGeometry(100, 100, 700, 400)

        self.menu_bar = self.menuBar()
        self.toolbar = self.addToolBar('')

        self.settings_action = QtGui.QAction('&Settings', self.centralWidget())
        self.settings_action.triggered.connect(self.show_settings)

        self.new_project_action = QtGui.QAction("New project", self.centralWidget())

        self.load_project_action = QtGui.QAction("Load project", self.centralWidget())
        # self.load_project_action.triggered.connect(self.show_load_project_dialog)

        self.show_correction_tool_action = QtGui.QAction("Correction tool", self.centralWidget())
        # self.settings_action.triggered.connect(self.show_correction_tool)

        self.file_menu = self.menu_bar.addMenu('&File')
        self.file_menu.addAction(self.settings_action)
        self.file_menu.addAction(self.new_project_action)
        self.file_menu.addAction(self.load_project_action)

        self.update()
        self.show()

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
            else:
                self.statusBar().showMessage("Something went wrong during project loading!")

        if state == 'project_created':
            if isinstance(values, core.project.Project):
                self.project = values
                self.project.save()
                self.statusBar().showMessage("The project was successfully created.")
            else:
                self.statusBar().showMessage("Something went wrong during project creation!")

        if state == 'new_project_back':
            self.central_widget.setCurrentWidget(self.project_widget)

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

    app.exec_()
    app.deleteLater()
    sys.exit()

