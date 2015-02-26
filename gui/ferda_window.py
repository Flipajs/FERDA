__author__ = 'filip@naiser.cz'

import sys
from PyQt4 import QtGui, QtCore

# from gui.settings.dialogs import SettingsDialog
from gui import ferda_window_qt
from gui.init_window import init_window
from gui import control_window


class FerdaControls(QtGui.QMainWindow, ferda_window_qt.Ui_MainWindow):
    def __init__(self):
        super(FerdaControls, self).__init__()
        self.setupUi(self)

        self.central_widget = QtGui.QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.init_widget = init_window.InitWindow(self)
        self.init_widget.set_close_callback(self.start_ferda)
        self.central_widget.addWidget(self.init_widget)

        self.control_widget = None
        self.setWindowIcon(QtGui.QIcon('imgs/ferda.ico'))

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

        # self.toolbar.addAction(self.settings_action)
        # self.toolbar.addAction(self.load_project_action)
        # self.toolbar.addAction(self.show_correction_tool_action)

        # self.centralWidget().layout().add
        # self.layout().addWidget(w)

        self.update()
        self.show()

    def closeEvent(self, event):
        print "exiting"

        if self.control_widget is not None:
            self.control_widget.close()

        event.accept()

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
    ex = FerdaControls()

    app.exec_()
    app.deleteLater()
    sys.exit()
