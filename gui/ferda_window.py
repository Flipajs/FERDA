__author__ = 'flipajs'

import sys
from PyQt4 import QtGui, QtCore

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


    def control_widget_exit(self):
        self.close()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = FerdaControls()

    app.exec_()
    app.deleteLater()
    sys.exit()
