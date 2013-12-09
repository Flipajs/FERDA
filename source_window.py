__author__ = 'flip'

import sys
import os
from PyQt4 import QtGui
import ants_main
import experimentParams
import init_window


class SourceWindow(QtGui.QWidget, ants_main.Ui_Dialog):
    def __init__(self):
        super(SourceWindow, self).__init__()
        self.setupUi(self)

        self.init_ui()

        self.params = experimentParams.Params()

    def init_ui(self):
        self.choose_video_button.clicked.connect(self.show_file_dialog)
        self.start_button.clicked.connect(self.start)
        self.show()

    def show_file_dialog(self):
        self.params.video_file_name = str(QtGui.QFileDialog.getOpenFileName(self, "Select video file"))
        drive, path = os.path.splitdrive(self.params.video_file_name)
        path, filename = os.path.split(path)
        self.file_name_label.setText(filename)

    def get_video_file_name(self):
        return self.video_file_name

    def start(self):
        self.hide()
        p = init_window.InitWindow(self.params)
        p.show()
        p.exec_()


def main():
    app = QtGui.QApplication(sys.argv)
    ex = SourceWindow()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()