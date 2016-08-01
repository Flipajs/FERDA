from PyQt4 import QtGui, QtCore
import sys
import os

class LearningWidget(QtGui.QWidget):
    def __init__(self, project=None):
        super(LearningWidget, self).__init__()

        self.project = project
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        if not self.project:
            self.load_project_button = QtGui.QPushButton('load project')
            self.load_project_button.clicked.connect(self.load_project)
            self.vbox.addWidget(self.load_project_button)
        else:
            self.show_menu

    def load_project(self):
        path = ''
        if os.path.isdir(S_.temp.last_wd_path):
            path = S_.temp.last_wd_path

        working_directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select working directory", path, QtGui.QFileDialog.ShowDirsOnly))


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    ex = LearningWidget()
    ex.show()

    app.exec_()
    app.deleteLater()
    sys.exit()