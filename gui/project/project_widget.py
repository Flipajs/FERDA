__author__ = 'fnaiser'

import sys

from PyQt4 import QtGui

import core.project
import utils.gui
import utils.misc


class ProjectWidget(QtGui.QWidget):
    def __init__(self, finish_callback=None):
        super(ProjectWidget, self).__init__()
        self.layout = QtGui.QVBoxLayout(self)
        self.setLayout(self.layout)
        self.finish_callback = finish_callback

        self.new_project_button = QtGui.QPushButton('New Project', self)
        self.layout.addWidget(self.new_project_button)
        self.new_project_button.clicked.connect(self.new_project)

        self.load_project_button = QtGui.QPushButton('LoadProject', self)
        self.layout.addWidget(self.load_project_button)
        self.load_project_button.clicked.connect(self.load_project)

        self.update()
        self.show()

    def new_project(self):
        if self.finish_callback:
            self.finish_callback('new_project')

    def load_project(self):
        files = utils.gui.file_names_dialog(self, 'Select FERDA project', '*.fproj')
        if len(files) == 1:
            f = files[0]
            project = core.project.Project()
            project.load(f)

            if self.finish_callback:
                self.finish_callback('load_project', project)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    ex = ProjectWidget()

    app.exec_()
    app.deleteLater()
    sys.exit()