__author__ = 'fnaiser'

import os.path
import sys
import pickle

from PyQt4 import QtGui

from core.project.project import Project


class ImportWidget(QtGui.QWidget):
    def __init__(self):
        super(ImportWidget, self).__init__()

        self.file_paths = {}

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.group_box = QtGui.QGroupBox('Import')
        self.group_box.setLayout(QtGui.QVBoxLayout())
        self.vbox.addWidget(self.group_box)

        self.form = QtGui.QFormLayout()

        self.choose_dir = QtGui.QPushButton('folder import')
        self.choose_dir.clicked.connect(self.choose_dir_clicked)

        self.import_button = QtGui.QPushButton('import checked')
        self.import_button.clicked.connect(self.import_clicked)

        self.group_box.layout().addWidget(self.choose_dir)

        self.keys = ['bg_model', 'arena_model', 'classes', 'groups', 'animals']
        self.checkboxes = {}

        for k in self.keys:
            self.checkboxes[k] = QtGui.QCheckBox(k)
            self.checkboxes[k].setDisabled(True)
            b = QtGui.QPushButton('import')
            b.clicked.connect(lambda: self.imp(self.classes_ch, 'classes'))

            self.form.addRow(b, self.checkboxes[k])

        self.group_box.layout().addLayout(self.form)
        self.group_box.layout().addWidget(self.import_button)



    def choose_dir_clicked(self):
        dir = str(QtGui.QFileDialog.getExistingDirectory(self, "Select import directory"))
        if os.path.isdir(dir):
            for k in self.keys:
                f = dir+k+'.pkl'
                if os.path.isfile(f):
                    self.checkboxes[k].setDisabled(False)
                    self.checkboxes[k].setChecked(True)
                    self.file_paths[k] = f

        else:
            QtGui.QMessageBox.warning(self, 'Wrong', 'No folder was selected')
            self.choose_dir_clicked()

    def import_clicked(self):
        pass

    def imp(self, checkbox, key):
        f = str(QtGui.QFileDialog.getOpenFileName(self, '', '', filter='Pickle (*.pkl)'))
        self.file_paths[key] = f

        if os.path.isfile(f):
            checkbox.setDisabled(False)
            checkbox.setChecked(True)


    def finish_import(self, project):
        for k in self.file_paths:
            if self.checkboxes[k].isChecked():
                f = self.file_paths[k]
                val = pickle.load(f)
                eval('project.%s = val'% (k))


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    proj = Project()

    ex = ImportWidget(proj)
    ex.show()

    app.exec_()
    app.deleteLater()
    sys.exit()
