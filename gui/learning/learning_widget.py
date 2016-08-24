from PyQt4 import QtGui, QtCore
import sys
import os
from core.project.project import Project
from utils.img_manager import ImgManager
from core.learning.learning_process import LearningProcess
from core.settings import Settings as S_
import numpy as np


class LearningWidget(QtGui.QWidget):
    def __init__(self, project=None):
        super(LearningWidget, self).__init__()

        self.project = project
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.lp = None
        if not self.project:
            self.load_project_button = QtGui.QPushButton('load project')
            self.load_project_button.clicked.connect(self.load_project)
            self.vbox.addWidget(self.load_project_button)
        else:
            self.lp = LearningProcess(self.project, use_feature_cache=True, use_rf_cache=False,
                                      question_callback=self.question_callback, update_callback=self.update_callback)

        self.start_button = QtGui.QPushButton('start')
        self.start_button.clicked.connect(self.lp.run_learning)

        self.info_table = QtGui.QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setRowCount(10)

        self.info_table.setMinimumHeight(500)
        self.vbox.addWidget(self.info_table)

        # TODO: next step
        # TODO: next N steps
        # TODO: print info...
        # TODO: callback from learningProcess on change to update info

        # TODO: step by step

        # TODO: update callback... info about decisions...

        self.vbox.addWidget(self.start_button)

        self.tracklets_table = QtGui.QTableWidget()
        self.tracklets_table.setRowCount(self.lp.undecided_tracklets)
        num_animals = len(self.project.animals)
        self.tracklets_table.setColumnCount(2 * num_animals + 2)

        self.update_callback()

    def update_callback(self):
        self.info_table.setItem(0, 0, QtGui.QTableWidgetItem('#tracklets'))
        self.info_table.setItem(0, 1, QtGui.QTableWidgetItem(str(len(self.project.chm))))

        self.info_table.setItem(1, 0, QtGui.QTableWidgetItem('#collision tracklets'))
        self.info_table.setItem(1, 1, QtGui.QTableWidgetItem(str(len(self.lp.collision_chunks))))

        self.info_table.setItem(2, 0, QtGui.QTableWidgetItem('#undecided'))
        self.info_table.setItem(2, 1, QtGui.QTableWidgetItem(str(len(self.lp.undecided_tracklets))))

        self.info_table.setItem(3, 0, QtGui.QTableWidgetItem('#new T examples'))
        self.info_table.setItem(3, 0, QtGui.QTableWidgetItem(str(len(self.lp.X) - self.lp.old_x_size)))

        START = 4
        for i in range(len(self.project.animals)):
            self.info_table.setItem(START+i, 0, QtGui.QTableWidgetItem('#examples, ID: '+str(i)))
            self.info_table.setItem(START+i, 1, QtGui.QTableWidgetItem(str(np.count_nonzero(self.lp.y == i))))


        # update chunk infos...


    def load_project(self, default=''):
        path = ''
        if os.path.isdir(S_.temp.last_wd_path):
            path = S_.temp.last_wd_path

        working_directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select working directory", path, QtGui.QFileDialog.ShowDirsOnly))
        # TODO: load project...
        # self.project = ...
        # TODO: use_feature_cache...
        self.lp = LearningProcess(self.project, use_feature_cache=True, use_rf_cache=False)

    def question_callback(self, tracklet):
        items = map(str, range(len(self.project.animals)))

        item, ok = QtGui.QInputDialog.getItem(self, "select animal ID for tracklet ID: "+str(tracklet.id()),
                                              "list of ids", items, 0, False)

        return int(item)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    p = Project()
    p.load('/Users/flipajs/Documents/wd/GT/Cam1 copy/cam1.fproj')
    p.img_manager = ImgManager(p)

    ex = LearningWidget(project=p)
    ex.show()

    app.exec_()
    app.deleteLater()
    sys.exit()