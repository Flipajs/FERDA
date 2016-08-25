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
        self.hbox = QtGui.QHBoxLayout()
        self.top_stripe_layout = QtGui.QHBoxLayout()
        self.setLayout(self.vbox)

        self.vbox.addLayout(self.top_stripe_layout)
        self.vbox.addLayout(self.hbox)

        self.lp = None
        if not self.project:
            self.load_project_button = QtGui.QPushButton('load project')
            self.load_project_button.clicked.connect(self.load_project)
            self.top_stripe_layout.addWidget(self.load_project_button)
        else:
            self.lp = LearningProcess(self.project, use_feature_cache=True, use_rf_cache=False,
                                      question_callback=self.question_callback, update_callback=self.update_callback)

        self.start_button = QtGui.QPushButton('start')
        self.start_button.clicked.connect(self.lp.run_learning)
        self.top_stripe_layout.addWidget(self.start_button)

        self.info_table = QtGui.QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setRowCount(10)

        self.info_table.setMinimumHeight(500)
        self.info_table.setFixedWidth(220)
        self.hbox.addWidget(self.info_table)

        self.next_step_button = QtGui.QPushButton('next step')
        self.next_step_button.clicked.connect(self.lp.next_step)
        self.top_stripe_layout.addWidget(self.next_step_button)

        self.num_next_step = QtGui.QLineEdit()
        self.num_next_step.setText('10')
        self.top_stripe_layout.addWidget(self.num_next_step)
        self.n_next_steps_button = QtGui.QPushButton('do N steps')
        self.n_next_steps_button.clicked.connect(self.do_n_steps)
        self.top_stripe_layout.addWidget(self.n_next_steps_button)

        # TODO: last info label
        # TODO: integrate into main...
        # TODO: add option to add info...
        # TODO: add option to reset learning
        # TODO: add saving
        # TODO: update callback... info about decisions...

        self.tracklets_table = QtGui.QTableWidget()
        self.tracklets_table.setRowCount(len(self.lp.undecided_tracklets))
        num_animals = len(self.project.animals)
        self.tracklets_table.setColumnCount(2 * num_animals + 5)
        self.tracklets_table.setMinimumWidth(1000)
        self.tracklets_table.setMinimumHeight(1000)

        self.tracklets_table.setSortingEnabled(True)
        self.hbox.addWidget(self.tracklets_table)

        self.update_callback()

    def do_n_steps(self):
        try:
            num = int(self.num_next_step.text())
        except:
            QtGui.QMessageBox('not a valid number!')

        for i in range(num):
            self.lp.next_step()

    def update_callback(self):
        self.info_table.setItem(0, 0, QtGui.QTableWidgetItem('#tracklets'))
        self.info_table.setItem(0, 1, QtGui.QTableWidgetItem(str(len(self.project.chm))))

        self.info_table.setItem(1, 0, QtGui.QTableWidgetItem('#collision tracklets'))
        self.info_table.setItem(1, 1, QtGui.QTableWidgetItem(str(len(self.lp.collision_chunks))))

        self.info_table.setItem(2, 0, QtGui.QTableWidgetItem('#undecided'))
        self.info_table.setItem(2, 1, QtGui.QTableWidgetItem(str(len(self.lp.undecided_tracklets))))

        self.info_table.setItem(3, 0, QtGui.QTableWidgetItem('#new T examples'))
        self.info_table.setItem(3, 1, QtGui.QTableWidgetItem(str(len(self.lp.X) - self.lp.old_x_size)))

        START = 4
        for i in range(len(self.project.animals)):
            self.info_table.setItem(START+i, 0, QtGui.QTableWidgetItem('#examples, ID: '+str(i)))
            self.info_table.setItem(START+i, 1, QtGui.QTableWidgetItem(str(np.count_nonzero(self.lp.y == i))))

        # update tracklet info...
        num_animals = len(self.project.animals)
        self.tracklets_table.clear()
        self.tracklets_table.setSortingEnabled(False)
        header_labels = ("id", "len", "start", "end", "cert")
        for i in range(num_animals):
            header_labels += ('m'+str(i), )

        for i in range(num_animals):
            header_labels += (str(i), )

        it = QtGui.QTableWidgetItem

        self.tracklets_table.setHorizontalHeaderLabels(header_labels)
        for i, t_id in enumerate(self.lp.undecided_tracklets):
            t = self.project.chm[t_id]

            item = it()
            item.setData(QtCore.Qt.EditRole, t.id())
            self.tracklets_table.setItem(i, 0, item)

            item = it()
            item.setData(QtCore.Qt.EditRole, t.length())
            self.tracklets_table.setItem(i, 1, item)

            item = it()
            item.setData(QtCore.Qt.EditRole, t.start_frame(self.project.gm))
            self.tracklets_table.setItem(i, 2, item)

            item = it()
            item.setData(QtCore.Qt.EditRole, t.end_frame(self.project.gm))
            self.tracklets_table.setItem(i, 3, item)

            self.tracklets_table.setItem(i, 4, QtGui.QTableWidgetItem(self.__f2str(self.lp.tracklet_certainty[t_id])))

            d = self.lp.tracklet_measurements[t_id]
            for j in range(num_animals):
                self.tracklets_table.setItem(i, 5+j, QtGui.QTableWidgetItem(self.__f2str(d[j])))

            for j in range(num_animals):
                val = ''
                if j in self.lp.ids_not_present_in_tracklet[t_id]:
                    val = 'N'
                elif j in self.lp.ids_present_in_tracklet[t_id]:
                    val = 'P'

                self.tracklets_table.setItem(i, 5+num_animals+j, QtGui.QTableWidgetItem(val))

        self.tracklets_table.setSortingEnabled(True)
        self.tracklets_table.resizeColumnsToContents()

    def __f2str(self, f, prec=3):
        return ('{:.'+str(prec)+'}').format(f)

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