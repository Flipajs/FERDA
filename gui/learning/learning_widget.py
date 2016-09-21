from PyQt4 import QtGui, QtCore
import sys
import os
from core.project.project import Project
from utils.img_manager import ImgManager
from core.learning.learning_process import LearningProcess
from core.settings import Settings as S_
import numpy as np


class LearningWidget(QtGui.QWidget):
    def __init__(self, project=None, show_tracklet_callback=None):
        super(LearningWidget, self).__init__()

        self.project = project
        self.show_tracklet_callback = show_tracklet_callback
        self.vbox = QtGui.QVBoxLayout()
        self.hbox = QtGui.QHBoxLayout()
        self.top_stripe_layout = QtGui.QHBoxLayout()
        self.setLayout(self.vbox)

        self.vbox.addLayout(self.top_stripe_layout)
        self.vbox.addLayout(self.hbox)

        # TODO: get features...

        self.lp = None
        if not self.project:
            self.load_project_button = QtGui.QPushButton('load project')
            self.load_project_button.clicked.connect(self.load_project)
            self.top_stripe_layout.addWidget(self.load_project_button)
        else:
            print "LOADING LP"
            self.lp = LearningProcess(self.project, use_feature_cache=True, use_rf_cache=False,
                                      question_callback=self.question_callback, update_callback=self.update_callback)

        self.start_button = QtGui.QPushButton('start')
        self.start_button.clicked.connect(self.lp.run_learning)
        self.top_stripe_layout.addWidget(self.start_button)

        self.info_table = QtGui.QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setRowCount(12)

        self.info_table.setMinimumHeight(500)
        self.info_table.setFixedWidth(220)
        self.hbox.addWidget(self.info_table)

        self.next_step_button = QtGui.QPushButton('next step')
        self.next_step_button.clicked.connect(self.lp.next_step)
        self.top_stripe_layout.addWidget(self.next_step_button)

        self.label_certainty_eps = QtGui.QLabel('certainty eps:')
        self.top_stripe_layout.addWidget(self.label_certainty_eps)

        self.certainty_eps_spinbox = QtGui.QDoubleSpinBox()
        self.certainty_eps_spinbox.setSingleStep(0.01)
        self.certainty_eps_spinbox.setValue(0.3)
        self.certainty_eps_spinbox.setMaximum(1)
        self.certainty_eps_spinbox.setMinimum(0)

        self.certainty_eps_spinbox.valueChanged.connect(self.certainty_eps_changed)
        self.top_stripe_layout.addWidget(self.certainty_eps_spinbox)

        self.num_next_step = QtGui.QLineEdit()
        self.num_next_step.setText('10')
        self.top_stripe_layout.addWidget(self.num_next_step)
        self.n_next_steps_button = QtGui.QPushButton('do N steps')
        self.n_next_steps_button.clicked.connect(self.do_n_steps)
        self.top_stripe_layout.addWidget(self.n_next_steps_button)

        self.show_tracklet_button = QtGui.QPushButton('show selected tracklet')
        self.show_tracklet_button.clicked.connect(self.show_tracklet)
        self.top_stripe_layout.addWidget(self.show_tracklet_button)

        self.reset_learning_button = QtGui.QPushButton('reset learning')
        self.reset_learning_button.clicked.connect(self.reset_learning)
        self.top_stripe_layout.addWidget(self.reset_learning_button)

        self.save_button = QtGui.QPushButton('save')
        self.save_button.clicked.connect(self.save)
        self.top_stripe_layout.addWidget(self.save_button)

        # TODO: last info label
        # TODO: update callback... info about decisions...

        self.add_tracklet_table()
        self.update_callback()

    def add_tracklet_table(self):
        self.tracklets_table = QtGui.QTableWidget()

        self.tracklets_table.setRowCount(len(self.lp.undecided_tracklets))
        num_animals = len(self.project.animals)
        self.tracklets_table.setColumnCount(2 * num_animals + 5)
        self.tracklets_table.setMinimumWidth(1000)
        self.tracklets_table.setMinimumHeight(1000)

        self.tracklets_table.setSortingEnabled(True)
        self.hbox.addWidget(self.tracklets_table)


    def certainty_eps_changed(self):
        self.lp.eps_certainty = self.certainty_eps_spinbox.value()

    def save(self):
        self.lp.save_learning()
        self.project.save()

        print "SAVED"

    def reset_learning(self):
        self.lp.reset_learning()
        self.update_callback()

    def show_tracklet(self):
        indexes = self.tracklets_table.selectionModel().selectedRows()
        indexes = sorted(indexes)

        if len(indexes):
            # pick first
            row = indexes[0].row()
            id_ = int(self.tracklets_table.item(row, 0).text())
            tracklet = self.project.chm[id_]
            self.show_tracklet_callback(tracklet)

    def do_n_steps(self):
        try:
            num = int(self.num_next_step.text())
        except:
            QtGui.QMessageBox('not a valid number!')

        for i in range(num):
            if not self.lp.next_step():
                break

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

        self.info_table.setItem(10, 0, QtGui.QTableWidgetItem('# user decisions: '))
        self.info_table.setItem(10, 1, QtGui.QTableWidgetItem(str(len(self.lp.user_decisions))))

        self.info_table.setItem(11, 0, QtGui.QTableWidgetItem('id coverage:'))
        self.info_table.setItem(11, 1, QtGui.QTableWidgetItem(self.__f2str(self.get_id_coverage())))


        # update tracklet info...
        self.tracklets_table.clear()
        self.tracklets_table.setRowCount(len(self.lp.undecided_tracklets))

        num_animals = len(self.project.animals)
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
                if j in t.P:
                    val = 'N'
                elif j in t.N:
                    val = 'P'

                self.tracklets_table.setItem(i, 5+num_animals+j, QtGui.QTableWidgetItem(val))

        self.tracklets_table.setSortingEnabled(True)
        self.tracklets_table.resizeColumnsToContents()

    def test_one_id_in_tracklet(self, t):
        return len(t.P) == 1 and \
               len(t.N) == len(self.project.animals) - 1

    def get_id_coverage(self):
        from utils.video_manager import get_auto_video_manager
        vm = get_auto_video_manager(self.project)

        frames = vm.total_frame_count()

        coverage = 0
        for t in self.project.chm.chunk_gen():
            if self.test_one_id_in_tracklet(t):
                coverage += t.length()

        return coverage / float(frames*len(self.project.animals))

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
        self.show_tracklet_callback(tracklet)

    def decide_tracklet_question(self, tracklet):
        items = map(str, range(len(self.project.animals)))

        item, ok = QtGui.QInputDialog.getItem(self, "select animal ID for tracklet ID: "+str(tracklet.id()),
                                              "list of ids", items, 0, False)
        if ok:
            self.lp.assign_identity(int(item), tracklet, user=True)
            self.update_callback()
        else:
            print "..."


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