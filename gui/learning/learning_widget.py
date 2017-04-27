from PyQt4 import QtGui, QtCore
import sys
import os
from core.project.project import Project
from utils.img_manager import ImgManager
from core.id_detection.learning_process import LearningProcess
from core.settings import Settings as S_
import numpy as np
from gui.qt_flow_layout import FlowLayout


class Filter(QtCore.QObject):
    def __init__(self, line_edit, callback):
        super(Filter, self).__init__()
        self.line_edit = line_edit
        self.callback = callback
    
    def eventFilter(self, widget, event):
        # FocusOut event
        if event.type() == QtCore.QEvent.FocusOut:
            # do custom stuff
            self.callback(int(self.line_edit.text()))
            # return False so that the widget will also handle the event
            # otherwise it won't focus out
            return False
        else:
            # we don't care about other events
            return False

class LearningWidget(QtGui.QWidget):
    def __init__(self, project=None, show_tracklet_callback=None):
        super(LearningWidget, self).__init__()

        self.project = project
        self.show_tracklet_callback = show_tracklet_callback
        self.vbox = QtGui.QVBoxLayout()
        self.hbox = QtGui.QHBoxLayout()
        self.top_stripe_layout = FlowLayout()
        self.setLayout(self.vbox)

        self.vbox.addLayout(self.top_stripe_layout)
        self.vbox.addLayout(self.hbox)

        self.lp = None
        if not self.project:
            self.load_project_button = QtGui.QPushButton('load project')
            self.load_project_button.clicked.connect(self.load_project)
            self.top_stripe_layout.addWidget(self.load_project_button)
        else:
            self.lp = LearningProcess(self.project, ghost=False)

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
        # self.lp will change...
        self.next_step_button.clicked.connect(lambda x: self.lp.next_step())
        self.top_stripe_layout.addWidget(self.next_step_button)

        self.top_stripe_layout.addWidget(QtGui.QLabel('min examples to retrain:'))

        self.min_examples_to_retrain_i = QtGui.QLineEdit()
        if self.lp is not None:
            self.min_examples_to_retrain_i.setText(str(self.lp.min_new_samples_to_retrain))
        self.min_examples_to_retrain_i.adjustSize()

        self._filter = Filter(self.min_examples_to_retrain_i, self.lp.set_min_new_samples_to_retrain)
        self.min_examples_to_retrain_i.installEventFilter(self._filter)
        self.top_stripe_layout.addWidget(self.min_examples_to_retrain_i)

        self.label_tracklet_min_length = QtGui.QLabel('tracklet min len: ')
        self.top_stripe_layout.addWidget(self.label_tracklet_min_length)

        self.tracklet_min_length_sb = QtGui.QSpinBox()
        self.tracklet_min_length_sb.setValue(50)
        self.tracklet_min_length_sb.setMinimum(0)
        self.tracklet_min_length_sb.setMaximum(10000)
        self.top_stripe_layout.addWidget(self.tracklet_min_length_sb)
        self.update_tracklet_len_b = QtGui.QPushButton('update')
        self.update_tracklet_len_b.clicked.connect(self.tracklet_min_length_changed)
        self.top_stripe_layout.addWidget(self.update_tracklet_len_b)

        self.label_certainty_eps = QtGui.QLabel('certainty eps:')
        self.top_stripe_layout.addWidget(self.label_certainty_eps)

        self.certainty_eps_spinbox = QtGui.QDoubleSpinBox()
        self.certainty_eps_spinbox.setSingleStep(0.01)
        self.certainty_eps_spinbox.setValue(0.7)
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

        self.load_features_b = QtGui.QPushButton('load features')
        self.load_features_b.clicked.connect(self.load_features)
        self.top_stripe_layout.addWidget(self.load_features_b)

        self.compute_features_b = QtGui.QPushButton('compute features')
        self.compute_features_b.clicked.connect(self.recompute_features)
        self.top_stripe_layout.addWidget(self.compute_features_b)

        self.save_button = QtGui.QPushButton('save')
        self.save_button.clicked.connect(self.save)
        self.top_stripe_layout.addWidget(self.save_button)

        # TODO: last info label
        # TODO: update callback... info about decisions...

        self.update_b = QtGui.QPushButton('update')
        self.update_b.clicked.connect(self.update_callback)
        self.top_stripe_layout.addWidget(self.update_b)

        self.delete_user_decisions_b = QtGui.QPushButton('delete user decisions')
        self.delete_user_decisions_b.clicked.connect(self.clear_user_decisions)
        self.top_stripe_layout.addWidget(self.delete_user_decisions_b)

        self.update_undecided_tracklets_b = QtGui.QPushButton('debug: update undecided')
        self.update_undecided_tracklets_b.clicked.connect(self.update_undecided_tracklets)
        self.top_stripe_layout.addWidget(self.update_undecided_tracklets_b)

        self.compute_distinguishability_b = QtGui.QPushButton('comp. disting.')
        # self.lp will change...
        self.compute_distinguishability_b.clicked.connect(lambda x: self.lp.compute_distinguishability())
        self.top_stripe_layout.addWidget(self.compute_distinguishability_b)

        self.auto_init_b = QtGui.QPushButton('auto_init')
        self.auto_init_b.clicked.connect(self.auto_init)
        self.top_stripe_layout.addWidget(self.auto_init_b)

        self.auto_init_method_cb = QtGui.QComboBox()
        self.auto_init_method_cb.addItem("max min")
        self.auto_init_method_cb.addItem("max sum")
        self.top_stripe_layout.addWidget(self.auto_init_method_cb)

        # self.add_tracklet_table()
        # self.update_callback()

    def load_features(self):
        path = 'fm.sqlite3'
        self.lp.load_features(path)

        self.add_tracklet_table()
        self.update_callback()

        self.lp.update_callback = self.update_callback
        self.lp.question_callback = self.question_callback

        # self.lp = LearningProcess(self.project, use_feature_cache=True, use_rf_cache=True,
        #                           question_callback=self.question_callback, update_callback=self.update_callback)
        #
        # self.min_examples_to_retrain_i.setText(str(self.lp.min_new_samples_to_retrain))
        #
        # self.add_tracklet_table()
        # self.update_callback()

    def recompute_features(self):
        # self.lp = LearningProcess(self.project, use_feature_cache=False, use_rf_cache=False,
        #                           question_callback=self.question_callback, update_callback=self.update_callback)

        self.lp.compute_features()
        self.lp.update_callback = self.update_callback
        self.lp.question_callback = self.question_callback
        self.min_examples_to_retrain_i.setText(str(self.lp.min_new_samples_to_retrain))

        self.add_tracklet_table()
        self.update_callback()

    def add_tracklet_table(self):
        if not hasattr(self, 'tracklets_table') or self.tracklets_table is None:
            self.tracklets_table = QtGui.QTableWidget()

            self.tracklets_table.setRowCount(len(self.lp.undecided_tracklets))
            num_animals = len(self.project.animals)
            self.tracklets_table.setColumnCount(num_animals + 5)
            self.tracklets_table.setMinimumWidth(1000)
            self.tracklets_table.setMinimumHeight(1000)

            self.tracklets_table.setSortingEnabled(True)
            self.hbox.addWidget(self.tracklets_table)

    def certainty_eps_changed(self):
        self.lp.set_eps_certainty(1-self.certainty_eps_spinbox.value())

    def tracklet_min_length_changed(self):
        self.lp.set_tracklet_length_k(self.tracklet_min_length_sb.value())

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
        from utils.misc import print_progress
        try:
            num = int(self.num_next_step.text())
        except:
            QtGui.QMessageBox('not a valid number!')

        for i in range(num):
            print_progress(i, num, "deciding {} most certain tracklets".format(num))
            if not self.lp.next_step(update_gui=False):
                break

        print_progress(num, num, "deciding {} most certain tracklets".format(num), "DONE")

        self.update_callback()
        print i, "steps finished"

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

        if hasattr(self, 'tracklets_table'):
            # update tracklet info...
            self.tracklets_table.clear()
            self.tracklets_table.setRowCount(len(self.lp.undecided_tracklets))

            num_animals = len(self.project.animals)
            self.tracklets_table.setSortingEnabled(False)
            header_labels = ("id", "len", "start", "end", "cert")
            for i in range(num_animals):
                header_labels += ('m'+str(i), )

            # for i in range(num_animals):
            #     header_labels += (str(i), )

            it = QtGui.QTableWidgetItem

            self.tracklets_table.setHorizontalHeaderLabels(header_labels)
            if len(self.lp.tracklet_certainty):
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

                        if j in t.N:
                            self.tracklets_table.item(i, 5+j).setBackgroundColor(QtGui.QColor(150, 150, 150))

                    # for j in range(num_animals):
                    #     val = ''
                    #     if j in t.P:
                    #         val = 'N'
                    #     elif j in t.N:
                    #         val = 'P'
                    #
                    #     self.tracklets_table.setItem(i, 5+num_animals+j, QtGui.QTableWidgetItem(val))

            self.tracklets_table.setSortingEnabled(True)
            self.tracklets_table.resizeColumnsToContents()

    def test_one_id_in_tracklet(self, t):
        return len(t.P) == 1 and \
               len(t.N) == len(self.project.animals) - 1

    def get_id_coverage(self):
        from utils.video_manager import get_auto_video_manager
        vm = get_auto_video_manager(self.project)

        coverage = 0
        max_ = 0
        for t in self.project.chm.chunk_gen():
            if self.test_one_id_in_tracklet(t):
                coverage += t.length()

            end_f_ = t.end_frame(self.project.gm)
            max_ = max(max_, end_f_)

        return coverage / float(max_*len(self.project.animals))

    def __f2str(self, f, prec=1):
        return ('{:.'+str(prec)+'%}').format(f)

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

    def edit_tracklet(self, tracklet):
        from gui.learning.edit_tracklet_advanced import EditTrackletAdvanced
        w = EditTrackletAdvanced(tracklet, len(self.lp.all_ids), self.lp.edit_tracklet)
        w.show()

    def decide_tracklet_question(self, tracklet, id_=None):
        if id_ is None:
            items = map(str, self.lp.all_ids - tracklet.N)
            items = sorted(items)

            item, ok = QtGui.QInputDialog.getItem(self, "select animal ID for tracklet ID: "+str(tracklet.id()),
                                                  "list of ids", items, 0, False)
        else:
            ok = True
            item = id_

        if ok:
            self.lp.assign_identity(int(item), tracklet, user=True)
            self.update_callback()
        else:
            print "..."

    def clear_user_decisions(self):
        msg = "Do you really want to delete all USERs decisions?"
        reply = QtGui.QMessageBox.question(self, 'Message',
                                           msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

        if reply == QtGui.QMessageBox.Yes:
            self.lp.user_decisions = []

        self.update_callback()

    def update_undecided_tracklets(self):
        print "UPDATING UNDECIDED"
        self.lp.update_undecided_tracklets()

    def auto_init(self):
        method = 'maxmin'
        if self.auto_init_method_cb.currentIndex == 0:
            method = 'maxsum'

        self.lp.auto_init(method=method)

        self.update_callback()

    def get_separated_frame(self):
        return self.lp.separated_frame

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