import sys

import numpy as np
import os
import warnings
from PyQt6 import QtCore, QtGui, QtWidgets

from core.id_detection.learning_process import LearningProcess
from core.project.project import Project
from gui.qt_flow_layout import FlowLayout
from gui.settings import Settings as S_
from utils.img_manager import ImgManager


try:
    QString = unicode
except NameError:
    # Python 3
    QString = str

class QCustomTableWidgetItem (QtWidgets.QTableWidgetItem):
    def __init__ (self, value=''):
        super(QCustomTableWidgetItem, self).__init__(QString('%s' % value))

    def __lt__ (self, other):
        if (isinstance(other, QCustomTableWidgetItem)):
            try:
                s1 = self.data(QtCore.Qt.EditRole).toString()
                s1 = s1[:-1] if s1[-1] == '%' else s1

                s2 = other.data(QtCore.Qt.EditRole).toString()
                s2 = s2[:-1] if s2[-1] == '%' else s2
                selfDataValue  = float(s1)
                otherDataValue = float(s2)
                return selfDataValue < otherDataValue
            except:
                return self.data(QtCore.Qt.EditRole).toString() < other.data(QtCore.Qt.EditRole).toString()
        else:
            return QtWidgets.QTableWidgetItem.__lt__(self, other)

class QCustomTableWidget (QtWidgets.QTableWidget):
    def __init__ (self, parent=None):
        super(QCustomTableWidget, self).__init__(parent)
        # self.setColumnCount(2)
        # self.setRowCount(5)
        # for row in range(self.rowCount()):
        #     self.setItem(row, 0, QCustomTableWidgetItem(random.random() * 1e4))
        #     self.setItem(row, 1, QtGui.QTableWidgetItem(QtCore.QString(65 + row)))
        self.setSortingEnabled(True)


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


class LearningWidget(QtWidgets.QWidget):
    def __init__(self, project=None, show_tracklet_callback=None, progressbar_callback=None):
        super(LearningWidget, self).__init__()

        self.project = project
        self.progressbar_callback = progressbar_callback
        self.show_tracklet_callback = show_tracklet_callback
        self.vbox = QtWidgets.QVBoxLayout()
        self.hbox = QtWidgets.QHBoxLayout()
        self.top_stripe_layout = FlowLayout()
        self.setLayout(self.vbox)

        self.vbox.addLayout(self.top_stripe_layout)
        self.vbox.addLayout(self.hbox)

        self.lp = None
        if not self.project:
            self.load_project_button = QtWidgets.QPushButton('load project')
            self.load_project_button.clicked.connect(self.load_project)
            self.top_stripe_layout.addWidget(self.load_project_button)
        else:
            self.lp = LearningProcess(self.project, ghost=False, progressbar_callback=progressbar_callback)

        self.start_button = QtWidgets.QPushButton('start')
        self.start_button.clicked.connect(self.lp.run_learning)
        # self.top_stripe_layout.addWidget(self.start_button)

        self.info_table = QCustomTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setRowCount(13)

        self.info_table.setMinimumHeight(500)
        self.info_table.setFixedWidth(220)
        self.hbox.addWidget(self.info_table)

        self.next_step_button = QtWidgets.QPushButton('next step')
        # self.lp will change...
        self.next_step_button.clicked.connect(lambda x: self.lp.next_step())
        # self.top_stripe_layout.addWidget(self.next_step_button)

        # self.top_stripe_layout.addWidget(QtGui.QLabel('min examples to retrain:'))

        self.min_examples_to_retrain_i = QtWidgets.QLineEdit()
        if self.lp is not None:
            self.min_examples_to_retrain_i.setText(str(self.lp.min_new_samples_to_retrain))
        self.min_examples_to_retrain_i.adjustSize()

        self._filter = Filter(self.min_examples_to_retrain_i, self.lp.set_min_new_samples_to_retrain)
        self.min_examples_to_retrain_i.installEventFilter(self._filter)
        # self.top_stripe_layout.addWidget(self.min_examples_to_retrain_i)
        #
        # self.load_classifier_b = QtGui.QPushButton('load classifier')
        # self.load_classifier_b.clicked.connect(self.load_classifier)
        # self.top_stripe_layout.addWidget(self.load_classifier_b)

        self.load_features_b = QtWidgets.QPushButton('load features')
        self.load_features_b.clicked.connect(self.load_features)
        self.top_stripe_layout.addWidget(self.load_features_b)

        self.compute_features_b = QtWidgets.QPushButton('compute features')
        self.compute_features_b.clicked.connect(self.recompute_features)
        self.top_stripe_layout.addWidget(self.compute_features_b)

        self.auto_init_method_cb = QtWidgets.QComboBox()
        self.auto_init_method_cb.addItem("max min")
        self.auto_init_method_cb.addItem("max sum")
        self.top_stripe_layout.addWidget(self.auto_init_method_cb)

        self.auto_init_b = QtWidgets.QPushButton('auto_init')
        self.auto_init_b.clicked.connect(self.auto_init)
        self.top_stripe_layout.addWidget(self.auto_init_b)

        self.reset_learning_button = QtWidgets.QPushButton('learn/restart classifier')
        self.reset_learning_button.clicked.connect(self.reset_learning)
        self.top_stripe_layout.addWidget(self.reset_learning_button)

        self.prepare_unassigned_cs_b = QtWidgets.QPushButton('prepare unassigned CS')
        self.prepare_unassigned_cs_b.clicked.connect(self.prepare_unassigned_cs)
        self.top_stripe_layout.addWidget(self.prepare_unassigned_cs_b)

        self.label_tracklet_min_length = QtWidgets.QLabel('tracklet min len: ')
        self.top_stripe_layout.addWidget(self.label_tracklet_min_length)

        self.tracklet_min_length_sb = QtWidgets.QSpinBox()
        self.tracklet_min_length_sb.setValue(20)
        self.tracklet_min_length_sb.setMinimum(0)
        self.tracklet_min_length_sb.setMaximum(10000)
        self.top_stripe_layout.addWidget(self.tracklet_min_length_sb)
        self.update_tracklet_len_b = QtWidgets.QPushButton('apply')
        self.update_tracklet_len_b.clicked.connect(self.tracklet_min_length_changed)
        self.top_stripe_layout.addWidget(self.update_tracklet_len_b)

        self.label_certainty_eps = QtWidgets.QLabel('certainty eps:')
        self.top_stripe_layout.addWidget(self.label_certainty_eps)

        self.certainty_eps_spinbox = QtWidgets.QDoubleSpinBox()
        self.certainty_eps_spinbox.setSingleStep(0.01)
        self.certainty_eps_spinbox.setValue(0.7)
        self.certainty_eps_spinbox.setMaximum(1)
        self.certainty_eps_spinbox.setMinimum(0)

        self.certainty_eps_spinbox.valueChanged.connect(self.certainty_eps_changed)
        self.top_stripe_layout.addWidget(self.certainty_eps_spinbox)

        self.num_next_step = QtWidgets.QLineEdit()
        self.num_next_step.setText('10')
        self.top_stripe_layout.addWidget(self.num_next_step)
        self.n_next_steps_button = QtWidgets.QPushButton('do N steps')
        self.n_next_steps_button.clicked.connect(self.do_n_steps)
        self.top_stripe_layout.addWidget(self.n_next_steps_button)

        self.show_tracklet_button = QtWidgets.QPushButton('show selected tracklet')
        self.show_tracklet_button.clicked.connect(self.show_tracklet)
        self.top_stripe_layout.addWidget(self.show_tracklet_button)


        self.save_button = QtWidgets.QPushButton('save')
        self.save_button.clicked.connect(self.save)
        self.top_stripe_layout.addWidget(self.save_button)

        # TODO: last info label
        # TODO: update callback... info about decisions...

        self.update_b = QtWidgets.QPushButton('update table')
        self.update_b.clicked.connect(self.update_callback)
        self.top_stripe_layout.addWidget(self.update_b)

        self.delete_user_decisions_b = QtWidgets.QPushButton('delete user decisions')
        self.delete_user_decisions_b.clicked.connect(self.clear_user_decisions)
        self.top_stripe_layout.addWidget(self.delete_user_decisions_b)

        self.use_xgboost_ch = QtWidgets.QCheckBox("use XGBoost")
        self.use_xgboost_ch.setChecked(False)
        self.top_stripe_layout.addWidget(self.use_xgboost_ch)

        self.use_idcr_ch = QtWidgets.QCheckBox("use IDCR")
        self.use_idcr_ch.setChecked(True)
        self.use_idcr_ch.stateChanged.connect(self.use_idcr_update)
        self.top_stripe_layout.addWidget(self.use_idcr_ch)

        self.show_init_summary_b = QtWidgets.QPushButton('show init summary')
        self.show_init_summary_b.clicked.connect(self.show_init_summary)
        self.top_stripe_layout.addWidget(self.show_init_summary_b)

        self.compute_distinguishability_b = QtWidgets.QPushButton('debug: comp. disting.')
        # self.lp will change...
        self.compute_distinguishability_b.clicked.connect(lambda x: self.lp.compute_distinguishability())
        self.top_stripe_layout.addWidget(self.compute_distinguishability_b)

        self.update_undecided_tracklets_b = QtWidgets.QPushButton('debug: update undecided')
        self.update_undecided_tracklets_b.clicked.connect(self.update_undecided_tracklets)
        self.top_stripe_layout.addWidget(self.update_undecided_tracklets_b)

        self.tracklet_debug_info_b = QtWidgets.QPushButton("debug: tracklet info")
        self.tracklet_debug_info_b.clicked.connect(self.tracklet_debug_info)
        self.top_stripe_layout.addWidget(self.tracklet_debug_info_b)

        self.lp.set_tracklet_length_k(self.tracklet_min_length_sb.value(), first_time=True)
        self.add_tracklet_table()
        self.update_callback()

        # self.enable_all()

    def enable_all(self):
        self.auto_init_method_cb.setEnabled(True)
        self.auto_init_b.setEnabled(True)
        self.tracklet_min_length_sb.setEnabled(True)
        self.update_tracklet_len_b.setEnabled(True)
        self.certainty_eps_spinbox.setEnabled(True)
        self.num_next_step.setEnabled(True)
        self.n_next_steps_button.setEnabled(True)
        self.show_tracklet_button.setEnabled(True)
        self.reset_learning_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.update_b.setEnabled(True)
        self.delete_user_decisions_b.setEnabled(True)
        self.compute_distinguishability_b.setEnabled(True)
        self.use_xgboost_ch.setEnabled(True)
        self.update_undecided_tracklets_b.setEnabled(True)
        self.tracklet_debug_info_b.setEnabled(True)

        if not os.path.isfile(self.project.working_directory+'/fm.sqlite3'):
            self.load_features_b.setDisabled(True)
        else:
            self.compute_features_b.setDisabled(True)

    def disable_before_classifier(self):
        self.auto_init_method_cb.setEnabled(True)
        self.auto_init_b.setEnabled(True)
        self.tracklet_min_length_sb.setEnabled(True)
        self.update_tracklet_len_b.setEnabled(True)
        self.certainty_eps_spinbox.setEnabled(True)
        self.num_next_step.setEnabled(False)
        self.n_next_steps_button.setEnabled(False)
        self.show_tracklet_button.setEnabled(False)
        self.reset_learning_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.update_b.setEnabled(False)
        self.delete_user_decisions_b.setEnabled(True)
        self.compute_distinguishability_b.setEnabled(False)
        self.use_xgboost_ch.setEnabled(True)
        self.update_undecided_tracklets_b.setEnabled(False)
        self.tracklet_debug_info_b.setEnabled(False)

        if not os.path.isfile(self.project.working_directory+'/fm.sqlite3'):
            self.load_features_b.setDisabled(True)
        else:
            self.compute_features_b.setDisabled(True)

    def disable_before_features(self):
        self.auto_init_method_cb.setEnabled(False)
        self.auto_init_b.setEnabled(False)
        self.tracklet_min_length_sb.setEnabled(False)
        self.update_tracklet_len_b.setEnabled(False)
        self.certainty_eps_spinbox.setEnabled(False)
        self.num_next_step.setEnabled(False)
        self.n_next_steps_button.setEnabled(False)
        self.show_tracklet_button.setEnabled(False)
        self.reset_learning_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.update_b.setEnabled(False)
        self.delete_user_decisions_b.setEnabled(False)
        self.compute_distinguishability_b.setEnabled(False)
        self.use_xgboost_ch.setEnabled(False)
        self.update_undecided_tracklets_b.setEnabled(False)
        self.tracklet_debug_info_b.setEnabled(False)

        if not os.path.isfile(self.project.working_directory+'/fm.sqlite3'):
            self.load_features_b.setDisabled(True)
        else:
            self.compute_features_b.setDisabled(True)

    def load_features(self):
        path = 'fm.sqlite3'
        self.lp.load_features(path)

        self.add_tracklet_table()
        self.update_callback()

        self.lp.update_callback = self.update_callback
        self.lp.question_callback = self.question_callback

        # self.disable_before_classifier()

    def recompute_features(self):
        # self.lp = LearningProcess(self.project, use_feature_cache=False, use_rf_cache=False,
        #                           question_callback=self.question_callback, update_callback=self.update_callback)

        self.lp.compute_features()
        self.lp.update_callback = self.update_callback
        self.lp.question_callback = self.question_callback
        self.min_examples_to_retrain_i.setText(str(self.lp.min_new_samples_to_retrain))

        self.add_tracklet_table()
        self.load_features()
        self.update_callback()

    def add_tracklet_table(self):
        if not hasattr(self, 'tracklets_table') or self.tracklets_table is None:
            self.tracklets_table = QtWidgets.QTableWidget()

            self.tracklets_table.setRowCount(len(self.lp.undecided_tracklets))
            num_animals = len(self.project.animals)
            self.tracklets_table.setColumnCount(num_animals + 5)
            self.tracklets_table.setMinimumWidth(1000)
            self.tracklets_table.setMinimumHeight(900)

            self.tracklets_table.setSortingEnabled(True)
            self.hbox.addWidget(self.tracklets_table)

    def certainty_eps_changed(self):
        self.lp.set_eps_certainty(1-self.certainty_eps_spinbox.value())

    def tracklet_min_length_changed(self):
        self.lp.set_tracklet_length_k(self.tracklet_min_length_sb.value())
        self.update_callback()

    def save(self):
        self.lp.save_learning()
        self.project.save()

        print("SAVED")

    def reset_learning(self):
        self.lp.reset_learning(use_xgboost=self.use_xgboost_ch.isChecked())
        self.lp.save_learning()
        self.project.save()
        self.enable_all()
        self.update_callback()


    def prepare_unassigned_cs(self):
        self.lp.prepare_unassigned_cs()
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

    def tracklet_debug_info(self):
        indexes = self.tracklets_table.selectionModel().selectedRows()
        indexes = sorted(indexes)

        if len(indexes):
            # pick first
            row = indexes[0].row()
            id_ = int(self.tracklets_table.item(row, 0).text())
            tracklet = self.project.chm[id_]

            self.lp._get_tracklet_proba(tracklet, debug=True)

    def do_n_steps(self):
        from utils.misc import print_progress
        try:
            num = int(self.num_next_step.text())
        except:
            QtWidgets.QMessageBox('not a valid number!')

        for i in range(num):
            print_progress(i, num, "deciding {} most certain tracklets".format(num))
            if not self.lp.next_step(update_gui=False):
                break

        print_progress(num, num, "deciding {} most certain tracklets".format(num), "DONE")

        self.update_callback()
        print((i, "steps finished"))

    def update_callback(self):
        self.info_table.setItem(0, 0, QCustomTableWidgetItem('#tracklets'))
        if self.project.chm is not None:
            num_tracklets = str(len(self.project.chm))
        else:
            num_tracklets = '-'
        self.info_table.setItem(0, 1, QCustomTableWidgetItem())

        self.info_table.setItem(1, 0, QCustomTableWidgetItem('#collision tracklets'))
        self.info_table.setItem(1, 1, QCustomTableWidgetItem(str(len(self.lp.collision_chunks))))

        self.info_table.setItem(2, 0, QCustomTableWidgetItem('#undecided'))
        self.info_table.setItem(2, 1, QCustomTableWidgetItem(str(len(self.lp.undecided_tracklets))))

        self.info_table.setItem(3, 0, QCustomTableWidgetItem('#new T examples'))
        self.info_table.setItem(3, 1, QCustomTableWidgetItem(str(len(self.lp.X) - self.lp.old_x_size)))

        id_representants = self.get_id_representants()

        START = 4
        for i in range(len(self.project.animals)):
            s = 0
            for tid in id_representants[i]:
                if tid is not None:
                    s += len(self.project.chm[tid])

            self.info_table.setItem(START+i, 0, QCustomTableWidgetItem('#examples, ID: '+str(i)))
            self.info_table.setItem(START+i, 1, QCustomTableWidgetItem(str(s)))

        self.info_table.setItem(10, 0, QCustomTableWidgetItem('# user decisions: '))
        self.info_table.setItem(10, 1, QCustomTableWidgetItem(str(len(self.lp.user_decisions))))

        full_coverage, single_id_coverage = 0, 0
        # TODO: speed up
        # full_coverage, single_id_coverage = self.get_id_coverage()
        self.info_table.setItem(11, 0, QCustomTableWidgetItem('full coverage:'))
        self.info_table.setItem(11, 1, QCustomTableWidgetItem(self.__f2str(full_coverage)))

        self.info_table.setItem(12, 0, QCustomTableWidgetItem('single-ID coverage:'))
        self.info_table.setItem(12, 1, QCustomTableWidgetItem(self.__f2str(single_id_coverage)))

        if hasattr(self, 'tracklets_table'):
            # update tracklet info...
            self.tracklets_table.clear()
            self.tracklets_table.setRowCount(len(self.lp.undecided_tracklets))

            num_animals = len(self.project.animals)
            self.tracklets_table.setSortingEnabled(False)
            header_labels = ("id", "len", "start", "end", "cert")
            for i in range(num_animals):
                header_labels += ('ID'+str(i), )

            # for i in range(num_animals):
            #     header_labels += (str(i), )

            it = QCustomTableWidgetItem
            from tqdm import tqdm

            self.tracklets_table.setHorizontalHeaderLabels(header_labels)
            if len(self.lp.tracklet_certainty):
                for i, t_id in tqdm(enumerate(self.lp.undecided_tracklets)):
                    if t_id not in self.lp.tracklet_certainty:
                        warnings.warn("tracklet id not in lp.tracklet_certainty")
                        continue
                        # self.lp._update_certainty(self.project.chm[t_id])

                    t = self.project.chm[t_id]

                    item = it()
                    item.setData(QtCore.Qt.EditRole, t.id())
                    self.tracklets_table.setItem(i, 0, item)

                    item = it()
                    item.setData(QtCore.Qt.EditRole, t.length())
                    self.tracklets_table.setItem(i, 1, item)

                    item = it()
                    item.setData(QtCore.Qt.EditRole, t.start_frame())
                    self.tracklets_table.setItem(i, 2, item)

                    item = it()
                    item.setData(QtCore.Qt.EditRole, t.end_frame())
                    self.tracklets_table.setItem(i, 3, item)

                    self.tracklets_table.setItem(i, 4, QCustomTableWidgetItem(self.__f2str(self.lp.tracklet_certainty[t_id])))

                    id_ = None
                    try:
                        if len(self.project.chm[t_id]) > 10:
                            ids = self.lp.GT.tracklet_id_set_without_checks(self.project.chm[t_id], self.project)

                        if len(ids) == 1:
                            id_ = ids[0]
                    except:
                        pass

                    d = np.mean(self.lp.tracklet_measurements[t_id], axis=0)
                    stds = self.lp.tracklet_stds[t_id]
                    for j in range(num_animals):
                        self.tracklets_table.setItem(i, 5+j, QCustomTableWidgetItem(self.__f2str(d[j])+", {:.2f}".format(stds[j])))

                        b = 255 if j == np.argmax(d) else 0
                        r = 255 if j in t.N else 0
                        g = 255 if j == id_ else 0

                        # use white...
                        if r == g == b == 0:
                            r, g, b = 255, 255, 255

                        if r == g == 0 and b == 255:
                            g = 102

                        self.tracklets_table.item(i, 5 + j).setBackgroundColor(QtGui.QColor(r, g, b))

            self.tracklets_table.setSortingEnabled(True)
            self.tracklets_table.resizeColumnsToContents()

    def test_one_id_in_tracklet(self, t):
        return len(t.P) == 1 and \
               len(t.N) == len(self.project.animals) - 1

    def get_id_coverage(self):
        coverage = 0
        max_ = 0
        single_id_sum = 0
        for t in self.project.chm.chunk_gen():
            if t.is_single():
                single_id_sum += len(t)

            if self.test_one_id_in_tracklet(t):
                coverage += t.length()

            end_f_ = t.end_frame()
            max_ = max(max_, end_f_)

        full_coverage = coverage / float(max_*len(self.project.animals))
        single_id_coverage = coverage / float(single_id_sum)
        return full_coverage, single_id_coverage

    def __f2str(self, f, prec=1):
        return ('{:.'+str(prec)+'%}').format(f)

    def load_project(self, default=''):
        path = ''
        if os.path.isdir(S_.temp.last_wd_path):
            path = S_.temp.last_wd_path

        working_directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select working directory", path, QtWidgets.QFileDialog.ShowDirsOnly))
        # TODO: load project...
        # self.project = ...
        # TODO: use_feature_cache...
        self.lp = LearningProcess(self.project, question_callback=self.show_tracklet_callback, use_feature_cache=True,
                                  use_rf_cache=False, progressbar_callback=self.progressbar_callback)

    def question_callback(self, tracklet):
        self.show_tracklet_callback(tracklet)

    def edit_tracklet(self, tracklet):
        from gui.learning.edit_tracklet_advanced import EditTrackletAdvanced
        w = EditTrackletAdvanced(tracklet, len(self.lp.all_ids), self.lp.edit_tracklet)
        w.show()

    def decide_tracklet_question(self, tracklet, id_=None):
        if id_ is None:
            items = list(map(str, self.lp.all_ids - tracklet.N))
            items = sorted(items)

            item, ok = QtWidgets.QInputDialog.getItem(self, "select animal ID for tracklet ID: "+str(tracklet.id()),
                                                  "list of ids", items, 0, False)
        else:
            ok = True
            item = id_

        if ok:
            self.lp.assign_identity(int(item), tracklet, user=True)
            self.update_callback()
        else:
            print("...")

    def clear_user_decisions(self):
        msg = "Do you really want to delete all USERs decisions?"
        reply = QtWidgets.QMessageBox.question(self, 'Message',
                                           msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            self.lp.user_decisions = []

        self.update_callback()

    def update_undecided_tracklets(self):
        print("UPDATING UNDECIDED")
        self.lp.update_undecided_tracklets()

    def auto_init(self):
        method = 'maxmin'
        if self.auto_init_method_cb.currentIndex == 0:
            method = 'maxsum'

        self.lp.auto_init(method=method, use_xgboost=self.use_xgboost_ch.isChecked())

        self.update_callback()

    def get_separated_frame(self):
        return self.lp.separated_frame

    def get_id_representants(self):
        id_representants = {}
        for i in range(len(self.project.animals)):
            id_representants[i] = []

        for d in self.lp.user_decisions:
            if d['type'] == 'P':
                tid = d['tracklet_id_set']
                id_representants[d['ids'][0]].append(tid)

        return id_representants

    def show_init_summary(self):
        from tqdm import trange
        from core.id_detection.features import get_colornames_hists_saturated
        import random
        num_examples = 20
        from utils.video_manager import get_auto_video_manager
        vm = get_auto_video_manager(self.project)

        # print id_representants
        id_representants = self.get_id_representants()

        print("EXAMPLES: ")
        for i in range(len(self.project.animals)):
            s = 0
            for tid in id_representants[i]:
                s += len(self.project.chm[tid])

            print(("\tA_ID: {}, len sum: {}, tids: {}".format(i, s, id_representants[i])))

        region_representants = {}
        img_representants = {}

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        plt.ion()
        major_ticks = np.arange(0, 11, 1)
        num_animals = len(self.project.animals)
        for aid in trange(num_animals):
            region_representants[aid] = []
            img_representants[aid] = []

            probs = []
            for i in range(num_examples):
                tid = id_representants[aid][i % len(id_representants[aid])]

                t = self.project.chm[tid]
                id_ = t[random.randint(0, len(t)-1)]
                r = self.project.gm.region(id_)

                f = get_colornames_hists_saturated(r, self.project)
                probs.append(f)

                region_representants[aid].append(r)

                im = draw_region(self.project, vm, id_)

                # h = 0.05
                # for j, val in enumerate(f):
                #     im[-min(im.shape[0]-1, int(h*val)):, j*3:(j+1)*3, :] = [0, 0, 255]

                img_representants[aid].append(im)

            probs = np.array(probs)
            ind = np.arange(probs.shape[1])
            width = 1.
            ax.bar(ind + (width/num_animals)/2 + (aid * (width/num_animals)), np.mean(probs, 0), width/(num_animals+1), yerr=np.std(probs, 0, ddof=1))
            # ax.bar(ind + width, np.mean(wprobs, 0), width, yerr=np.std(wprobs, 0, ddof=1))
        # ax.set_xticks(major_ticks)
        plt.xticks(major_ticks, ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'])
        plt.grid()
        plt.show()

        from gui.img_grid.img_grid_widget import ImgGridWidget
        w = ImgGridWidget(cols=len(self.project.animals), element_width=100)

        HH = 100
        WW = 100
        for i in range(num_examples):
            for aid in range(len(self.project.animals)):
                item = make_item(img_representants[aid][i], region_representants[aid][i], HH, WW)
                w.add_item(item)

        win = QtWidgets.QMainWindow()
        win.setCentralWidget(w)
        win.show()
        self.w = win

    def update_N_sets(self):
        self.project.chm.update_N_sets(self.project)

        self.lp.force=False

    def tracklet_measurements(self, id_):
        try:
            return self.lp.tracklet_measurements[id_]
        except:
            return None

    def use_idcr_update(self):
        self.lp.id_N_propagate = self.use_idcr_ch.isChecked()
        self.lp.id_N_f = self.use_idcr_ch.isChecked()

def draw_region(p, vm, v):
    from utils.img import img_saturation_coef
    # from utils.drawing.points import draw_points
    r1 = p.gm.region(v)
    im1 = vm.get_frame(r1.frame()).copy()
    # c1 = QtGui.QColor(255, 0, 0, 255)
    # draw_points(im1, r1.contour(), color=c1)
    roi = r1.roi().safe_expand(30, im1)
    im = im1[roi.slices()].copy()
    # im = img_saturation_coef(im, 2.0, 1.05)
    im = img_saturation_coef(im, 1.5, 0.95)

    return im

def make_item(im, id_, HH, WW):
    from PyQt6 import QtGui, QtWidgets
    from gui.gui_utils import SelectableQLabel
    from PIL import ImageQt
    im_ = np.zeros((max(im.shape[0], HH), max(im.shape[1], WW), 3), dtype=np.uint8)
    im_[:im.shape[0], :im.shape[1], :] = im
    im = im_

    img_q = ImageQt.QImage(im.data, im.shape[1], im.shape[0], im.shape[1] * 3, 13)
    pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

    item = SelectableQLabel(id=id_)

    item.setScaledContents(True)
    if im.shape[0] > HH or im.shape[1] > WW:
        item.setFixedSize(HH, WW)

    item.setPixmap(pix_map)

    return item

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    p = Project()
    p.load('/Users/flipajs/Documents/wd/GT/Cam1 copy/cam1.fproj')
    p.img_manager = ImgManager(p)

    ex = LearningWidget(project=p)
    ex.show()

    app.exec_()
    app.deleteLater()
    sys.exit()
