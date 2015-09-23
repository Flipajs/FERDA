__author__ = 'fnaiser'


from gui.correction.noise_filter_computer import NoiseFilterComputer
from gui.gui_utils import get_img_qlabel
from utils.video_manager import get_auto_video_manager
import sys
from PyQt4 import QtGui, QtCore
import numpy as np
import pickle
from functools import partial
from core.region.fitting import Fitting
from copy import deepcopy
from case_widget import CaseWidget
from new_region_widget import NewRegionWidget
from core.region.region import Region
from core.log import LogCategories, ActionNames
from gui.img_grid.img_grid_widget import ImgGridWidget
from core.settings import Settings as S_
import math
from gui.view.graph_visualizer import call_visualizer
from gui.loading_widget import LoadingWidget


class ConfigurationsVisualizer(QtGui.QWidget):
    def __init__(self, solver, vid):
        super(ConfigurationsVisualizer, self).__init__()
        self.setLayout(QtGui.QVBoxLayout())
        self.scenes_widget = QtGui.QWidget()
        self.scenes_widget.setLayout(QtGui.QVBoxLayout())
        self.scenes_widget.layout().setContentsMargins(0, 0, 0, 0)

        self.noise_nodes_widget = None
        self.progress_bar = None

        self.scroll_ = QtGui.QScrollArea()
        self.scroll_.setWidgetResizable(True)
        self.scroll_.setWidget(self.scenes_widget)
        self.layout().addWidget(self.scroll_)

        self.solver = solver
        self.project = solver.project
        self.vid = vid
        self.cws = []
        self.ccs = []
        self.cws_sorted = False

        # nodes with undecided edges
        self.nodes = []
        # index of active node in self.nodes
        self.active_node_id = -1
        # img cache indexed by frame
        self.img_cache = {}
        # node visualization cache indexed by node ref
        self.node_vis_cache = {}

        self.t1_nodes_cc_refs = {}
        self.t2_nodes_cc_refs = {}

        self.active_cw = None

        self.add_actions()

        self.cc_number_label = QtGui.QLabel('')
        self.layout().addWidget(self.cc_number_label)

        self.autosave_timer = QtCore.QTimer()
        self.autosave_timer.timeout.connect(partial(self.solver.save, True))
        # TODO: add interval to settings
        # self.autosave_timer.start(1000*60*10)
        self.order_by = 'time'

        self.join_regions_active = False
        self.join_regions_n1 = None

        self.layout().setContentsMargins(0, 0, 0, 0)

        self.order_by_sb = None
        self.tool_w = self.create_tool_w()

    def create_tool_w(self):
        w = QtGui.QWidget()
        w.setLayout(QtGui.QHBoxLayout())

        self.order_by_sb = QtGui.QComboBox()
        self.order_by_sb.addItem('frame')
        self.order_by_sb.addItem('chunk length')
        self.order_by_sb.currentIndexChanged.connect(self.next_case)
        w.layout().addWidget(QtGui.QLabel('order by: '))
        w.layout().addWidget(self.order_by_sb)

        return w

    def set_nodes_queue(self, nodes):
        for n in nodes:
            if n.frame_ != self.solver.start_t and n.frame_ != self.solver.end_t:
                self.nodes.append(n)

        self.nodes = sorted(self.nodes, key=lambda k: k.frame_)
        self.active_node_id = 0

    def new_region(self, t_offset=-1):
        if t_offset < 0:
            t_offset = self.active_cw.active_col
            print t_offset
            if t_offset < 0:
                return

        cw = self.active_cw
        im = cw.crop_pixmaps_cache[t_offset]

        w = NewRegionWidget(im, cw.crop_offset, cw.frame_t + t_offset, self.new_region_finished)
        self.d_ = QtGui.QDialog()
        self.d_.setLayout(QtGui.QVBoxLayout())
        self.d_.layout().addWidget(w)
        self.d_.setFixedWidth(500)
        self.d_.setFixedHeight(500)
        self.d_.show()
        self.d_.exec_()

    def new_region_finished(self, confirmed, data):
        self.d_.close()
        if confirmed:
            r = Region()
            r.pts_ = data['pts']
            r.centroid_ = data['centroid']
            r.frame_ = data['frame']
            r.is_virtual = True
            #TODO: get rid of this hack... also in antlikness test in solver.py
            # flag for virtual region
            r.min_intensity_ = -2

            self.project.log.add(LogCategories.USER_ACTION, ActionNames.NEW_REGION, r)
            self.solver.add_virtual_region(r)
            self.next_case()

    def remove_region(self, node=None, suppress_next_case=False):
        if not node:
            if not self.active_cw.active_node:
                return
            node = self.active_cw.active_node

        self.project.log.add(LogCategories.USER_ACTION, ActionNames.REMOVE, node)
        self.solver.remove_region(node)
        if not suppress_next_case:
            self.next_case()

    def strong_remove_region(self, n=None, suppress_next_case=False):
        if not n:
            n = self.active_cw.active_node

        if not n:
            return

        self.project.log.add(LogCategories.USER_ACTION, ActionNames.STRONG_REMOVE, n)
        self.solver.strong_remove(n)
        if not suppress_next_case:
            self.next_case()

    def choose_node(self, pos):
        cw = self.active_cw
        if pos >= cw.num_of_nodes:
            return

        cw.dehighlight_node()
        self.active_cw.highlight_node(cw.get_node_at_pos(pos))

        if self.join_regions_active:
            self.join_regions(self.join_regions_n1, self.active_cw.active_node)
            self.join_regions_active = False
            self.join_regions_n1 = None

    def chunk_len_(self, n):
        is_ch, t_reversed, ch = self.solver.is_chunk(n)
        if is_ch:
            return ch.length()

        return 0

    def order_nodes(self):
        if self.order_by_sb.currentText() == 'chunk length':
            self.nodes = sorted(self.nodes, key=lambda k: -self.chunk_len_(k))
        else:
            self.nodes = sorted(self.nodes, key=lambda k: k.frame_)

    def next_case(self, move_to_different_case=False):
        if move_to_different_case:
            self.active_node_id += 1

        self.nodes = self.solver.g.nodes()
        self.order_nodes()

        if self.active_node_id < len(self.nodes):
            n = self.nodes[self.active_node_id]
            if n in self.solver.ignored_nodes:
                self.active_node_id += 1
                self.next_case()
                return

            # test end
            if n.frame_ == self.solver.end_t:
                self.active_node_id += 1
                self.next_case()
                return

            # test beginning
            if n.frame_ == 0:
                is_ch, _, _ = self.solver.is_chunk(n)
                if is_ch:
                    self.active_node_id += 1
                    self.next_case()
                    return

            # test if it is different cc:
            if move_to_different_case and self.active_cw:
                for g in self.active_cw.nodes_groups:
                    for n_ in g:
                        if n == n_:
                            self.next_case(move_to_different_case)
                            return

            # remove previous case (if exists)
            if self.scenes_widget.layout().count():
                it = self.scenes_widget.layout().itemAt(0)
                self.scenes_widget.layout().removeItem(it)
                it.widget().setParent(None)

            # add new widget
            nodes_groups = self.solver.get_cc_from_node(n)
            if len(nodes_groups) == 0:
                print n, "empty nodes groups"
                # self.nodes.pop(self.active_node_id)
                self.active_node_id += 1
                self.next_case()
                return

            # nodes_groups = []
            # for i in range(100):
            #     nodes_groups.append([])
            #
            # nodes_ = self.solver.g.nodes()
            # nodes_ = sorted(nodes_, key=lambda k: k.frame_)
            # while nodes_:
            #     n = nodes_.pop(0)
            #     nodes_groups[n.frame_].append(n)

            config = self.best_greedy_config(nodes_groups)

            self.active_cw = CaseWidget(self.solver.g, self.project, nodes_groups, config, self.vid, self)
            self.scenes_widget.layout().addWidget(self.active_cw)

    def best_greedy_config(self, nodes_groups):
        config = {}
        for i in range(len(nodes_groups) - 1):
            r1 = list(nodes_groups[i])
            r2 = list(nodes_groups[i+1])

            while r1 and r2:
                changed = False
                values = []
                for n1 in r1:
                    for n2 in r2:
                        try:
                            s = self.solver.g[n1][n2]['score']
                            values.append([s, n1, n2])
                            changed = True
                        except:
                            pass

                if not changed:
                    break

                values = sorted(values, key=lambda k: k[0])
                r1.remove(values[0][1])
                r2.remove(values[0][2])
                config[values[0][1]] = values[0][2]

        return config

    def prev_case(self):
        self.active_node_id -= 1

        self.nodes = self.solver.g.nodes()
        self.order_nodes()

        n = self.nodes[self.active_node_id]

        if n in self.solver.ignored_nodes:
            self.active_node_id -= 1
            self.prev_case()
            return

        if n.frame_ == self.solver.end_t:
            self.active_node_id -= 1
            self.prev_case()
            return

        # test beginning
        if n.frame_ == 0:
            is_ch, _, _ = self.solver.is_chunk(n)
            if is_ch:
                self.active_node_id -= 1
                self.prev_case()
                return

        # test if it is different cc:
        if self.active_cw:
            for g in self.active_cw.nodes_groups:
                for n_ in g:
                    if n == n_:
                        self.prev_case()
                        return

        # remove previous case (if exists)
        if self.scenes_widget.layout().count():
            it = self.scenes_widget.layout().itemAt(0)
            self.scenes_widget.layout().removeItem(it)
            it.widget().setParent(None)

        # add new widget
        nodes_groups = self.solver.get_cc_from_node(n)
        if len(nodes_groups) == 0:
            # self.nodes.pop(self.active_node_id)
            self.active_node_id -= 1
            self.prev_case()
            return

        config = self.best_greedy_config(nodes_groups)

        self.active_cw = CaseWidget(self.solver.g, self.project, nodes_groups, config, self.vid, self)
        self.active_cw.active_node = None
        self.scenes_widget.layout().addWidget(self.active_cw)

    def confirm_cc(self):
        self.active_cw.confirm_clicked()

    def fitting_get_model(self, t_reversed):
        region = self.active_cw.active_node

        merged_t = region.frame_ - self.active_cw.frame_t
        model_t = merged_t + 1 if t_reversed else merged_t - 1

        if len(self.active_cw.nodes_groups[model_t]) > 0 and len(self.active_cw.nodes_groups[merged_t]) > 0:
            t1_ = self.active_cw.nodes_groups[model_t]

            objects = []
            for c1 in t1_:
                a = deepcopy(c1)
                if t_reversed:
                    a.frame_ -= 1
                else:
                    a.frame_ += 1

                objects.append(a)

        return objects

    def fitting(self, t_reversed=False, one_step=False):
        if self.active_cw.active_node:
            self.project.log.add(LogCategories.USER_ACTION,
                                 ActionNames.MERGED_SELECTED,
                                 {
                                     'n': self.active_cw.active_node,
                                     't_reversed': t_reversed
                                 })

            is_ch, _, chunk = self.solver.is_chunk(self.active_cw.active_node)
            if is_ch:
                print "FITTING WHOLE CHUNK, WAIT PLEASE"
                model = None

                q = chunk.last if t_reversed else chunk.first
                i = 0
                while True:
                    merged = q()
                    if not merged:
                        break

                    # TODO: find better way
                    if chunk.length() < 3:
                        chunk.pop_last(self.solver) if t_reversed else chunk.pop_first(self.solver)

                    # TODO: settings, safe break
                    if i > 15:
                        break

                    if not model:
                        model = self.fitting_get_model(t_reversed)

                    # TODO : add to settings
                    f = Fitting(merged, model, num_of_iterations=10)
                    f.fit()

                    self.solver.merged(f.animals, merged, t_reversed)

                    model = deepcopy(f.animals)
                    for m in model:
                        m.frame_ += -1 if t_reversed else 1

                    if one_step:
                        break

                    i += 1

                print "CHUNK FINISHED"
            else:
                model = self.fitting_get_model(t_reversed)
                f = Fitting(self.active_cw.active_node, model, num_of_iterations=10)
                f.fit()

                self.solver.merged(f.animals, self.active_cw.active_node, t_reversed)

            self.next_case()

    def partially_confirm(self):
        if self.active_cw.active_node:
            cw = self.active_cw

            n1 = self.active_cw.active_node
            conf = cw.suggested_config

            pairs = []
            for _, n1_, n2_ in conf:
                if n1_ == n1:
                    pairs.append((n1, n2_))

                if n2_ == n1:
                    pairs.append((n1_, n1))

            self.confirm_edges(pairs)

    def path_confirm(self):
        print "PATH CONFIRM"
        n = self.active_cw.active_node
        if n:
            cw = self.active_cw
            conf = cw.suggested_config

            edges = []

            print "WHILE"
            while True:
                finish = True
                for _, n1, n2 in conf:
                    if n1 == n:
                        edges.append((n1, n2))
                        n = n2
                        finish = False
                        break

                if finish:
                    break

            print "END"

            self.confirm_edges(edges)

    def confirm_edges(self, pairs):
        self.project.log.add(LogCategories.USER_ACTION, ActionNames.CONFIRM, {'pairs': pairs})
        self.solver.confirm_edges(pairs)
        self.next_case()

    def join_regions_pick_second(self):
        self.active_cw.join_with_()

    def join_regions(self, n1, n2):
        if n1.area() < n2.area():
            n1, n2 = n2, n1

        self.project.log.add(LogCategories.USER_ACTION, ActionNames.JOIN_REGIONS, {'n1': n1, 'n2': n2})

        # TODO: update also other moments etc...
        n_new = deepcopy(n1)
        n_new.pts_ = np.concatenate((n_new.pts_, n2.pts_), 0)
        n_new.centroid_ = np.mean(n_new.pts_, 0)
        self.solver.remove_region(n1)
        self.solver.remove_region(n2)
        self.solver.add_virtual_region(n_new)
        self.next_case()

    def noise_part_done(self, val, img, n):
        elem_width = 200
        self.progress_bar.setValue(val)
        item = get_img_qlabel(n.pts(), img, n, elem_width, elem_width, filled=True)
        item.set_selected(True)
        self.noise_nodes_widget.add_item(item)

    def noise_finished(self):
        self.progress_bar.setParent(None)
        self.noise_nodes_confirm_b.show()
        self.noise_nodes_back_b.show()

    def noise_nodes_filter(self):
        if self.scenes_widget.layout().count():
            it = self.scenes_widget.layout().itemAt(0)
            self.scenes_widget.layout().removeItem(it)
            it.widget().setParent(None)

        elem_width = 200
        cols = math.floor(self.scenes_widget.width() / elem_width)
        self.noise_nodes_widget = ImgGridWidget()
        self.noise_nodes_widget.reshape(cols, elem_width)

        steps = 100

        self.progress_bar = QtGui.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.scenes_widget.layout().addWidget(self.progress_bar)
        self.scenes_widget.layout().addWidget(self.noise_nodes_widget)

        self.thread = NoiseFilterComputer(self.solver, self.project, steps)
        self.thread.part_done.connect(self.noise_part_done)
        self.thread.proc_done.connect(self.noise_finished)
        self.thread.set_range.connect(self.progress_bar.setMaximum)
        self.thread.start()

    def remove_noise(self):
        # TODO: add actions

        to_remove = self.noise_nodes_widget.get_selected()
        for n in to_remove:
            if n in self.solver.g:
                self.strong_remove_region(n, suppress_next_case=True)

        to_confirm = self.noise_nodes_widget.get_unselected()
        for n in to_confirm:
            if n in self.solver.g:
                self.solver.g.node[n]['antlikeness'] = 1.0

        self.mode_tools_noise.hide()
        self.next_case()

    def remove_noise_back(self):
        self.mode_tools_noise.hide()
        self.next_case()

    def ignore_node(self):
        self.project.log.add(LogCategories.USER_ACTION, ActionNames.IGNORE_NODES)
        for g in self.active_cw.nodes_groups:
            for n in g:
                self.solver.ignored_nodes[n] = True
                self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.IGNORE_NODE, n)

        self.next_case()

    def clear_scenew_widget(self):
        if self.scenes_widget.layout().count():
            it = self.scenes_widget.layout().itemAt(0)
            self.scenes_widget.layout().removeItem(it)
            it.widget().setParent(None)

    def show_global_view(self):
        self.clear_scenew_widget()

        w_loading = LoadingWidget()
        self.scenes_widget.layout().addWidget(w_loading)
        QtGui.QApplication.processEvents()

        start_t = self.gv_start_t.value()
        end_t = self.gv_end_t.value()
        min_chunk_len = self.gv_chunk_len_threshold.value()
        w = call_visualizer(start_t, end_t, self.project, self.solver, min_chunk_len, w_loading.update_progress)
        self.clear_scenew_widget()
        self.scenes_widget.layout().addWidget(w)

    def add_actions(self):
        self.next_action = QtGui.QAction('next', self)
        self.next_action.triggered.connect(partial(self.next_case, True))
        self.next_action.setShortcut(S_.controls.next_case)
        self.addAction(self.next_action)

        self.prev_action = QtGui.QAction('prev', self)
        self.prev_action.triggered.connect(self.prev_case)
        self.prev_action.setShortcut(S_.controls.prev_case)
        self.addAction(self.prev_action)

        self.confirm_cc_action = QtGui.QAction('confirm', self)
        self.confirm_cc_action.triggered.connect(self.confirm_cc)
        self.confirm_cc_action.setShortcut(S_.controls.confirm)
        self.addAction(self.confirm_cc_action)

        self.partially_confirm_action = QtGui.QAction('partially confirm', self)
        self.partially_confirm_action.triggered.connect(self.partially_confirm)
        self.partially_confirm_action.setShortcut(S_.controls.partially_confirm)
        self.addAction(self.partially_confirm_action)

        self.path_confirm_action = QtGui.QAction('path confirm', self)
        self.path_confirm_action.triggered.connect(self.path_confirm)
        self.path_confirm_action.setShortcut(S_.controls.confirm_path)
        self.addAction(self.path_confirm_action)

        self.fitting_action = QtGui.QAction('fitting', self)
        self.fitting_action.triggered.connect(partial(self.fitting, False))
        self.fitting_action.setShortcut(S_.controls.fitting_from_left)
        self.addAction(self.fitting_action)

        self.fitting_rev_action = QtGui.QAction('fitting rev', self)
        self.fitting_rev_action.triggered.connect(partial(self.fitting, True))
        self.fitting_rev_action.setShortcut(S_.controls.fitting_from_right)
        self.addAction(self.fitting_rev_action)

        # self.new_region_t1_action = QtGui.QAction('new region t1', self)
        # self.new_region_t1_action.triggered.connect(partial(self.new_region, 0))
        # self.new_region_t1_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Q))
        # self.addAction(self.new_region_t1_action)
        #
        # self.new_region_t2_action = QtGui.QAction('new region t2', self)
        # self.new_region_t2_action.triggered.connect(partial(self.new_region, 1))
        # self.new_region_t2_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_W))
        # self.addAction(self.new_region_t2_action)

        self.remove_region_action = QtGui.QAction('remove region', self)
        self.remove_region_action.triggered.connect(self.remove_region)
        self.remove_region_action.setShortcut(S_.controls.remove_region)
        self.addAction(self.remove_region_action)

        self.strong_remove_action = QtGui.QAction('strong remove', self)
        self.strong_remove_action.triggered.connect(self.strong_remove_region)
        self.strong_remove_action.setShortcut(S_.controls.remove_chunk)
        self.addAction(self.strong_remove_action)

        self.join_regions_action = QtGui.QAction('join regions', self)
        self.join_regions_action.triggered.connect(self.join_regions_pick_second)
        self.join_regions_action.setShortcut(S_.controls.join_regions)
        self.addAction(self.join_regions_action)

        self.action0 = QtGui.QAction('0', self)
        self.action0.triggered.connect(partial(self.choose_node, 9))
        self.action0.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_0))
        self.addAction(self.action0)

        self.action1 = QtGui.QAction('1', self)
        self.action1.triggered.connect(partial(self.choose_node, 0))
        self.action1.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_1))
        self.addAction(self.action1)

        self.action2 = QtGui.QAction('2', self)
        self.action2.triggered.connect(partial(self.choose_node, 1))
        self.action2.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_2))
        self.addAction(self.action2)

        self.action3 = QtGui.QAction('3', self)
        self.action3.triggered.connect(partial(self.choose_node, 2))
        self.action3.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_3))
        self.addAction(self.action3)

        self.action4 = QtGui.QAction('4', self)
        self.action4.triggered.connect(partial(self.choose_node, 3))
        self.action4.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_4))
        self.addAction(self.action4)

        self.action5 = QtGui.QAction('5', self)
        self.action5.triggered.connect(partial(self.choose_node, 4))
        self.action5.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_5))
        self.addAction(self.action5)

        self.action6 = QtGui.QAction('6', self)
        self.action6.triggered.connect(partial(self.choose_node, 5))
        self.action6.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_6))
        self.addAction(self.action6)

        self.action7 = QtGui.QAction('7', self)
        self.action7.triggered.connect(partial(self.choose_node, 6))
        self.action7.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_7))
        self.addAction(self.action7)

        self.action8 = QtGui.QAction('8', self)
        self.action8.triggered.connect(partial(self.choose_node, 7))
        self.action8.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_8))
        self.addAction(self.action8)

        self.action9 = QtGui.QAction('9', self)
        self.action9.triggered.connect(partial(self.choose_node, 8))
        self.action9.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_9))
        self.addAction(self.action9)

        self.ignore_action = QtGui.QAction('ignore', self)
        self.ignore_action.triggered.connect(self.ignore_node)
        self.ignore_action.setShortcut(S_.controls.ignore_case)
        self.addAction(self.ignore_action)

        self.new_region_t_action = QtGui.QAction('new region', self)
        self.new_region_t_action.triggered.connect(partial(self.new_region, -1))
        self.new_region_t_action.setShortcut(S_.controls.new_region)
        self.addAction(self.new_region_t_action)

        self.fitting_one_step_a = QtGui.QAction('fitting one step', self)
        self.fitting_one_step_a.triggered.connect(partial(self.fitting, False, True))
        self.fitting_one_step_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_F))
        self.addAction(self.fitting_one_step_a)

        self.d_ = None

    def update_content(self):
        self.next_case()