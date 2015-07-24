__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore
from gui.img_controls.my_scene import MyScene
from gui.gui_utils import cvimg2qtpixmap, get_img_qlabel
import numpy as np
from skimage.transform import resize
from gui.gui_utils import get_image_label
from utils.drawing.points import draw_points_crop, draw_points
from utils.video_manager import get_auto_video_manager, optimize_frame_access
from core.region.mser import get_msers_, get_all_msers
from skimage.transform import resize
from gui.img_controls.my_view import MyView
from gui.img_controls.my_scene import MyScene
import sys
from PyQt4 import QtGui, QtCore
from gui.img_controls.utils import cvimg2qtpixmap
import numpy as np
import pickle
from functools import partial
from core.animal import colors_
from core.region.fitting import Fitting
import cv2
from copy import deepcopy
from config_widget import ConfigWidget
from case_widget import CaseWidget
from new_region_widget import NewRegionWidget
from core.region.region import Region
from core.log import LogCategories, ActionNames
from gui.img_grid.img_grid_widget import ImgGridWidget
from utils.img import prepare_for_segmentation
from gui.gui_utils import get_image_label
from core.settings import Settings as S_
import time
import math


VISU_MARGIN = 10

class ComputeInQueue(QtCore.QThread):
    def __init__(self, callback):
        super(ComputeInQueue, self).__init__()
        self.model_ready = False
        self.bg_model = None
        self.callback = callback

    def run(self):
        """
        this method is called only when you want to run it in parallel.
        :return:
        """
        self.compute_model()


class ConfigurationsVisualizer(QtGui.QWidget):
    def __init__(self, solver, vid, graph_visu_callback):
        super(ConfigurationsVisualizer, self).__init__()
        self.setLayout(QtGui.QVBoxLayout())
        self.scenes_widget = QtGui.QWidget()
        self.scenes_widget.setLayout(QtGui.QVBoxLayout())
        self.scenes_widget.layout().setContentsMargins(0, 0, 0, 0)

        self.noise_nodes_filter_b = QtGui.QPushButton('noise filter')
        self.noise_nodes_filter_b.clicked.connect(self.noise_nodes_filter)
        self.layout().addWidget(self.noise_nodes_filter_b)

        self.noise_nodes_confirm_b = QtGui.QPushButton('remove selected')
        self.noise_nodes_confirm_b.clicked.connect(self.remove_noise)
        self.layout().addWidget(self.noise_nodes_confirm_b)
        self.noise_nodes_confirm_b.hide()

        self.noise_nodes_back_b = QtGui.QPushButton('back')
        self.noise_nodes_back_b.clicked.connect(self.next_case)
        self.layout().addWidget(self.noise_nodes_back_b)
        self.noise_nodes_back_b.hide()

        self.noise_nodes_widget = None

        self.scroll_ = QtGui.QScrollArea()
        self.scroll_.setWidgetResizable(True)
        self.scroll_.setWidget(self.scenes_widget)
        self.layout().addWidget(self.scroll_)

        self.graph_visu_callback = graph_visu_callback
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
        self.autosave_timer.timeout.connect(partial(self.save, True))
        # TODO: add interval to settings
        # self.autosave_timer.start(1000*60*10)
        self.order_by = 'time'

        self.join_regions_active = False
        self.join_regions_n1 = None

        self.layout().setContentsMargins(0, 0, 0, 0)

    def save(self, autosave=False):
        wd = self.solver.project.working_directory

        name = '/progress_save.pkl'
        if autosave:
            name = '/temp/__autosave.pkl'

        with open(wd+name, 'wb') as f:
            pc = pickle.Pickler(f)
            pc.dump(self.solver.g)
            pc.dump(self.solver.project.log)
            pc.dump(self.solver.ignored_nodes)

        print "PROGRESS SAVED"

    def set_nodes_queue(self, nodes):
        for n in nodes:
            if n.frame_ != self.solver.start_t and n.frame_ != self.solver.end_t:
                self.nodes.append(n)

        self.nodes = sorted(self.nodes, key=lambda k: k.frame_)
        self.active_node_id = 0

    def new_region(self, t_offset):
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

    def next_case(self, move_to_different_case=False):
        if move_to_different_case:
            self.active_node_id += 1

        self.nodes = self.solver.g.nodes()
        self.nodes = sorted(self.nodes, key=lambda k: k.frame_)

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

            self.active_cw = CaseWidget(self.solver.g, nodes_groups, config, self.vid, self)
            self.active_cw.active_node = None
            self.scenes_widget.layout().addWidget(self.active_cw)

            # self.graph_visu_callback(500, 600)

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
        self.nodes = sorted(self.nodes, key=lambda k: k.frame_)

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

        self.active_cw = CaseWidget(self.solver.g, nodes_groups, config, self.vid, self)
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

    def fitting(self, t_reversed=False):
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
        self.join_regions_active = True
        self.join_regions_n1 = self.active_cw.active_node

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

    def undo(self):
        S_.general.log_graph_edits = False

        log = self.solver.project.log
        last_actions = log.pop_last_user_action()

        solver = self.solver

        i = 0
        ignore_node = False
        for a in last_actions:
            # if a.action_name != ActionNames.CHUNK_REMOVE_FROM_REDUCED and a.action_name != ActionNames.CHUNK_ADD_TO_REDUCED:
            #     print a
            if a.action_name == ActionNames.ADD_NODE:
                solver.remove_node(a.data)
            elif a.action_name == ActionNames.REMOVE_NODE:
                solver.add_node(a.data)
            elif a.action_name == ActionNames.ADD_EDGE:
                try:
                    solver.remove_edge(a.data['n1'], a.data['n2'])
                except:
                    print "NOT EXISTING EDGE"
            elif a.action_name == ActionNames.REMOVE_EDGE:
                solver.add_edge(a.data['n1'], a.data['n2'], **a.data['data'])
            elif a.action_name == ActionNames.CHUNK_ADD_TO_REDUCED:
                a.data['chunk'].remove_from_reduced_(-1, self.solver)
                a.data['chunk'].is_sorted = False
            elif a.action_name == ActionNames.CHUNK_REMOVE_FROM_REDUCED:
                a.data['chunk'].add_to_reduced_(a.data['node'], self.solver, a.data['index'])
                a.data['chunk'].is_sorted = False
            elif a.action_name == ActionNames.CHUNK_SET_START:
                a.data['chunk'].set_start(a.data['old_start_n'], self.solver)
            elif a.action_name == ActionNames.CHUNK_SET_END:
                a.data['chunk'].set_end(a.data['old_end_n'], self.solver)
            elif a.action_name == ActionNames.IGNORE_NODE:
                del self.solver.ignored_nodes[a.data]
                ignore_node = True

            i += 1

        S_.general.log_graph_edits = True

        if ignore_node:
            self.active_node_id -= 1

        self.next_case()

    def noise_nodes_filter(self):
        self.noise_nodes_filter_b.hide()
        self.noise_nodes_confirm_b.show()
        self.noise_nodes_back_b.show()
        if self.scenes_widget.layout().count():
            it = self.scenes_widget.layout().itemAt(0)
            self.scenes_widget.layout().removeItem(it)
            it.widget().setParent(None)

        self.noise_nodes_widget = ImgGridWidget()

        self.scenes_widget.layout().addWidget(self.noise_nodes_widget)

        # TODO: add some settings...
        th = 0.2
        elem_width = 200

        print "COMPUTING, hold on..."

        to_process = []
        for n in self.solver.g.nodes():
            prob = self.project.stats.antlikeness_svm.get_prob(n)
            if prob[1] < th:
                to_process.append(n)

        start = time.time()

        optimized = optimize_frame_access(to_process)

        i = 0
        for n, seq, _ in optimized:
            if seq:
                while self.vid.frame_number() < n.frame_:
                    self.vid.move2_next()

                img = self.vid.img()
            else:
                img = self.vid.seek_frame(n.frame_)

            img = prepare_for_segmentation(img, self.project)
            item = get_img_qlabel(n.pts(), img, n, elem_width, elem_width, filled=True)
            item.set_selected(True)
            self.noise_nodes_widget.add_item(item)

            i += 1

            if i > 100:
                break

        print "DONE", time.time() - start

    def remove_noise(self):
        # TODO: add actions

        to_remove = self.noise_nodes_widget.get_selected()
        for n in to_remove:
            if n in self.solver.g.nodes():
                self.strong_remove_region(n, suppress_next_case=True)

        self.noise_nodes_back_b.hide()
        self.noise_nodes_confirm_b.hide()
        self.noise_nodes_filter_b.show()

        self.next_case()

    def ignore_node(self):
        self.project.log.add(LogCategories.USER_ACTION, ActionNames.IGNORE_NODES)
        for g in self.active_cw.nodes_groups:
            for n in g:
                self.solver.ignored_nodes[n] = True
                self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.IGNORE_NODE, n)

        self.next_case()

    def add_actions(self):
        self.next_action = QtGui.QAction('next', self)
        self.next_action.triggered.connect(partial(self.next_case, True))
        self.next_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_N))
        self.addAction(self.next_action)

        self.prev_action = QtGui.QAction('prev', self)
        self.prev_action.triggered.connect(self.prev_case)
        self.prev_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_B))
        self.addAction(self.prev_action)

        self.confirm_cc_action = QtGui.QAction('confirm', self)
        self.confirm_cc_action.triggered.connect(self.confirm_cc)
        self.confirm_cc_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_Space))
        self.addAction(self.confirm_cc_action)

        self.partially_confirm_action = QtGui.QAction('partially confirm', self)
        self.partially_confirm_action.triggered.connect(self.partially_confirm)
        self.partially_confirm_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_C))
        self.addAction(self.partially_confirm_action)

        self.path_confirm_action = QtGui.QAction('path confirm', self)
        self.path_confirm_action.triggered.connect(self.path_confirm)
        self.path_confirm_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_C))
        self.addAction(self.path_confirm_action)

        self.fitting_action = QtGui.QAction('fitting', self)
        self.fitting_action.triggered.connect(partial(self.fitting, False))
        self.fitting_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_F))
        self.addAction(self.fitting_action)

        self.fitting_rev_action = QtGui.QAction('fitting rev', self)
        self.fitting_rev_action.triggered.connect(partial(self.fitting, True))
        self.fitting_rev_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_G))
        self.addAction(self.fitting_rev_action)

        self.new_region_t1_action = QtGui.QAction('new region t1', self)
        self.new_region_t1_action.triggered.connect(partial(self.new_region, 0))
        self.new_region_t1_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Q))
        self.addAction(self.new_region_t1_action)

        self.new_region_t2_action = QtGui.QAction('new region t2', self)
        self.new_region_t2_action.triggered.connect(partial(self.new_region, 1))
        self.new_region_t2_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_W))
        self.addAction(self.new_region_t2_action)

        self.remove_region_action = QtGui.QAction('remove region', self)
        self.remove_region_action.triggered.connect(self.remove_region)
        self.remove_region_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace))
        self.addAction(self.remove_region_action)

        self.strong_remove_action = QtGui.QAction('strong remove', self)
        self.strong_remove_action.triggered.connect(self.strong_remove_region)
        self.strong_remove_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_Backspace))
        self.addAction(self.strong_remove_action)

        self.join_regions_action = QtGui.QAction('join regions', self)
        self.join_regions_action.triggered.connect(self.join_regions_pick_second)
        self.join_regions_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_J))
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

        self.save_progress = QtGui.QAction('save', self)
        self.save_progress.triggered.connect(self.save)
        self.save_progress.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_S))
        self.addAction(self.save_progress)

        self.undo_action = QtGui.QAction('undo', self)
        self.undo_action.triggered.connect(self.undo)
        self.undo_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_Z))
        self.addAction(self.undo_action)

        self.ignore_action = QtGui.QAction('ignore', self)
        self.ignore_action.triggered.connect(self.ignore_node)
        self.ignore_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_I))
        self.addAction(self.ignore_action)

        self.d_ = None


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    with open('/Volumes/Seagate Expansion Drive/mser_svm/eight/certainty_visu.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        g = up.load()
        ccs = up.load()
        vid_path = up.load()

    cv = ConfigurationsVisualizer(g, get_auto_video_manager(vid_path))

    i = 0
    for c_ in ccs:
        # if i == 10:
        #     break

        cv.add_configuration(c_)
        i += 1

    cv.showMaximized()

    app.exec_()
    sys.exit()