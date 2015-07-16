__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore
from gui.img_controls.my_scene import MyScene
from gui.gui_utils import cvimg2qtpixmap
import numpy as np
from skimage.transform import resize
from utils.img import get_roi, ROI
from gui.gui_utils import get_image_label
from utils.drawing.points import draw_points_crop, draw_points
from utils.video_manager import get_auto_video_manager
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
        self.autosave_timer.start(1000*60*10)
        self.order_by = 'time'

        self.join_regions_active = False
        self.join_regions_n1 = None

    def save(self, autosave=False):
        wd = self.solver.project.working_directory

        name = '/progress_save.pkl'
        if autosave:
            name = '/temp/__autosave.pkl'

        with open(wd+name, 'wb') as f:
            pc = pickle.Pickler(f)
            pc.dump(self.solver.g)
            pc.dump(self.solver.project.log)

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

    def remove_region(self, node=None):
        if not node:
            if not self.active_cw.active_node:
                return
            node = self.active_cw.active_node

        self.project.log.add(LogCategories.USER_ACTION, ActionNames.REMOVE, node)
        self.solver.remove_region(node)
        self.next_case()

    def strong_remove_region(self):
        if not self.active_cw.active_node:
            return

        self.project.log.add(LogCategories.USER_ACTION, ActionNames.STRONG_REMOVE, self.active_cw.active_node)
        self.solver.strong_remove(self.active_cw.active_node)
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
                # self.nodes.pop(self.active_node_id)
                self.active_node_id += 1
                self.next_case()
                return

            config = self.best_greedy_config(nodes_groups)

            self.active_cw = CaseWidget(self.solver.g, nodes_groups, config, self.vid, self)
            self.active_cw.active_node = None
            self.scenes_widget.layout().addWidget(self.active_cw)

            # min_t = self.active_cw.frame_t
            # max_t = min_t + 10
            # self.graph_visu_callback(min_t - math.ceil(VISU_MARGIN / 5.), max_t + VISU_MARGIN)

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
        if self.active_node_id == 0:
            return

        self.active_node_id -= 1

        self.nodes = self.solver.g.nodes()
        self.nodes = sorted(self.nodes, key=lambda k: k.frame_)

        n = self.nodes[self.active_node_id]

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
            self.prev_case()
            return

        config = self.best_greedy_config(nodes_groups)

        self.active_cw = CaseWidget(self.solver.g, nodes_groups, config, self.vid, self)
        self.active_cw.active_node = None
        self.scenes_widget.layout().addWidget(self.active_cw)

    def confirm_cc(self):
        self.active_cw.confirm_clicked()

    def fitting(self, t_reversed=False):
        if self.active_cw.active_node:
            self.project.log.add(LogCategories.USER_ACTION,
                                 ActionNames.MERGED_SELECTED,
                                 {
                                     'n': self.active_cw.active_node,
                                     't_reversed': t_reversed
                                 })

            is_ch, ch_t_reversed, chunk_ref = self.solver.is_chunk(self.active_cw.active_node)
            if is_ch:
                print "FITTING WHOLE CHUNK, WAIT PLEASE"
                merged = self.active_cw.active_node
                model = None

                # TODO: add to settings
                for i in range(min(chunk_ref.length()+3, 15)):
                    print i
                    if i == 0:
                        res = self.active_cw.mark_merged(merged, t_reversed)
                        merged = res[0]
                        model = deepcopy(res[1])
                    else:
                        f = Fitting(merged, model, num_of_iterations=10)
                        f.fit()

                        model = deepcopy(f.animals)
                        merged = self.solver.merged(f.animals, merged, t_reversed)

                    if not merged:
                        break

                    for m in model:
                        m.frame_ += -1 if t_reversed else 1

                print "CHUNK FINISHED"
            else:
                self.active_cw.mark_merged(self.active_cw.active_node, t_reversed)

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

    def undo(self):
        log = self.solver.project.log
        last_actions = log.pop_last_user_action()

        solver = self.solver

        for a in last_actions:
            print a
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
            elif a.action_name == ActionNames.DISASSEMBLE_CHUNK:
                solver.simplify_to_chunks([a.data['n']])
                _, _, ch = solver.is_chunk(a.data['n'])
                print ch
            elif a.action_name == ActionNames.ASSEMBLE_CHUNK:
                solver.disassemble_chunk([a.data['n']])
                _, _, ch = solver.is_chunk(a.data['n'])
                print ch
            elif a.action_name == ActionNames.MERGE_CHUNKS:
                solver.split_chunks(a.data['n'], a.data['chunk'])

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