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
from new_region_widget import NewRegionWidget
from core.region.region import Region
import math

VISU_MARGIN = 10


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
        self.vid = vid
        self.cws = []
        self.cws_sorted = False

        self.node_ccs_refs = {}
        self.t1_nodes_cc_refs = {}
        self.t2_nodes_cc_refs = {}

        self.active_cw = 0
        self.active_cw_node = -1

        self.add_actions()

        self.cc_number_label = QtGui.QLabel('')
        self.layout().addWidget(self.cc_number_label)

        self.autosave_timer = QtCore.QTimer()
        self.autosave_timer.timeout.connect(partial(self.save, True))
        # TODO: add interval to settings
        self.autosave_timer.start(1000*60*10)


        # list of tuples (cw_id, str action, data)
        self.edit_actions = []

    def add_actions(self):
        self.next_action = QtGui.QAction('next', self)
        self.next_action.triggered.connect(self.next)
        self.next_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_N))
        self.addAction(self.next_action)

        self.prev_action = QtGui.QAction('prev', self)
        self.prev_action.triggered.connect(self.prev)
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
        self.fitting_action.triggered.connect(self.fitting)
        self.fitting_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_F))
        self.addAction(self.fitting_action)

        self.new_region_t1_action = QtGui.QAction('new region t1', self)
        self.new_region_t1_action.triggered.connect(partial(self.new_region, True))
        self.new_region_t1_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Q))
        self.addAction(self.new_region_t1_action)

        self.new_region_t2_action = QtGui.QAction('new region t2', self)
        self.new_region_t2_action.triggered.connect(partial(self.new_region, False))
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

        self.d_ = None


    def save(self, autosave=False):
        wd = self.solver.project.working_directory

        name = '/progress_save.pkl'
        if autosave:
            name = '/temp/__autosave.pkl'

        with open(wd+name, 'wb') as f:
            pc = pickle.Pickler(f)
            pc.dump(self.solver.g)
            pc.dump(self.edit_actions)

        print "PROGRESS SAVED"

    def new_region(self, is_t1):
        cw = self.get_cw_widget_at(self.active_cw)
        im = cw.crop_t1_widget.pixmap() if is_t1 else cw.crop_t2_widget.pixmap()
        frame = cw.frame_t if is_t1 else cw.frame_t+1

        w = NewRegionWidget(im, cw.crop_offset, frame, self.new_region_finished)
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
            # ADDING ACTION
            self.edit_actions.append(('new_region_finished', (confirmed, data)))

            r = Region()
            r.pts_ = data['pts']
            r.centroid_ = data['centroid']
            r.frame_ = data['frame']
            #TODO: get rid of this hack... also in antlikness test in solver.py
            # flag for virtual region
            r.min_intensity_ = -2

            new_ccs, node_representatives = self.solver.add_virtual_region(r)

            self.update_ccs(new_ccs, node_representatives)

    def remove_region(self, node=None):
        # ADDING ACTION
        self.edit_actions.append(('remove_region', node))

        if not node:
            p = self.active_cw_node
            cw = self.get_cw_widget_at(self.active_cw)

            if p < 0 or p > len(cw.c.regions_t1) + len(cw.c.regions_t2):
                return

            if p < len(cw.c.regions_t1):
                node = cw.c.regions_t1[p]
            else:
                node = cw.c.regions_t2[p - len(cw.c.regions_t1)]

            # # remove cw
            # if len(cw.c.regions_t1) + len(cw.c.regions_t2) == 1:
            #     widget_i, it = self.get_cc_item_position(cw.c)
            #     self.scenes_widget.layout().removeItem(it)
            #     self.cws.remove(cw)
            #     if node in self.t1_nodes_cc_refs:
            #         del self.t1_nodes_cc_refs[node]
            #     elif node in self.t2_nodes_cc_refs:
            #         del self.t2_nodes_cc_refs[node]

        new_ccs, node_representatives = self.solver.remove_region(node)
        self.update_ccs(new_ccs, node_representatives)

    def strong_remove_region(self):
        self.edit_actions.append(('strong_remove'))

        p = self.active_cw_node
        cw = self.get_cw_widget_at(self.active_cw)

        if p < 0 or p > len(cw.c.regions_t1) + len(cw.c.regions_t2):
            return

        if p < len(cw.c.regions_t1):
            node = cw.c.regions_t1[p]
        else:
            node = cw.c.regions_t2[p - len(cw.c.regions_t1)]

        new_ccs, node_representatives = self.solver.strong_remove(node)
        self.update_ccs(new_ccs, node_representatives)

    def choose_node(self, pos):
        # ADDING ACTION
        self.edit_actions.append(('choose_node', pos))

        cw = self.get_cw_widget_at(self.active_cw)
        cw.dehighlight_node(self.active_cw_node)
        self.active_cw_node = pos
        cw.highlight_node(pos)

    def get_cw_widget_at(self, i):
        return self.scenes_widget.layout().itemAt(i).widget()

    def next(self):
        # ADDING ACTION
        self.edit_actions.append(('next', None))

        if self.active_cw < self.scenes_widget.layout().count() - 1:
            cw = self.get_cw_widget_at(self.active_cw)
            cw.dehighlight_node(self.active_cw_node)
            self.active_cw_node = -1
            self.cw_set_inactive(cw)
            self.active_cw += 1
            self.cw_set_active(self.get_cw_widget_at(self.active_cw))
            self.scroll_.ensureWidgetVisible(self.get_cw_widget_at(self.active_cw))

    def prev(self):
        # ADDING ACTION
        self.edit_actions.append(('prev', None))

        if self.active_cw > 0:
            cw = self.get_cw_widget_at(self.active_cw)
            cw.dehighlight_node(self.active_cw_node)
            self.active_cw_node = -1
            self.cw_set_inactive(cw)
            self.active_cw -= 1
            self.cw_set_active(self.get_cw_widget_at(self.active_cw))
            self.scroll_.ensureWidgetVisible(self.get_cw_widget_at(self.active_cw))

    def confirm_cc(self):
        # ADDING ACTION
        self.edit_actions.append(('confirm_cc', None))

        cw = self.get_cw_widget_at(self.active_cw)
        cw.confirm_clicked()

    def fitting(self):
        # ADDING ACTION
        self.edit_actions.append(('fitting', None))

        if self.active_cw_node > -1:
            cw = self.get_cw_widget_at(self.active_cw)

            t_reversed = False
            if self.active_cw_node < len(cw.c.regions_t1):
                t_reversed = True

            cw.mark_merged(t_reversed)

    def cw_set_active(self, cw):
        cw.setStyleSheet("""QGraphicsView {background-color: rgb(200,200,200);} QPushButton {background-color: rgb(205,207,252);}""")

    def cw_set_inactive(self, cw):
        cw.setStyleSheet("""QGraphicsView {background-color: rgb(255,255,255);} QPushButton {background-color: rgb(255,255,255);}""")

    def visualize_n_sorted(self, n=np.inf, start=0):
        n = max(n, len(self.cws))

        # if not self.cws_sorted:
        #     self.cws = sorted(self.cws, key=lambda k: k.c.t)
        #     self.cws_sorted = True

        for i in range(start, min(start+n, len(self.cws))):
            self.scenes_widget.layout().addWidget(self.cws[i])

        if self.cws:
            self.cw_set_active(self.cws[0])

        self.cc_number_label.setText(str(self.scenes_widget.layout().count()))

    def add_configuration(self, cc):
        for n in cc.regions_t1+cc.regions_t2:
            if n in self.node_ccs_refs:
                self.node_ccs_refs[n].append(cc)
            else:
                self.node_ccs_refs[n] = [cc]

        for n in cc.regions_t1:
            self.t1_nodes_cc_refs[n] = cc

        for n in cc.regions_t2:
            self.t2_nodes_cc_refs[n] = cc

        self.cws_sorted = False
        cw = ConfigWidget(self.solver.g, cc, self.vid, self)
        self.cws.append(cw)
        # self.scenes_widget.layout().addWidget(cw)

    def replace_cw(self, new_cc, cc_to_be_replaced=None):
        if not self.is_cc_familiar(new_cc) and not cc_to_be_replaced:
            cw = ConfigWidget(self.solver.g, new_cc, self.vid, self)
            self.cws.append(cw)

            for i in range(0, self.scenes_widget.layout().count()):
                it = self.scenes_widget.layout().itemAt(i)
                if it.widget().frame_t > cw.frame_t:
                    break

            self.cws.insert(i, cw)
            self.scenes_widget.layout().insertWidget(i, cw)
        else:
            if not cc_to_be_replaced:
                if self.is_cc_familiar(new_cc):
                    cc_to_be_replaced = self.find_ref(new_cc)

            widget_i, it = self.get_cc_item_position(cc_to_be_replaced)

            # if cc_to_be_replaced, then there is a risc that the nodes will be different and as the color is based on node reference, it will fail...
            c_assignment = it.widget().color_assignments
            if cc_to_be_replaced:
                c_assignment = None

            cw = ConfigWidget(self.solver.g, new_cc, self.vid, self, color_assignments=c_assignment)
            self.cws.append(cw)

            self.scenes_widget.layout().removeItem(it)
            self.scenes_widget.layout().insertWidget(widget_i, cw)

            self.cws.remove(it.widget())
            it.widget().setParent(None)

        self.update_node_cc_refs(cc_to_be_replaced, new_cc)

    def get_cc_item_position(self, cc):
        for i in range(0, self.scenes_widget.layout().count()):
            it = self.scenes_widget.layout().itemAt(i)
            if it.widget().c == cc:
                return i, it

        return -1, None

    def update_node_cc_refs(self, old_cc, new_cc):
        if old_cc:
            for n_ in old_cc.regions_t1:
                del self.t1_nodes_cc_refs[n_]

            for n_ in old_cc.regions_t2:
                del self.t2_nodes_cc_refs[n_]

        if new_cc:
            for n_ in new_cc.regions_t1:
                self.t1_nodes_cc_refs[n_] = new_cc

            for n_ in new_cc.regions_t2:
                self.t2_nodes_cc_refs[n_] = new_cc

    def find_ref(self, cc):
        for n in cc.regions_t1:
            if n in self.t1_nodes_cc_refs:
                return self.t1_nodes_cc_refs[n]

        for n in cc.regions_t2:
            if n in self.t2_nodes_cc_refs:
                return self.t2_nodes_cc_refs[n]

        return None

    def is_cc_familiar_t1(self, cc):
        if cc:
            for n in cc.regions_t1:
                if n in self.t1_nodes_cc_refs:
                    return True

        return False

    def is_cc_familiar_t2(self, cc):
        if cc:
            for n in cc.regions_t2:
                if n in self.t2_nodes_cc_refs:
                    return True

        return False

    def is_cc_familiar(self, cc):
        if self.is_cc_familiar_t1(cc):
            return True

        return self.is_cc_familiar_t2(cc)

    def delete_empty_ccs(self):
        for cw in self.cws:
            delete = False
            for n in cw.c.regions_t1 + cw.c.regions_t2:
                if n in self.solver.g.nodes():
                    break

                delete = True

            if delete:
                _, it = self.get_cc_item_position(cw.c)
                if it:
                    self.scenes_widget.layout().removeItem(it)
                    self.update_node_cc_refs(it.widget().c, None)
                    self.cws.remove(it.widget())
                    it.widget().setParent(None)
                else:
                    self.cws.remove(cw)


    def update_ccs(self, new_ccs, node_representatives):
        min_t = np.inf
        max_t = 0

        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        for new_cc, n in zip(new_ccs, node_representatives):
            min_t = min(n.frame_, min_t)
            max_t = max(n.frame_, max_t)
            if new_cc:
                self.replace_cw(new_cc)
            else:
                if not self.is_cc_familiar_t1(new_cc) and n not in self.t1_nodes_cc_refs:
                    # already removed
                    continue

                old_cc = self.t1_nodes_cc_refs[n]
                _, it = self.get_cc_item_position(old_cc)
                self.scenes_widget.layout().removeItem(it)

                self.update_node_cc_refs(it.widget().c, new_cc)

                self.cws.remove(it.widget())
                it.widget().setParent(None)

        self.delete_empty_ccs()

        self.cw_set_active(self.get_cw_widget_at(self.active_cw))
        self.cc_number_label.setText(str(self.scenes_widget.layout().count()))
        QtGui.QApplication.processEvents()
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.ArrowCursor)
        self.scroll_.ensureWidgetVisible(self.get_cw_widget_at(self.active_cw), 0)

        # self.graph_visu_callback(min_t - math.ceil(VISU_MARGIN / 5.), max_t + VISU_MARGIN)

    def partially_confirm(self):
        # ADDING ACTION
        self.edit_actions.append(('partially_confirm', None))

        cw = self.get_cw_widget_at(self.active_cw)
        if -1 < self.active_cw_node < len(cw.c.regions_t1) + len(cw.c.regions_t2):
            p = self.active_cw_node
            cw = self.get_cw_widget_at(self.active_cw)

            if p < 0 or p > len(cw.c.regions_t1) + len(cw.c.regions_t2):
                return

            if p < len(cw.c.regions_t1):
                r = cw.c.regions_t1[p]
            else:
                r = cw.c.regions_t2[p - len(cw.c.regions_t1)]

            n1 = r
            conf = cw.c.configurations[0]

            i = 0
            for n1_, n2_ in conf:
                if n1_ == n1:
                    n2 = n2_
                    break

                if n2_ == n1:
                    n1 = n1_
                    n2 = n2_
                    break

                i += 1

            self.confirm_edges([(n1, n2)])

        self.active_cw_node = -1

    def confirm_edges(self, pairs):
        # ADDING ACTION
        self.edit_actions.append(('confirm_edges', pairs))

        new_ccs, node_representatives = self.solver.confirm_edges(pairs)
        self.update_ccs(new_ccs, node_representatives)

    def merged(self, new_regions, t_reversed, from_cc):
        # ADDING ACTION
        self.edit_actions.append(('merged', (new_regions, t_reversed, from_cc)))

        replace = from_cc.regions_t2
        nodes_to_connect = from_cc.regions_t1
        if t_reversed:
            replace = from_cc.regions_t1
            nodes_to_connect = from_cc.regions_t2

        new_ccs, node_representatives = self.solver.merged(new_regions, replace, nodes_to_connect, t_reversed)
        self.update_ccs(new_ccs, node_representatives)

    def join_regions(self, n1, n2):
        if n1.area() < n2.area():
            n1, n2 = n2, n1

        # TODO: update also other moments etc...
        n_new = deepcopy(n1)
        n_new.pts_ = np.concatenate((n_new.pts_, n2.pts_), 0)
        n_new.centroid_ = np.mean(n_new.pts_, 0)
        self.solver.remove_region(n1)
        self.solver.remove_region(n2)
        new_ccs, node_representativs = self.solver.add_virtual_region(n_new)
        self.update_ccs(new_ccs, node_representativs)

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