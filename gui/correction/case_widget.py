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
from skimage.transform import rescale
from core.settings import Settings as S_


class CaseWidget(QtGui.QWidget):
    def __init__(self, G, node_groups, suggested_config, vid, parent_widget, color_assignments=None):
        super(CaseWidget, self).__init__()

        self.G = G
        self.nodes_groups = node_groups
        self.parent = parent_widget
        self.vid = vid

        self.suggested_config = None
        self.num_of_nodes = 0
        for g in self.nodes_groups:
            for n in g:
                self.num_of_nodes += 1

        self.process_suggested_config(suggested_config)

        self.node_size = 70
        self.frame_visu_margin = 100

        self.config_lines = []
        self.node_positions = []
        self.h_ = self.node_size + 3
        self.w_ = self.node_size + 100

        self.user_actions = []

        self.active_node = None
        self.connect_with_active = False
        self.join_with_active = False

        self.sub_g = self.G.subgraph([r for regions in self.nodes_groups for r in regions])

        self.it_nodes = {}

        self.active_config = 0
        self.frame_cache = []

        self.crop_visualize = True

        self.node_positions = {}

        self.frame_t = self.nodes_groups[0][0].frame_
        self.opacity = 0.5

        if color_assignments:
            self.color_assignments = color_assignments
        else:
            self.color_assignments = {}

            i = 0
            for g in self.nodes_groups:
                for n in g:
                    self.color_assignments[n] = colors_[i%len(colors_)] + (self.opacity, )
                    i += 1

            for _, n1, n2 in self.suggested_config:
                self.color_assignments[n2] = self.color_assignments[n1]

        self.pop_menu_node = QtGui.QMenu(self)
        self.action_remove_node = QtGui.QAction('remove', self)
        self.action_remove_node.triggered.connect(self.remove_node_)

        self.action_partially_confirm = QtGui.QAction('confirm this connection', self)
        self.action_partially_confirm.triggered.connect(self.partially_confirm)

        self.action_mark_merged = QtGui.QAction('merged', self)
        self.action_mark_merged.triggered.connect(self.mark_merged)

        self.new_region_t1 = QtGui.QAction('new region t1', self)
        self.new_region_t1.triggered.connect(partial(self.parent.new_region, True))

        self.new_region_t2 = QtGui.QAction('new region t2', self)
        self.new_region_t2.triggered.connect(partial(self.parent.new_region, False))

        self.connect_with = QtGui.QAction('connect with and confirm', self)
        self.connect_with.triggered.connect(self.connect_with_)

        self.join_with = QtGui.QAction('join with', self)
        self.join_with.triggered.connect(self.join_with_)

        self.pop_menu_node.addAction(self.action_remove_node)
        self.pop_menu_node.addAction(self.action_mark_merged)
        self.pop_menu_node.addAction(self.action_partially_confirm)
        self.pop_menu_node.addAction(self.new_region_t1)
        self.pop_menu_node.addAction(self.new_region_t2)
        self.pop_menu_node.addAction(self.connect_with)
        self.pop_menu_node.addAction(self.join_with)

        self.setLayout(QtGui.QVBoxLayout())
        self.v = QtGui.QGraphicsView()
        self.scene = MyScene()

        self.edge_pen = QtGui.QPen(QtCore.Qt.SolidLine)
        self.edge_pen.setColor(QtGui.QColor(0, 0, 0, 0x38))
        self.edge_pen.setWidth(1)

        self.strong_edge_pen = QtGui.QPen(QtCore.Qt.SolidLine)
        self.strong_edge_pen.setColor(QtGui.QColor(0, 255, 0, 0x78))
        self.strong_edge_pen.setWidth(2)

        self.frames_layout = QtGui.QHBoxLayout()
        self.layout().addLayout(self.frames_layout)

        self.layout().addWidget(self.v)
        self.v.setScene(self.scene)

        self.scene.clicked.connect(self.scene_clicked)

        self.cache_frames()
        self.draw_scene()
        self.draw_frame()

        self.score_list = QtGui.QVBoxLayout()
        self.layout().addLayout(self.score_list)

        self.confirm_b = QtGui.QPushButton('confirm')
        self.confirm_b.clicked.connect(self.confirm_clicked)

        self.v.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.connect(self.v, QtCore.SIGNAL('customContextMenuRequested(const QPoint&)'), self.on_context_menu)

        for _, n1, n2 in self.suggested_config:
            if n1 not in self.color_assignments:
                self.color_assignments[n1] = colors_[10]

            self.color_assignments[n2] = self.color_assignments[n1]

            t = n1.frame_ - self.frame_t
            line_ = QtGui.QGraphicsLineItem(self.node_size + self.w_ * t, self.node_positions[n1]*self.h_ + self.h_/2, self.w_ * (t + 1), self.node_positions[n2]*self.h_ + self.h_/2)
            line_.setPen(self.strong_edge_pen)
            self.config_lines.append(line_)
            self.scene.addItem(line_)

    def remove_node_(self):
        self.parent.remove_region(self.active_node)

    def get_node_item(self, node):
        n_it = None
        for it, key in self.it_nodes.iteritems():
            if key == node:
                n_it = it

        return n_it

    def get_node_at_pos(self, node_pos):
        i = 0
        for g in self.nodes_groups:
            for n in g:
                if i == node_pos:
                    return n

                i += 1

        return None

    def get_node_item_at_pos(self, node_pos):
        n_key = None

        for key, pos in self.node_positions.iteritems():
            if self.frame_t < key.frame_:
                pos += len(self.nodes_groups[0])

            if pos == node_pos:
                n_key = key
                break

        n_it = None

        for it, key in self.it_nodes.iteritems():
            if key == n_key:
                n_it = it
                break

        return n_it

    def connect_with_(self):
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
        self.connect_with_active = True

    def join_with_(self):
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
        self.join_with_active = True

    def highlight_node(self, node):
        if node:
            it = self.get_node_item(node)
            it.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)
            it.setSelected(True)

    def dehighlight_node(self, node):
        if node:
            it = self.get_node_item(node)
            it.setSelected(False)
            it.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)

    def on_context_menu(self, point):
        it = self.scene.itemAt(self.v.mapToScene(point))

        if isinstance(it, QtGui.QGraphicsPixmapItem):
            self.active_node = self.it_nodes[it]
            self.pop_menu_node.exec_(self.v.mapToGlobal(point))
        else:
            self.active_node = None

    def scene_clicked(self, point):
        it = self.scene.itemAt(point)

        if isinstance(it, QtGui.QGraphicsPixmapItem):
            n1 = self.active_node
            n2 = self.it_nodes[it]

            if self.connect_with_active:
                if self.active_node.frame_ > self.it_nodes[it].frame_:
                    n1 = self.it_nodes[it]
                    n2 = self.active_node

                self.parent.confirm_edges([(n1, n2)])
                self.connect_with_active = False
                QtGui.QApplication.setOverrideCursor(QtCore.Qt.ArrowCursor)

            elif self.join_with_active:
                self.parent.join_regions(n1, n2)
                self.join_with_active = False
                QtGui.QApplication.setOverrideCursor(QtCore.Qt.ArrowCursor)

            self.active_node = self.it_nodes[it]
        else:
            self.connect_with_active = False
            self.join_with_active = False
            QtGui.QApplication.setOverrideCursor(QtCore.Qt.ArrowCursor)
            self.active_node = None

    def confirm_clicked(self):
        if len(self.nodes_groups) < 1:
            return

        w = len(self.nodes_groups[0])
        for g in self.nodes_groups:
            if w != len(g):
                print "UNBALANCED configuration, ignoring confirmation"
                return
            w = len(g)

        pairs = []

        for _, n1, n2 in self.suggested_config:
            pairs.append((n1, n2))

        self.parent.confirm_edges(pairs)

    def get_im(self, n, t1=True):
        im = self.frame_cache[n.frame_-self.frame_t].copy()

        vis = draw_points_crop(im, n.pts(), color=self.get_node_color(n), square=True)

        if vis.shape[0] > self.node_size or vis.shape[1] > self.node_size:
            vis = np.asarray(resize(vis, (self.node_size, self.node_size)) * 255, dtype=np.uint8)

        return cvimg2qtpixmap(vis)

    def get_node_color(self, n):
        return self.color_assignments[n]

    def draw_frame(self):
        centroids = []
        for n in self.nodes_groups[0]:
            centroids.append(n.centroid())

        roi = get_roi(np.array(centroids))
        m = self.frame_visu_margin

        h_, w_, _ = self.frame_cache[0].shape
        roi = ROI(max(0, roi.y() - m), max(0, roi.x() - m), min(roi.height() + 2*m, h_), min(roi.width() + 2*m, w_))

        for i in range(len(self.nodes_groups)):
            im = self.frame_cache[i].copy()

            if self.crop_visualize:
                for r in self.nodes_groups[i]:
                    im = draw_points(im, r.pts(), color=self.get_node_color(r))

            self.crop_offset = roi.top_left_corner()
            crop = np.copy(im[roi.y():roi.y()+roi.height(), roi.x():roi.x()+roi.width(), :])
            cv2.putText(crop, str(self.frame_t+i), (1, 10), cv2.FONT_HERSHEY_PLAIN, 0.55, (255, 255, 255), 1, cv2.cv.CV_AA)

            self.frames_layout.addWidget(get_image_label(crop))

    def cache_frames(self):
        for i in range(len(self.nodes_groups)):
            if i == 0:
                im = self.vid.seek_frame(self.frame_t)
            else:
                im = self.vid.move2_next()

            if S_.mser.img_subsample_factor > 1.0:
                im = np.asarray(rescale(im, 1 / S_.mser.img_subsample_factor) * 255, dtype=np.uint8)

            self.frame_cache.append(im)

    def mark_merged(self, region, t_reversed=None):
        merged_t = region.frame_ - self.frame_t
        model_t = merged_t + 1 if t_reversed else merged_t - 1

        if len(self.nodes_groups[model_t]) > 0 and len(self.nodes_groups[merged_t]) > 0:
            if t_reversed is None:
                avg_area_c1 = 0
                for c1 in self.nodes_groups.regions_t1:
                    avg_area_c1 += c1.area()
                avg_area_c1 /= float(len(self.nodes1_groups.regions_t1))

                avg_area_c2 = 0
                for c2 in self.nodes_groups.regions_t2:
                    avg_area_c2 += c2.area()

                avg_area_c2 /= float(len(self.nodes_groups.regions_t2))

                t_reversed = False
                if avg_area_c1 > avg_area_c2:
                    t_reversed = True

            t1_ = self.nodes_groups[model_t]
            t2_ = self.nodes_groups[merged_t]

            reg = []
            for c2 in t2_:
                if not reg:
                    reg = deepcopy(c2)
                else:
                    reg.pts_ = np.append(reg.pts_, c2.pts_, axis=0)

            objects = []
            for c1 in t1_:
                a = deepcopy(c1)
                if t_reversed:
                    a.frame_ -= 1
                else:
                    a.frame_ += 1

                objects.append(a)

            f = Fitting(reg, objects, num_of_iterations=10)
            f.fit()

            print "PREVIOUS FITTING FUNCTION USED"

            return [self.parent.solver.merged(f.animals, region, t_reversed), f.animals]

    def partially_confirm(self):
        conf = self.nodes_groups.configurations[self.active_config]
        n1 = self.active_node

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

        self.parent.confirm_edges([(n1, n2)])

    def draw_scene(self):
        max_h_pos = 0
        for i in range(len(self.nodes_groups)):
            h_pos = 0
            for n in self.nodes_groups[i]:
                self.node_positions[n] = h_pos
                if i == 0:
                    it = self.scene.addPixmap(self.get_im(n))
                else:
                    # TODO:
                    it = self.scene.addPixmap(self.get_im(n, t1=False))

                self.it_nodes[it] = n
                it.setPos(self.w_ * i, h_pos * self.h_)
                h_pos += 1

            max_h_pos = max(max_h_pos, h_pos)

        max_h_pos = max(max_h_pos, h_pos)
        self.v.setFixedHeight(max_h_pos * self.h_)
        for i in range(len(self.nodes_groups) - 1):
            for n in self.nodes_groups[i]:
                for _, n2 in self.G.out_edges(n):
                    if n2 not in self.node_positions:
                        print "n2 not in node_positions case_widget.py", n.frame_, n2.frame_
                        continue

                    line_ = QtGui.QGraphicsLineItem(self.node_size + self.w_*i,
                                                    self.node_positions[n]*self.h_ + self.h_/2,
                                                    self.w_ + i * self.w_,
                                                    self.node_positions[n2]*self.h_ + self.h_/2)
                    line_.setPen(self.edge_pen)
                    self.scene.addItem(line_)

    def process_suggested_config(self, suggested_config):
        l_ = []
        for n1, n2 in suggested_config.iteritems():
            l_.append([n1.frame_, n1, n2])

        self.suggested_config = sorted(l_, key=lambda k: k[0])