__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore
from gui.img_controls.my_scene import MyScene
from gui.gui_utils import cvimg2qtpixmap
import numpy as np
from skimage.transform import resize
from utils.roi import ROI, get_roi
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


class ConfigWidget(QtGui.QWidget):
    def __init__(self, G, c, vid, parent_widget, color_assignments=None):
        super(ConfigWidget, self).__init__()

        self.G = G
        self.c = c
        self.parent = parent_widget

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

        self.sub_g = self.G.subgraph(self.c.regions_t1 + self.c.regions_t2)

        self.it_nodes = {}

        self.active_config = 0
        self.im_t1 = None
        self.im_t2 = None
        self.crop_t1_widget = None
        self.crop_t2_widget = None
        self.crop_offset = None

        self.crop_visualize = True

        self.node_positions = {}

        self.frame_t = self.c.regions_t1[0].frame_

        if color_assignments:
            self.color_assignments = color_assignments
        else:
            self.color_assignments = {}

            for n, i in zip(self.c.regions_t1, range(len(self.c.regions_t1))):
                self.color_assignments[n] = colors_[i]

        if not self.im_t1:
            self.im_t1 = vid.seek_frame(self.frame_t)
            self.im_t2 = vid.move2_next()
            if S_.mser.img_subsample_factor > 1.0:
                self.im_t1 = np.asarray(rescale(self.im_t1, 1/S_.mser.img_subsample_factor) * 255, dtype=np.uint8)
                self.im_t2 = np.asarray(rescale(self.im_t2, 1/S_.mser.img_subsample_factor) * 255, dtype=np.uint8)

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

        self.setLayout(QtGui.QHBoxLayout())
        self.v = QtGui.QGraphicsView()
        self.scene = MyScene()

        self.edge_pen = QtGui.QPen(QtCore.Qt.SolidLine)
        self.edge_pen.setColor(QtGui.QColor(0, 0, 0, 0x38))
        self.edge_pen.setWidth(1)

        self.strong_edge_pen = QtGui.QPen(QtCore.Qt.SolidLine)
        self.strong_edge_pen.setColor(QtGui.QColor(0, 255, 0, 0x78))
        self.strong_edge_pen.setWidth(2)

        self.layout().addWidget(self.v)
        self.v.setScene(self.scene)
        self.layout().addWidget(QtGui.QLabel(str(0 if c.certainty < 0.001 else c.certainty)[0:5]))

        self.scene.clicked.connect(self.scene_clicked)

        self.draw_scene()
        self.draw_frame()

        self.score_list = QtGui.QVBoxLayout()
        self.layout().addLayout(self.score_list)

        self.confirm_b = QtGui.QPushButton('confirm')
        self.confirm_b.clicked.connect(self.confirm_clicked)

        self.layout().addWidget(self.confirm_b)
        self.v.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.connect(self.v, QtCore.SIGNAL('customContextMenuRequested(const QPoint&)'), self.on_context_menu)

        self.show_configuration(0)

    def remove_node_(self):
        self.parent.remove_region(self.active_node)

    def get_node_item(self, node):
        n_it = None
        for it, key in self.it_nodes.iteritems():
            if key == node:
                n_it = it

        return n_it

    def get_node_at_pos(self, node_pos):
        node = None

        for key, pos in self.node_positions.iteritems():
            if self.frame_t < key.frame_:
                pos += len(self.c.regions_t1)

            if pos == node_pos:
                node = key
                break

        return node

    def get_node_item_at_pos(self, node_pos):
        n_key = None

        for key, pos in self.node_positions.iteritems():
            if self.frame_t < key.frame_:
                pos += len(self.c.regions_t1)

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
        if len(self.c.regions_t1) != len(self.c.regions_t2):
            print "UNBALANCED configuration, ignoring confirmation"
            return

        pairs = []
        for n1, n2 in self.c.configurations[self.active_config]:
            pairs.append((n1, n2))

        self.parent.confirm_edges(pairs)

    def get_im(self, n, t1=True):
        if t1:
            im = self.im_t1
        else:
            im = self.im_t2


        vis = draw_points_crop(im, n.pts(), color=self.get_node_color(n), square=True)
        # vis = self.G.node[n]['img']

        if vis.shape[0] > self.node_size or vis.shape[1] > self.node_size:
            vis = np.asarray(resize(vis, (self.node_size, self.node_size)) * 255, dtype=np.uint8)

        return cvimg2qtpixmap(vis)

    def get_node_color(self, n):
        opacity = 0.5
        c = colors_[self.node_positions[n] + len(self.c.regions_t1)] + (opacity, )
        for c1, c2 in self.c.configurations[self.active_config]:
            if c1 == n:
                c = self.color_assignments[n] + (opacity, )
                break
            if c2 == n:
                c = self.color_assignments[c1] + (opacity, )
                break

        return c

    def draw_frame(self):
        centroids = []
        for n in self.c.regions_t1:
            centroids.append(n.centroid())

        roi = get_roi(np.array(centroids))
        m = self.frame_visu_margin

        h_, w_, _ = self.im_t1.shape

        im = self.im_t1
        if self.crop_visualize:
            im = self.im_t1.copy()

            for r in self.c.regions_t1:
                im = draw_points(im, r.pts(), color=self.get_node_color(r))

        roi = ROI(max(0, roi.y() - m), max(0, roi.x() - m), min(roi.height() + 2*m, h_), min(roi.width() + 2*m, w_))
        self.crop_offset = roi.top_left_corner()
        crop = np.copy(im[roi.y():roi.y()+roi.height(), roi.x():roi.x()+roi.width(), :])
        cv2.putText(crop, str(self.frame_t), (1, 10), cv2.FONT_HERSHEY_PLAIN, 0.55, (255, 255, 255), 1, cv2.cv.CV_AA)

        self.crop_t1_widget = get_image_label(crop)

        self.layout().insertWidget(0, self.crop_t1_widget)

        if self.crop_visualize:
            im = self.im_t2.copy()

            for r in self.c.regions_t2:
                im = draw_points(im, r.pts(), color=self.get_node_color(r))

        crop = np.copy(im[roi.y():roi.y()+roi.height(), roi.x():roi.x()+roi.width(), :])
        self.crop_t2_widget = get_image_label(crop)
        self.layout().insertWidget(1, self.crop_t2_widget)

    def mark_merged(self, t_reversed=None):
        if len(self.c.regions_t1) > 0 and len(self.c.regions_t2) > 0:
            if t_reversed is None:
                avg_area_c1 = 0
                for c1 in self.c.regions_t1:
                    avg_area_c1 += c1.area()
                avg_area_c1 /= float(len(self.c.regions_t1))

                avg_area_c2 = 0
                for c2 in self.c.regions_t2:
                    avg_area_c2 += c2.area()

                avg_area_c2 /= float(len(self.c.regions_t2))

                t_reversed = False
                if avg_area_c1 > avg_area_c2:
                    t_reversed = True

            t1_ = self.c.regions_t2 if t_reversed else self.c.regions_t1
            t2_ = self.c.regions_t1 if t_reversed else self.c.regions_t2

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

            self.parent.merged(f.animals, t_reversed, self.c)

    def show_configuration(self, id):
        self.active_config = id
        for it in self.config_lines:
            self.scene.removeItem(it)

        self.config_lines = []

        for n1, n2 in self.c.configurations[id]:
            if n2 is None:
                continue

            line_ = QtGui.QGraphicsLineItem(self.node_size, self.node_positions[n1]*self.h_ + self.h_/2, self.w_, self.node_positions[n2]*self.h_ + self.h_/2)
            line_.setPen(self.strong_edge_pen)
            self.config_lines.append(line_)
            self.scene.addItem(line_)

    def partially_confirm(self):
        conf = self.c.configurations[self.active_config]
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

    def redraw_config(self):
        self.scene = MyScene()
        self.v.setScene(self.scene)
        self.draw_scene()

        self.layout().removeWidget(self.crop_t1_widget)
        self.crop_t1_widget.setParent(None)
        self.layout().removeWidget(self.crop_t2_widget)
        self.crop_t2_widget.setParent(None)

        self.draw_frame()

    def draw_scene(self):
        h_pos = 0
        for n in self.c.regions_t1:
            self.node_positions[n] = h_pos
            it = self.scene.addPixmap(self.get_im(n))
            self.it_nodes[it] = n
            it.setPos(0, h_pos * self.h_)
            h_pos += 1

        max_h_pos = h_pos

        h_pos = 0
        for n in self.c.regions_t2:
            self.node_positions[n] = h_pos
            it = self.scene.addPixmap(self.get_im(n, t1=False))

            self.it_nodes[it] = n
            it.setPos(self.w_, h_pos * self.h_)
            h_pos += 1

        max_h_pos = max(max_h_pos, h_pos)
        self.v.setFixedHeight(max_h_pos * self.h_)
        for n in self.c.regions_t1:
            for _, n2 in self.sub_g.out_edges(n):
                line_ = QtGui.QGraphicsLineItem(self.node_size, self.node_positions[n]*self.h_ + self.h_/2, self.w_, self.node_positions[n2]*self.h_ + self.h_/2)
                line_.setPen(self.edge_pen)
                self.scene.addItem(line_)