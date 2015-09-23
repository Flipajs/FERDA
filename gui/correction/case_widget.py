__author__ = 'fnaiser'

from utils.roi import ROI, get_roi
from utils.drawing.points import draw_points_crop, draw_points
from skimage.transform import resize
from gui.img_controls.my_scene import MyScene
from PyQt4 import QtGui, QtCore
from gui.img_controls.utils import cvimg2qtpixmap
import numpy as np
from functools import partial
from core.animal import colors_
from core.region.fitting import Fitting
import cv2
from copy import deepcopy
from skimage.transform import rescale
from core.settings import Settings as S_


class CaseWidget(QtGui.QWidget):
    def __init__(self, G, project, node_groups, suggested_config, vid, parent_widget, color_assignments=None):
        super(CaseWidget, self).__init__()

        print "--------------------------------------------"
        self.project = project
        self.G = G
        self.nodes_groups = node_groups
        self.parent = parent_widget
        self.vid = vid

        self.suggested_config = None
        self.num_of_nodes = 0
        for g in self.nodes_groups:
            for _ in g:
                self.num_of_nodes += 1

        self.process_suggested_config(suggested_config)

        self.node_size = 70
        self.frame_visu_margin = 30

        self.config_lines = []
        self.node_positions = []
        self.h_ = self.node_size + 2
        self.w_ = self.node_size + 100
        self.top_margin = 0
        self.left_margin = 10

        self.user_actions = []

        self.connect_with_active = False
        self.join_with_active = False

        self.sub_g = self.G.subgraph([r for regions in self.nodes_groups for r in regions])

        self.it_nodes = {}

        self.active_config = 0
        self.frame_cache = []
        self.crop_pixmaps_cache = []
        self.crop_clear_frames_items = []
        self.visualization_hidden = False

        self.active_row = 0
        self.active_col = 0
        self.active_row_it = None
        self.active_col_it = None
        self.rows = -1
        self.cols = -1

        self.crop_visualize = True
        self.crop_offset = None

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
        self.action_partially_confirm.triggered.connect(self.parent.partially_confirm)

        self.action_mark_merged = QtGui.QAction('merged', self)
        self.action_mark_merged.triggered.connect(self.mark_merged)

        self.new_region_t1 = QtGui.QAction('new region t1', self)
        self.new_region_t1.triggered.connect(partial(self.parent.new_region, 0))

        self.new_region_t2 = QtGui.QAction('new region t2', self)
        self.new_region_t2.triggered.connect(partial(self.parent.new_region, 1))

        self.connect_with = QtGui.QAction('connect with and confirm', self)
        self.connect_with.triggered.connect(self.connect_with_)

        self.join_with = QtGui.QAction('join with', self)
        self.join_with.triggered.connect(self.join_with_)

        self.get_info_action = QtGui.QAction('get info', self)
        self.get_info_action.triggered.connect(self.get_info)
        self.get_info_action.setShortcut(S_.controls.get_info)
        self.addAction(self.get_info_action)

        # ARROW KEYS
        self.row_up = QtGui.QAction('row up', self)
        self.row_up.triggered.connect(partial(self.row_changed, -1))
        self.row_up.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Up))
        self.addAction(self.row_up)

        self.row_down = QtGui.QAction('row down', self)
        self.row_down.triggered.connect(partial(self.row_changed, 1))
        self.row_down.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Down))
        self.addAction(self.row_down)

        self.col_left = QtGui.QAction('col left', self)
        self.col_left.triggered.connect(partial(self.col_changed, -1))
        self.col_left.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left))
        self.addAction(self.col_left)

        self.col_right = QtGui.QAction('col right', self)
        self.col_right.triggered.connect(partial(self.col_changed, 1))
        self.col_right.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right))
        self.addAction(self.col_right)

        self.hide_visualization_a = QtGui.QAction('hide visualization', self)
        self.hide_visualization_a.triggered.connect(self.hide_visualization)
        self.hide_visualization_a.setShortcut(S_.controls.hide_show)
        self.addAction(self.hide_visualization_a)

        self.pop_menu_node.addAction(self.action_remove_node)
        self.pop_menu_node.addAction(self.action_mark_merged)
        self.pop_menu_node.addAction(self.action_partially_confirm)
        self.pop_menu_node.addAction(self.new_region_t1)
        self.pop_menu_node.addAction(self.new_region_t2)
        self.pop_menu_node.addAction(self.connect_with)
        self.pop_menu_node.addAction(self.join_with)
        self.pop_menu_node.addAction(self.get_info_action)

        self.setLayout(QtGui.QVBoxLayout())
        self.v = QtGui.QGraphicsView()
        self.scene = MyScene()

        self.edge_pen = QtGui.QPen(QtCore.Qt.SolidLine)
        self.edge_pen.setColor(QtGui.QColor(0, 0, 0, 0x16))
        self.edge_pen.setWidth(1)

        self.strong_edge_pen = QtGui.QPen(QtCore.Qt.SolidLine)
        self.strong_edge_pen.setColor(QtGui.QColor(0, 255, 0, 0x78))
        self.strong_edge_pen.setWidth(2)

        self.node_bg_color = QtGui.QColor(230, 230, 230, 230)
        op = 100
        self.bg_light_stripe = QtGui.QColor(255, 255, 255, op)
        self.bg_dark_stripe = QtGui.QColor(255, 255, 190, op)

        self.bg_light_stripe_r = QtGui.QColor(212, 250, 255, op)
        self.bg_dark_stripe_r = QtGui.QColor(242, 220, 232, op)

        self.chunk_highlight_pen = QtGui.QPen(QtCore.Qt.DotLine)
        self.chunk_highlight_pen.setColor(QtGui.QColor(255, 0, 0, 0x78))
        self.chunk_highlight_pen.setWidth(2)

        self.grid_pen = QtGui.QPen(QtCore.Qt.SolidLine)
        self.grid_pen.setColor(QtGui.QColor(135, 185, 201, 0x86))
        self.grid_pen.setWidth(1)

        self.grid_mark_pen= QtGui.QPen(QtCore.Qt.SolidLine)
        self.grid_mark_pen.setColor(QtGui.QColor(0, 0, 0, 0xff))
        self.grid_mark_pen.setWidth(2)

        self.layout().addWidget(self.v)
        self.v.setScene(self.scene)

        self.scene.clicked.connect(self.scene_clicked)
        self.v.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.cache_frames()
        self.draw_frames()
        self.draw_grid()
        self.draw_scene()
        self.highlight_chunk_nodes()
        self.active_node = None
        # self.highlight_node(self.nodes_groups[self.active_col][self.active_row])
        #self.draw_selection_rect()

        self.v.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.connect(self.v, QtCore.SIGNAL('customContextMenuRequested(const QPoint&)'), self.on_context_menu)

        for _, n1, n2 in self.suggested_config:
            if n1 not in self.color_assignments:
                self.color_assignments[n1] = colors_[10]

            self.color_assignments[n2] = self.color_assignments[n1]

            t = n1.frame_ - self.frame_t
            line_ = QtGui.QGraphicsLineItem(self.left_margin + self.node_size + self.w_ * t,
                                            self.top_margin + self.node_positions[n1]*self.h_ + self.h_/2,
                                            self.left_margin + self.w_ * (t + 1),
                                            self.top_margin + self.node_positions[n2]*self.h_ + self.h_/2)
            line_.setPen(self.strong_edge_pen)
            self.config_lines.append(line_)
            self.scene.addItem(line_)

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet("border: 0px")
        self.draw_selection_rect()

    def get_info(self):
        n = self.active_node

        antlikeness = self.parent.solver.project.stats.antlikeness_svm.get_prob(n)[1]
        virtual = False
        try:
            if n.is_virtual:
                antlikeness = 1.0
                virtual = True
        except:
            pass

        best_out = 0
        for _, _, d in self.parent.solver.g.out_edges(n, data=True):
            if 'score' in d:
                best_out = min(best_out, d['score'])

        best_in = 0
        for _, _, d in self.parent.solver.g.in_edges(n, data=True):
            if 'score' in d:
                best_in = min(best_in, d['score'])

        is_ch, _, ch = self.parent.solver.is_chunk(n)
        ch_info = ''
        if is_ch:
            ch_info = str(ch)
        QtGui.QMessageBox.about(self, "My message box",
                                "Area = %i\nCentroid = %s\nMargin = %i\nAntlikeness = %f\nIs virtual: %s\nBest in = %s\nBest out = %s\nChunk info = %s" % (n.area(), str(n.centroid()), n.margin_, antlikeness, str(virtual), str(best_in), str(best_out), ch_info))

    def row_changed(self, off):
        self.active_row += off
        self.active_row = self.active_row % self.rows
        self.draw_selection_rect()

    def col_changed(self, off):
        self.active_col += off
        self.active_col = self.active_col % self.cols
        self.draw_selection_rect()


    def draw_selection_rect(self):
        c = self.active_col
        r = self.active_row

        col_it = QtGui.QGraphicsRectItem(self.left_margin + self.w_ * c - self.node_size / 2,
                               self.top_margin, self.w_, self.h_ * self.rows)
        if self.active_col_it:
            self.scene.removeItem(self.active_col_it)

        self.active_col_it = col_it
        self.scene.addItem(col_it)
        col_it.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)
        col_it.setZValue(-1)

        row_it = QtGui.QGraphicsRectItem(self.left_margin - self.node_size / 2,
                           self.top_margin + self.h_ * r, self.w_ * self.cols, self.h_)

        if self.active_row_it:
            self.scene.removeItem(self.active_row_it)

        self.active_row_it = row_it
        self.scene.addItem(row_it)
        row_it.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)
        row_it.setZValue(-1)

        n = self.nodes_groups[self.active_col][self.active_row]
        self.highlight_node(n)

    def draw_grid(self):
        rows = 0
        for g in self.nodes_groups:
            rows = max(rows, len(g))

        self.rows = rows

        cols = len(self.nodes_groups)
        self.cols = cols

        # in case when there is only end of chunk missing region...
        if self.cols == 1:
            self.cols += 1

        whole_grid = False
        light_stripes = False

        if light_stripes:
            for r in range(rows):
                p = self.bg_light_stripe if r % 2 else self.bg_dark_stripe
                self.scene.addRect(self.left_margin - self.node_size / 2,
                                   self.top_margin + self.h_ * r - 1,
                                   self.w_ * cols,
                                   self.h_ - 1,
                                   p,
                                   p)

            for c in range(cols):
                p = self.bg_light_stripe_r if c % 2 else self.bg_dark_stripe_r
                self.scene.addRect(self.left_margin + self.w_ * c - self.node_size / 2,
                                   self.top_margin,
                                   self.w_,
                                   self.h_ * rows,
                                   p,
                                   p)

        if whole_grid:
            for r in range(1, rows):
                line_ = QtGui.QGraphicsLineItem(self.left_margin - self.node_size / 2,
                                                self.top_margin + self.h_ * r - 1,
                                                self.left_margin - self.node_size / 2 + self.w_ * cols,
                                                self.top_margin + self.h_ * r - 1)

                line_.setPen(self.grid_pen)
                self.scene.addItem(line_)

            for c in range(1, cols):
                line_ = QtGui.QGraphicsLineItem(self.left_margin + self.w_ * c - self.node_size / 2,
                                                self.top_margin,
                                                self.left_margin + self.w_ * c - self.node_size / 2,
                                                self.top_margin + self.h_ * rows)

                line_.setPen(self.grid_pen)
                self.scene.addItem(line_)
        else:
            mark_w = 10
            for r in range(rows+1):
                a = (self.w_ - self.node_size) / 2
                line_ = QtGui.QGraphicsLineItem(self.left_margin - a - mark_w / 2,
                                                self.top_margin + self.h_ * r - 1,
                                                self.left_margin - a + mark_w / 2,
                                                self.top_margin + self.h_ * r - 1)

                line_.setPen(self.grid_mark_pen)
                self.scene.addItem(line_)

            for c in range(cols + 1):
                line_ = QtGui.QGraphicsLineItem(self.left_margin + self.w_ * c - self.node_size / 2,
                                                self.top_margin + self.h_ * rows - mark_w / 2,
                                                self.left_margin + self.w_ * c - self.node_size / 2,
                                                self.top_margin + self.h_ * rows + mark_w / 2)

                line_.setPen(self.grid_mark_pen)
                self.scene.addItem(line_)

    def remove_node_(self):
        self.parent.remove_region(self.active_node)

    def highlight_chunk_nodes(self):
        highlight_line_len = self.node_size / 2

        for g in self.nodes_groups:
            for n in g:
                is_ch, t_rev, _ = self.parent.solver.is_chunk(n)
                if is_ch:
                    t = n.frame_ - self.frame_t

                    if t_rev:
                        line_ = QtGui.QGraphicsLineItem(self.left_margin + self.w_ * t - highlight_line_len,
                                                        self.top_margin + self.node_positions[n]*self.h_ + self.h_/2,
                                                        self.left_margin + self.w_ * t,
                                                        self.top_margin + self.node_positions[n]*self.h_ + self.h_/2)
                    else:
                        line_ = QtGui.QGraphicsLineItem(self.left_margin + self.node_size + self.w_ * t,
                                                        self.top_margin + self.node_positions[n]*self.h_ + self.h_/2,
                                                        self.left_margin + self.node_size + self.w_ * t + highlight_line_len,
                                                        self.top_margin + self.node_positions[n]*self.h_ + self.h_/2)

                    line_.setPen(self.chunk_highlight_pen)
                    self.scene.addItem(line_)

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
        self.dehighlight_node()

        self.active_node = node
        it = self.get_node_item(node)
        it.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)
        it.setSelected(True)
        self.v.centerOn(QtCore.QPointF(it.pos().x(), it.pos().y()))

    def dehighlight_node(self, node=None):
        if not node:
            node = self.active_node

        if node:
            it = self.get_node_item(node)
            it.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)
            it.setSelected(False)
            self.active_node = None

    def on_context_menu(self, point):
        it = self.scene.itemAt(self.v.mapToScene(point))

        if isinstance(it, QtGui.QGraphicsPixmapItem):
            self.active_node = self.it_nodes[it]
            self.pop_menu_node.exec_(self.v.mapToGlobal(point))
        else:
            self.active_node = None

    def scene_clicked(self, point):
        it = self.scene.itemAt(point)

        if isinstance(it, QtGui.QGraphicsRectItem):
            br = it.boundingRect()
            print br.y(), br.x(), br.width(), br.height()
            it = self.scene.itemAt(br.x()+br.width()/2, br.y()+br.height()/2)

        if isinstance(it, QtGui.QGraphicsPixmapItem):
            # it is not a node:
            if it.pos().y() < self.top_margin:
                return

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

            else:
                self.active_row = int(round((it.pos().y() - self.top_margin) / (self.h_ + 0.0)))
                self.active_col = int(round(it.pos().x() / (self.w_ + 0.0)))
                print self.active_row, self.active_col
                self.draw_selection_rect()
                #self.highlight_node(n2)

            self.active_node = self.it_nodes[it]
        else:
            print "Fooo"
            self.connect_with_active = False
            self.join_with_active = False
            QtGui.QApplication.setOverrideCursor(QtCore.Qt.ArrowCursor)
            # self.active_node = None

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

    def get_im(self, n):
        im = self.frame_cache[n.frame_-self.frame_t].copy()

        vis = draw_points_crop(im, n.pts(), color=self.get_node_color(n), square=True)

        if vis.shape[0] > self.node_size or vis.shape[1] > self.node_size:
            vis = np.asarray(resize(vis, (self.node_size, self.node_size)) * 255, dtype=np.uint8)

        return cvimg2qtpixmap(vis)

    def get_node_color(self, n):
        return self.color_assignments[n]

    def draw_frames(self):
        centroids = []
        for g in self.nodes_groups:
            for n in g:
                centroids.append(n.centroid())

        roi = get_roi(np.array(centroids))
        m = self.frame_visu_margin

        h_, w_, _ = self.frame_cache[0].shape
        roi = ROI(max(0, roi.y() - m), max(0, roi.x() - m), min(roi.height() + 2*m, h_), min(roi.width() + 2*m, w_))
        self.w_ = max(roi.width() + 2, self.w_)
        self.top_margin = int(roi.height() + 1)
        self.crop_offset = roi.top_left_corner()

        special_case = 0 if len(self.nodes_groups) > 1 else 1
        for i in range(len(self.nodes_groups) + special_case):
            im = self.frame_cache[i].copy()

            if self.crop_visualize:
                if not(special_case and i > 0):
                    for r in self.nodes_groups[i]:
                        im = draw_points(im, r.pts(), color=self.get_node_color(r))

            crop = np.copy(im[roi.y():roi.y()+roi.height(), roi.x():roi.x()+roi.width(), :])
            cv2.putText(crop, str(self.frame_t+i), (1, 10), cv2.FONT_HERSHEY_PLAIN, 0.55, (255, 255, 255), 1, cv2.cv.CV_AA)

            pm = cvimg2qtpixmap(crop)
            it = self.scene.addPixmap(pm)

            off = (self.w_ - crop.shape[1] - self.node_size/2) / 2 if self.w_ > crop.shape[1] else 0
            off = max(0, off)

            it.setPos(1 + off + self.left_margin + i*(self.w_) - self.node_size/2, 0)
            self.crop_pixmaps_cache.append(pm)

            im = self.frame_cache[i]
            crop = np.copy(im[int(roi.y()):int(roi.y())+int(roi.height()), int(roi.x()):int(roi.x())+int(roi.width()), :])
            pm = cvimg2qtpixmap(crop)
            it = self.scene.addPixmap(pm)
            it.setPos(1 + off + self.left_margin + i*(self.w_) - self.node_size/2, 0)
            it.hide()
            self.crop_clear_frames_items.append(it)

        if special_case:
            i = 1
            im = self.frame_cache[i]
            crop = np.copy(im[int(roi.y()):int(roi.y())+int(roi.height()), int(roi.x()):int(roi.x())+int(roi.width()), :])
            pm = cvimg2qtpixmap(crop)
            it = self.scene.addPixmap(pm)
            it.setPos(1 + off + self.left_margin + i*(self.w_) - self.node_size/2, 0)
            it.hide()
            self.crop_clear_frames_items.append(it)

    def cache_frames(self):
        special_case = 0 if len(self.nodes_groups) > 1 else 1
        for i in range(len(self.nodes_groups)+special_case):
            if i == 0:
                im = self.vid.seek_frame(self.frame_t)
            else:
                im = self.vid.next_frame()

            # if not im:
            #     continue

            sf = self.project.other_parameters.img_subsample_factor
            if sf > 1.0:
                im = np.asarray(rescale(im, 1 / sf) * 255, dtype=np.uint8)

            self.frame_cache.append(im)

    def mark_merged(self, region, t_reversed=None):
        merged_t = region.frame_ - self.frame_t
        model_t = merged_t + 1 if t_reversed else merged_t - 1

        if len(self.nodes_groups[model_t]) > 0 and len(self.nodes_groups[merged_t]) > 0:
            t1_ = self.nodes_groups[model_t]

            objects = []
            for c1 in t1_:
                a = deepcopy(c1)
                if t_reversed:
                    a.frame_ -= 1
                else:
                    a.frame_ += 1

                objects.append(a)

            f = Fitting(region, objects, num_of_iterations=10)
            f.fit()

            return f.animals

    def draw_scene(self):
        for i in range(len(self.nodes_groups)):
            h_pos = 0
            for n in self.nodes_groups[i]:
                self.node_positions[n] = h_pos

                self.scene.addRect(self.left_margin + self.w_*i,
                                   self.top_margin + h_pos*self.h_,
                                   self.node_size,
                                   self.node_size,
                                   self.node_bg_color,
                                   self.node_bg_color)

                it = self.scene.addPixmap(self.get_im(n))

                self.it_nodes[it] = n
                off = (self.node_size - it.boundingRect().width()) / 2
                it.setPos(self.left_margin + self.w_ * i + off,
                          self.top_margin + h_pos * self.h_ + off)
                h_pos += 1

        for i in range(len(self.nodes_groups) - 1):
            for n in self.nodes_groups[i]:
                for _, n2, d in self.G.out_edges(n, data=True):
                    if 'chunk_ref' in d:
                        continue

                    try:
                        line_ = QtGui.QGraphicsLineItem(self.left_margin + self.node_size + self.w_*i,
                                                        self.top_margin + self.node_positions[n]*self.h_ + self.h_/2,
                                                        self.left_margin + self.w_ + i * self.w_,
                                                        self.top_margin + self.node_positions[n2]*self.h_ + self.h_/2)
                        line_.setPen(self.edge_pen)
                        self.scene.addItem(line_)
                    except:
                        print "potential problem in case_wdiget.py in draw_scene", n, n2
                        pass

    def process_suggested_config(self, suggested_config):
        l_ = []
        for n1, n2 in suggested_config.iteritems():
            l_.append([n1.frame_, n1, n2])

        self.suggested_config = sorted(l_, key=lambda k: k[0])

    def hide_visualization(self):
        for it in self.crop_clear_frames_items:
            if self.visualization_hidden:
                it.hide()
            else:
                it.show()

        self.visualization_hidden = not self.visualization_hidden
