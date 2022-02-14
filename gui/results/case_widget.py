__author__ = 'fnaiser'

from functools import partial

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from skimage.transform import rescale
from skimage.transform import resize

from core.animal import colors_
from gui.img_controls.gui_utils import cvimg2qtpixmap
from gui.img_controls.my_scene import MyScene
from gui.settings import Settings as S_
from utils.drawing.points import draw_points_crop, draw_points
from utils.roi import ROI, get_roi


class CaseWidget(QtWidgets.QWidget):
    def __init__(self, project, vertices_groups, suggested_config, vid, parent_widget, color_assignments=None):
        super(CaseWidget, self).__init__()

        self.project = project
        self.g = project.gm.g
        self.vertices_groups = vertices_groups
        self.regions_groups = self.get_regions_groups()
        self.parent = parent_widget
        self.vid = vid

        self.suggested_config = None
        self.num_of_nodes = 0
        for g in self.vertices_groups:
            for _ in g:
                self.num_of_nodes += 1

        self.process_suggested_config(suggested_config)

        self.node_size = 70
        self.frame_visu_margin = 100

        self.config_lines = []
        self.node_positions = []
        self.h_ = self.node_size + 2
        self.w_ = self.node_size + 100
        self.top_margin = 0
        self.left_margin = 10

        self.user_actions = []

        self.connect_with_active = False
        self.join_with_active = False

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

        self.frame_t = self.regions_groups[0][0].frame_
        self.opacity = 0.5

        if color_assignments:
            self.color_assignments = color_assignments
        else:
            self.color_assignments = {}

            i = 0
            chunk_nodes = set()
            for g in self.vertices_groups:
                for n in g:
                    ch, _ = self.project.gm.is_chunk(n)
                    if ch and ch.length() > 1:
                        chunk_nodes.add(n)
                        self.color_assignments[n] = tuple(ch.color) + (self.opacity, )
                    else:
                        self.color_assignments[n] = colors_[i % len(colors_)] + (self.opacity,)
                    i += 1

            for _, n1, n2 in reversed(self.suggested_config):
                if n2 in chunk_nodes and n1 not in chunk_nodes:
                    self.color_assignments[n1] = self.color_assignments[n2]
                    chunk_nodes.add(n1)

            for _, n1, n2 in self.suggested_config:
                self.color_assignments[n2] = self.color_assignments[n1]

        self.pop_menu_node = QtWidgets.QMenu(self)
        self.action_remove_node = QtGui.QAction('remove', self)
        self.action_remove_node.triggered.connect(self.remove_node_)

        self.action_partially_confirm = QtGui.QAction('confirm this connection', self)
        self.action_partially_confirm.triggered.connect(self.parent.partially_confirm)

        # self.action_mark_merged = QtGui.QAction('merged', self)
        # self.action_mark_merged.triggered.connect(self.mark_merged)

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
        self.row_up.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Up))
        self.addAction(self.row_up)

        self.row_down = QtGui.QAction('row down', self)
        self.row_down.triggered.connect(partial(self.row_changed, 1))
        self.row_down.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Down))
        self.addAction(self.row_down)

        self.col_left = QtGui.QAction('col left', self)
        self.col_left.triggered.connect(partial(self.col_changed, -1))
        self.col_left.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left))
        self.addAction(self.col_left)

        self.col_right = QtGui.QAction('col right', self)
        self.col_right.triggered.connect(partial(self.col_changed, 1))
        self.col_right.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right))
        self.addAction(self.col_right)

        # self.hide_visualization_a = QtGui.QAction('hide visualization', self)
        # self.hide_visualization_a.triggered.connect(self.hide_visualization)
        # self.hide_visualization_a.setShortcut(S_.controls.hide_show)
        # self.addAction(self.hide_visualization_a)

        self.pop_menu_node.addAction(self.action_remove_node)
        # self.pop_menu_node.addAction(self.action_mark_merged)
        self.pop_menu_node.addAction(self.action_partially_confirm)
        self.pop_menu_node.addAction(self.new_region_t1)
        self.pop_menu_node.addAction(self.new_region_t2)
        self.pop_menu_node.addAction(self.connect_with)
        self.pop_menu_node.addAction(self.join_with)
        self.pop_menu_node.addAction(self.get_info_action)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.v = QtWidgets.QGraphicsView()
        self.scene = MyScene()

        self.edge_pen = QtGui.QPen(QtCore.Qt.PenStyle.SolidLine)
        self.edge_pen.setColor(QtGui.QColor(0, 0, 0, 0x16))
        self.edge_pen.setWidth(1)

        self.strong_edge_pen = QtGui.QPen(QtCore.Qt.PenStyle.SolidLine)
        self.strong_edge_pen.setColor(QtGui.QColor(0, 255, 0, 0x78))
        self.strong_edge_pen.setWidth(2)

        self.node_bg_color = QtGui.QColor(230, 230, 230, 230)
        op = 100
        self.bg_light_stripe = QtGui.QColor(255, 255, 255, op)
        self.bg_dark_stripe = QtGui.QColor(255, 255, 190, op)

        self.bg_light_stripe_r = QtGui.QColor(212, 250, 255, op)
        self.bg_dark_stripe_r = QtGui.QColor(242, 220, 232, op)

        self.chunk_highlight_pen = QtGui.QPen(QtCore.Qt.PenStyle.DotLine)
        self.chunk_highlight_pen.setColor(QtGui.QColor(255, 0, 0, 0x78))
        self.chunk_highlight_pen.setWidth(2)

        self.grid_pen = QtGui.QPen(QtCore.Qt.PenStyle.SolidLine)
        self.grid_pen.setColor(QtGui.QColor(135, 185, 201, 0x86))
        self.grid_pen.setWidth(1)

        self.grid_mark_pen = QtGui.QPen(QtCore.Qt.PenStyle.SolidLine)
        self.grid_mark_pen.setColor(QtGui.QColor(0, 0, 0, 0xff))
        self.grid_mark_pen.setWidth(2)

        self.layout().addWidget(self.v)
        self.v.setScene(self.scene)

        self.scene.clicked.connect(self.scene_clicked)
        self.v.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        self.cache_frames()
        self.draw_frames()
        self.draw_grid()
        self.draw_scene()
        self.highlight_chunk_nodes()
        self.active_node = None
        # self.highlight_node(self.nodes_groups[self.active_col][self.active_row])
        # self.draw_selection_rect()

        self.v.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.v.customContextMenuRequested[QPoint].connect(self.on_context_menu)

        for _, n1, n2 in self.suggested_config:
            if n1 not in self.color_assignments:
                self.color_assignments[n1] = colors_[10]

            self.color_assignments[n2] = self.color_assignments[n1]

            r1 = self.project.gm.region(n1)
            t = r1.frame_ - self.frame_t
            line_ = QtWidgets.QGraphicsLineItem(self.left_margin + self.node_size + self.w_ * t,
                                            self.top_margin + self.node_positions[n1] * self.h_ + self.h_ / 2,
                                            self.left_margin + self.w_ * (t + 1),
                                            self.top_margin + self.node_positions[n2] * self.h_ + self.h_ / 2)
            line_.setPen(self.strong_edge_pen)
            self.config_lines.append(line_)
            self.scene.addItem(line_)

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet("border: 0px")
        self.draw_selection_rect()

    def get_regions_groups(self):
        rg = []
        for g in self.vertices_groups:
            rg.append([])
            for v in g:
                rg[-1].append(self.project.gm.region(v))

        return rg

    def get_info(self):
        n = self.active_node

        r = self.project.gm.region(n)

        virtual = False
        try:
            if r.is_origin_interaction():
                virtual = True
        except:
            pass

        vertex = self.project.gm.g.vertex(int(n))
        best_out_score, best_out_n = self.project.gm.get_2_best_out_vertices(vertex)
        best_out = best_out_score[0]

        new_s = -1
        do = -1
        dt = -1
        if best_out_n[0]:
            new_s, do, dt = self.project.solver.assignment_score_pos_orient(r, self.project.gm.region(best_out_n[0]))

        best_in_score, _ = self.project.gm.get_2_best_in_vertices(vertex)
        best_in = best_in_score[0]

        ch, _ = self.project.gm.is_chunk(vertex)
        ch_info = str(ch)

        ch_start = 1
        ch_end = -1

        is_merged = False
        if ch:
            ch_start = ch.start_frame()
            ch_end = ch.end_frame()

            ch_start_vertex = self.project.gm.g.vertex(ch.start_node())

            # ignore chunks of merged regions
            for e in ch_start_vertex.in_edges():
                if self.project.gm.g.ep['score'][e] == 0 and ch_start_vertex.in_degree() > 1:
                    is_merged = True

        QtWidgets.QMessageBox.about(self, "My message box",
                                "ID = %i\nArea = %i\nframe=%i\nCentroid = %s\nMargin = %i\n"
                                "Is virtual: %s\nBest in = %s, (%d)\nBest out = %s (%d)\nChunk info = %s\n"
                                "Chunk start: %d end: %d\ntest:%s\nnew_s:%f, %f, %f\ntheta: %f\n" %
                                (int(n), r.area(), r.frame_, str(r.centroid()), r.margin_, str(virtual),
                                 str(best_in_score[0]) + ', ' + str(best_in_score[1]), vertex.in_degree(),
                                 str(best_out_score[0]) + ', ' + str(best_out_score[1]), vertex.out_degree(), ch_info,
                                 ch_start, ch_end, str(is_merged), new_s, do, dt, r.theta_
                                ))

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

        col_it = QtWidgets.QGraphicsRectItem(self.left_margin + self.w_ * c - self.node_size / 2,
                                         self.top_margin, self.w_, self.h_ * self.rows)
        if self.active_col_it:
            self.scene.removeItem(self.active_col_it)

        self.active_col_it = col_it
        self.scene.addItem(col_it)
        col_it.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        col_it.setZValue(-1)

        row_it = QtWidgets.QGraphicsRectItem(self.left_margin - self.node_size / 2,
                                         self.top_margin + self.h_ * r, self.w_ * self.cols, self.h_)

        if self.active_row_it:
            self.scene.removeItem(self.active_row_it)

        self.active_row_it = row_it
        self.scene.addItem(row_it)
        row_it.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        row_it.setZValue(-1)

        n = self.vertices_groups[self.active_col][self.active_row]
        self.highlight_node(n)
        self.active_node = n

    def draw_grid(self):
        rows = 0
        for g in self.vertices_groups:
            rows = max(rows, len(g))

        self.rows = rows

        cols = len(self.vertices_groups)
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
                line_ = QtWidgets.QGraphicsLineItem(self.left_margin - self.node_size / 2,
                                                self.top_margin + self.h_ * r - 1,
                                                self.left_margin - self.node_size / 2 + self.w_ * cols,
                                                self.top_margin + self.h_ * r - 1)

                line_.setPen(self.grid_pen)
                self.scene.addItem(line_)

            for c in range(1, cols):
                line_ = QtWidgets.QGraphicsLineItem(self.left_margin + self.w_ * c - self.node_size / 2,
                                                self.top_margin,
                                                self.left_margin + self.w_ * c - self.node_size / 2,
                                                self.top_margin + self.h_ * rows)

                line_.setPen(self.grid_pen)
                self.scene.addItem(line_)
        else:
            mark_w = 10
            for r in range(rows + 1):
                a = (self.w_ - self.node_size) / 2
                line_ = QtWidgets.QGraphicsLineItem(self.left_margin - a - mark_w / 2,
                                                self.top_margin + self.h_ * r - 1,
                                                self.left_margin - a + mark_w / 2,
                                                self.top_margin + self.h_ * r - 1)

                line_.setPen(self.grid_mark_pen)
                self.scene.addItem(line_)

            for c in range(cols + 1):
                line_ = QtWidgets.QGraphicsLineItem(self.left_margin + self.w_ * c - self.node_size / 2,
                                                self.top_margin + self.h_ * rows - mark_w / 2,
                                                self.left_margin + self.w_ * c - self.node_size / 2,
                                                self.top_margin + self.h_ * rows + mark_w / 2)

                line_.setPen(self.grid_mark_pen)
                self.scene.addItem(line_)

    def remove_node_(self):
        self.parent.remove_region(self.active_node)

    def highlight_chunk_nodes(self):
        highlight_line_len = self.node_size / 2

        for g in self.vertices_groups:
            for n in g:
                ch, t_rev = self.project.gm.is_chunk(n)
                if ch and ch.length() > 1:
                    t = self.project.gm.region(n).frame_ - self.frame_t

                    if t_rev:
                        line_ = QtWidgets.QGraphicsLineItem(self.left_margin + self.w_ * t - highlight_line_len,
                                                        self.top_margin + self.node_positions[
                                                            n] * self.h_ + self.h_ / 2,
                                                        self.left_margin + self.w_ * t,
                                                        self.top_margin + self.node_positions[
                                                            n] * self.h_ + self.h_ / 2)
                    else:
                        line_ = QtWidgets.QGraphicsLineItem(self.left_margin + self.node_size + self.w_ * t,
                                                        self.top_margin + self.node_positions[
                                                            n] * self.h_ + self.h_ / 2,
                                                        self.left_margin + self.node_size + self.w_ * t + highlight_line_len,
                                                        self.top_margin + self.node_positions[
                                                            n] * self.h_ + self.h_ / 2)

                    line_.setPen(self.chunk_highlight_pen)
                    self.scene.addItem(line_)

    def get_node_item(self, node):
        n_it = None
        for it, key in self.it_nodes.items():
            if key == node:
                n_it = it

        return n_it

    def get_node_at_pos(self, node_pos):
        i = 0
        for g in self.vertices_groups:
            for n in g:
                if i == node_pos:
                    return n

                i += 1

        return None

    def get_node_item_at_pos(self, node_pos):
        n_key = None

        for key, pos in self.node_positions.items():
            if self.frame_t < key.frame_:
                pos += len(self.vertices_groups[0])

            if pos == node_pos:
                n_key = key
                break

        n_it = None

        for it, key in self.it_nodes.items():
            if key == n_key:
                n_it = it
                break

        return n_it

    def connect_with_(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.connect_with_active = True

    def join_with_(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.join_with_active = True

    def highlight_node(self, node):
        self.dehighlight_node()

        self.active_node = node
        it = self.get_node_item(node)
        it.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        it.setSelected(True)
        self.v.centerOn(QtCore.QPointF(it.pos().x(), it.pos().y()))

    def dehighlight_node(self, node=None):
        if not node:
            node = self.active_node

        if node:
            it = self.get_node_item(node)
            it.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            it.setSelected(False)
            self.active_node = None

    def on_context_menu(self, point):
        it = self.scene.itemAt(self.v.mapToScene(point))

        if isinstance(it, QtWidgets.QGraphicsPixmapItem):
            self.active_node = self.it_nodes[it]
            self.pop_menu_node.exec_(self.v.mapToGlobal(point))
        else:
            self.active_node = None

    def scene_clicked(self, point):
        it = self.scene.itemAt(point)

        if isinstance(it, QtWidgets.QGraphicsRectItem):
            br = it.boundingRect()
            it = self.scene.itemAt(br.x() + br.width() / 2, br.y() + br.height() / 2)

        if isinstance(it, QtWidgets.QGraphicsPixmapItem):
            # it is not a node:
            if it.pos().y() < self.top_margin:
                return

            n1 = self.active_node
            n2 = self.it_nodes[it]

            if self.connect_with_active:
                if self.project.gm.region(self.active_node).frame_ > self.project.gm.region(self.it_nodes[it]).frame_:
                    n1 = self.it_nodes[it]
                    n2 = self.active_node

                self.parent.confirm_edges([(n1, n2)])
                self.connect_with_active = False
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.ArrowCursor)

            elif self.join_with_active:
                self.parent.join_regions(n1, n2)
                self.join_with_active = False
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.ArrowCursor)

            else:
                self.active_row = int(round((it.pos().y() - self.top_margin) / (self.h_ + 0.0)))
                self.active_col = int(round(it.pos().x() / (self.w_ + 0.0)))
                self.draw_selection_rect()
                # self.highlight_node(n2)

            self.active_node = self.it_nodes[it]
        else:
            self.connect_with_active = False
            self.join_with_active = False
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.ArrowCursor)
            # self.active_node = None

    def confirm_clicked(self):
        if len(self.vertices_groups) < 1:
            return

        w = len(self.vertices_groups[0])
        for g in self.vertices_groups:
            if w != len(g):
                print("UNBALANCED configuration, ignoring confirmation")
                return
            w = len(g)

        pairs = []

        for _, n1, n2 in self.suggested_config:
            pairs.append((n1, n2))

        self.parent.confirm_edges(pairs)

    def get_im(self, n):
        r = self.project.gm.region(n)
        im = self.frame_cache[r.frame_ - self.frame_t].copy()

        vis = draw_points_crop(im, r.pts(), color=self.get_node_color(n), square=True)

        if vis.shape[0] > self.node_size or vis.shape[1] > self.node_size:
            vis = np.asarray(resize(vis, (self.node_size, self.node_size)) * 255, dtype=np.uint8)

        return cvimg2qtpixmap(vis)

    def get_node_color(self, n):
        return self.color_assignments[n]

    def draw_frames(self):
        centroids = []
        for g in self.regions_groups:
            for n in g:
                centroids.append(n.centroid())

        roi = get_roi(np.array(centroids))
        m = self.frame_visu_margin

        h_, w_, _ = self.frame_cache[0].shape
        roi = ROI(max(0, roi.y() - m), max(0, roi.x() - m), min(roi.height() + 2 * m, h_), min(roi.width() + 2 * m, w_))
        self.w_ = max(roi.width() + 2, self.w_)
        self.top_margin = int(roi.height() + 1)
        self.crop_offset = roi.top_left_corner()

        special_case = 0 if len(self.vertices_groups) > 1 else 1
        for i in range(len(self.vertices_groups) + special_case):
            im = self.frame_cache[i].copy()

            if self.crop_visualize:
                if not (special_case and i > 0):
                    for v, r in zip(self.vertices_groups[i], self.regions_groups[i]):
                        im = draw_points(im, r.pts(), color=self.get_node_color(v))

            crop = np.copy(
                im[int(roi.y()):int(roi.y()) + int(roi.height()), int(roi.x()):int(roi.x()) + int(roi.width()), :])
            cv2.putText(crop, str(self.frame_t + i), (1, 10), cv2.FONT_HERSHEY_PLAIN, 0.55, (255, 255, 255), 1,
                        cv2.cv.CV_AA)

            pm = cvimg2qtpixmap(crop)
            it = self.scene.addPixmap(pm)

            off = (self.w_ - crop.shape[1] - self.node_size / 2) / 2 if self.w_ > crop.shape[1] else 0
            off = max(0, off)

            it.setPos(1 + off + self.left_margin + i * (self.w_) - self.node_size / 2, 0)
            self.crop_pixmaps_cache.append(pm)

            im = self.frame_cache[i]
            crop = np.copy(
                im[int(roi.y()):int(roi.y()) + int(roi.height()), int(roi.x()):int(roi.x()) + int(roi.width()), :])
            pm = cvimg2qtpixmap(crop)
            it = self.scene.addPixmap(pm)
            it.setPos(1 + off + self.left_margin + i * (self.w_) - self.node_size / 2, 0)
            it.hide()
            self.crop_clear_frames_items.append(it)

        if special_case:
            i = 1
            im = self.frame_cache[i]
            crop = np.copy(
                im[int(roi.y()):int(roi.y()) + int(roi.height()), int(roi.x()):int(roi.x()) + int(roi.width()), :])
            pm = cvimg2qtpixmap(crop)
            it = self.scene.addPixmap(pm)
            it.setPos(1 + off + self.left_margin + i * (self.w_) - self.node_size / 2, 0)
            it.hide()
            self.crop_clear_frames_items.append(it)

    def cache_frames(self):
        special_case = 0 if len(self.vertices_groups) > 1 else 1
        for i in range(len(self.vertices_groups) + special_case):
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

    # def mark_merged(self, region, t_reversed=None):
    #     merged_t = region.frame_ - self.frame_t
    #     model_t = merged_t + 1 if t_reversed else merged_t - 1
    #
    #     if len(self.vertices_groups[model_t]) > 0 and len(self.vertices_groups[merged_t]) > 0:
    #         t1_ = self.vertices_groups[model_t]
    #
    #         objects = []
    #         for c1 in t1_:
    #             a = deepcopy(c1)
    #             if t_reversed:
    #                 a.frame_ -= 1
    #             else:
    #                 a.frame_ += 1
    #
    #             objects.append(a)
    #
    #         f = Fitting(region, objects, num_of_iterations=10)
    #         f.fit()
    #
    #         return f.animals

    def draw_scene(self):
        for i in range(len(self.vertices_groups)):
            h_pos = 0
            for n in self.vertices_groups[i]:
                self.node_positions[n] = h_pos

                self.scene.addRect(self.left_margin + self.w_ * i,
                                   self.top_margin + h_pos * self.h_,
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

        for i in range(len(self.vertices_groups) - 1):
            for n in self.vertices_groups[i]:
                for n2 in n.out_neighbors():
                    ch, ch_end = self.project.gm.is_chunk(n2)
                    if ch_end:
                        continue
                    try:
                        line_ = QtWidgets.QGraphicsLineItem(self.left_margin + self.node_size + self.w_ * i,
                                                        self.top_margin + self.node_positions[
                                                            n] * self.h_ + self.h_ / 2,
                                                        self.left_margin + self.w_ + i * self.w_,
                                                        self.top_margin + self.node_positions[
                                                            n2] * self.h_ + self.h_ / 2)
                        line_.setPen(self.edge_pen)
                        self.scene.addItem(line_)
                    except:
                        print("potential problem in case_wdiget.py in draw_scene", n, n2)
                        pass

    def process_suggested_config(self, suggested_config):
        l_ = []
        for v1, v2 in suggested_config.items():
            r1 = self.project.gm.region(v1)
            l_.append([r1.frame_, v1, v2])

        self.suggested_config = sorted(l_, key=lambda k: k[0])

    def hide_visualization(self):
        for it in self.crop_clear_frames_items:
            if self.visualization_hidden:
                it.hide()
            else:
                it.show()

        self.visualization_hidden = not self.visualization_hidden
