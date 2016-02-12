import threading
from PyQt4 import QtGui, QtCore

from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication, QLabel, QSizePolicy

import computer as comp
from gui.graph_view.edge import EdgeGraphical, ChunkGraphical
from gui.graph_view.info_manager import InfoManager
from gui.graph_view.node import Node
from gui.graph_view.node_zoom_manager import NodeZoomManager
from gui.gui_utils import cvimg2qtpixmap
from gui.img_controls.my_scene import MyScene
from gui.img_controls.my_view_zoomable import MyViewZoomable
from vis_loader import DEFAULT_TEXT, GAP, \
    MINIMUM, SPACE_BETWEEN_HOR, SPACE_BETWEEN_VER, WIDTH

__author__ = 'Simon Mandlik'


class \
        GraphVisualizer(QtGui.QWidget):
    """
    Requires list of regions and list of edge-tuples (node1, node2, type - chunk, line or partial, sureness).
    Those can be passed in constructor or using a method add_objects
    """

    def __init__(self, loader, img_manager, show_vertically=True, compress_axis=True, dynamically=True):
        super(GraphVisualizer, self).__init__()
        self.regions = set()
        self.regions_list = []
        self.edges = set()
        self.edges_list = []
        self.frames_columns = {}
        self.columns = []
        self.img_manager = img_manager
        self.relative_margin = loader.relative_margin
        self.width = loader.width
        self.height = loader.height
        self.loader = loader
        self.dynamically = dynamically
        self.setLayout(QtGui.QVBoxLayout())

        self.chunk_detail_scroll_horizontal = QtGui.QScrollArea()
        self.chunk_detail_scroll_vertical = QtGui.QScrollArea()
        self.chunk_detail_scroll_horizontal.setWidgetResizable(True)
        self.chunk_detail_scroll_vertical.setWidgetResizable(True)

        # hscroll_shint = self.chunk_detail_scroll_horizontal.horizontalScrollBar().sizeHint()
        self.chunk_detail_scroll_horizontal.setFixedHeight(self.height * 2)

        # vscroll_shint = self.chunk_detail_scroll_vertical.verticalScrollBar().sizeHint()
        self.chunk_detail_scroll_vertical.setFixedWidth(self.width * 2)

        self.chunk_detail_widget_horizontal = QtGui.QWidget()
        self.chunk_detail_widget_horizontal.setLayout(QtGui.QHBoxLayout())
        self.chunk_detail_widget_vertical = QtGui.QWidget()
        self.chunk_detail_widget_vertical.setLayout(QtGui.QVBoxLayout())

        self.chunk_detail_scroll_horizontal.setWidget(self.chunk_detail_widget_horizontal)
        self.chunk_detail_scroll_vertical.setWidget(self.chunk_detail_widget_vertical)

        self.upper_part = QtGui.QWidget()
        self.upper_part.setLayout(QtGui.QHBoxLayout())

        self.view = MyViewZoomable(self)
        self.view.setMouseTracking(True)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scene = MyScene()
        self.scene_width = 0
        self.view.setScene(self.scene)
        self.view.setLayout(QtGui.QHBoxLayout())
        # self.view.layout().addWidget(self.chunk_detail_scroll_vertical)
        self.scene.clicked.connect(self.scene_clicked)

        self.upper_part.layout().addWidget(self.view)
        self.upper_part.layout().addWidget(self.chunk_detail_scroll_vertical)

        self.layout().addWidget(self.upper_part)
        self.layout().addWidget(self.chunk_detail_scroll_horizontal)

        self.chunk_detail_scroll_horizontal.hide()
        self.chunk_detail_scroll_vertical.hide()

        self.text = QtGui.QLabel(DEFAULT_TEXT)
        self.text.setAlignment(QtCore.Qt.AlignCenter)
        stylesheet = "font: 25px"
        self.text.setStyleSheet(stylesheet)
        self.layout().addWidget(self.text)

        self.selected = []
        self.node_zoom_manager = NodeZoomManager()
        self.info_manager = InfoManager(loader)

        self.wheel_count = 1
        self.loaded = set()

        self.selected_in_menu = None
        self.menu_node = QtGui.QMenu(self)
        self.menu_edge = QtGui.QMenu(self)
        self.show_info_menu_action = QtGui.QAction('Show Info', self)
        self.show_info_menu_action.triggered.connect(self.show_info_action_method)
        self.show_zoom_menu_action = QtGui.QAction('Show Zoom', self)
        self.show_zoom_menu_action.triggered.connect(self.show_zoom_action_method)
        self.hide_zoom_menu_action = QtGui.QAction('Hide Zoom', self)
        self.hide_zoom_menu_action.triggered.connect(self.hide_zoom_action_method)
        self.hide_info_menu_action = QtGui.QAction('Hide Info', self)
        self.hide_info_menu_action.triggered.connect(self.hide_info_action_method)
        self.menu_node.addAction(self.show_info_menu_action)
        self.menu_node.addAction(self.hide_info_menu_action)
        self.menu_node.addAction(self.show_zoom_menu_action)
        self.menu_node.addAction(self.hide_zoom_menu_action)
        self.menu_edge.addAction(self.show_info_menu_action)
        self.menu_edge.addAction(self.hide_info_menu_action)
        self.view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.connect(self.view, QtCore.SIGNAL('customContextMenuRequested(const QPoint&)'), self.menu)

        self.show_vertically = show_vertically
        self.show_vertically_action = QtGui.QAction('show vertically', self)
        self.show_vertically_action.triggered.connect(self.toggle_show_vertically)
        self.show_vertically_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_V))
        self.addAction(self.show_vertically_action)

        self.compress_axis = compress_axis
        self.compress_axis_action = QtGui.QAction('compress axis', self)
        self.compress_axis_action.triggered.connect(self.compress_axis_toggle)
        self.compress_axis_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_C))
        self.addAction(self.compress_axis_action)

        self.stretch_action = QtGui.QAction('stretch', self)
        self.stretch_action.triggered.connect(self.stretch)
        self.stretch_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_W))
        self.addAction(self.stretch_action)

        self.shrink_action = QtGui.QAction('shrink', self)
        self.shrink_action.triggered.connect(self.shrink)
        self.shrink_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Q))
        self.addAction(self.shrink_action)

        self.show_info_action = QtGui.QAction('show_info', self)
        self.show_info_action.triggered.connect(self.info_manager.show_all_info)
        self.show_info_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_A))
        self.addAction(self.show_info_action)

        self.hide_info_action = QtGui.QAction('hide_info', self)
        self.hide_info_action.triggered.connect(self.info_manager.hide_all_info)
        self.hide_info_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_S))
        self.addAction(self.hide_info_action)

        self.show_zoom_node_action = QtGui.QAction('show_zoom', self)
        self.show_zoom_node_action.triggered.connect(self.node_zoom_manager.show_zoom_all)
        self.show_zoom_node_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_T))
        self.addAction(self.show_zoom_node_action)

        self.hide_zoom_node_action = QtGui.QAction('hide_zoom', self)
        self.hide_zoom_node_action.triggered.connect(self.node_zoom_manager.hide_zoom_all)
        self.hide_zoom_node_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Y))
        self.addAction(self.hide_zoom_node_action)

        if len(loader.edges) + len(loader.regions) > 0:
            self.add_objects(loader.regions, loader.edges)

    def show_info_action_method(self):
        if self.selected_in_menu:
            self.info_manager.show_info(self.selected_in_menu)

    def hide_info_action_method(self):
        if self.selected_in_menu:
            self.info_manager.hide_info(self.selected_in_menu)

    def show_zoom_action_method(self):
        if self.selected_in_menu and isinstance(self.selected_in_menu, Node):
            self.node_zoom_manager.show_zoom(self.selected_in_menu)

    def hide_zoom_action_method(self):
        if self.selected_in_menu and isinstance(self.selected_in_menu, Node):
            self.node_zoom_manager.hide_zoom(self.selected_in_menu)

    def menu(self, point):
        it = self.scene.itemAt(self.view.mapToScene(point))
        self.selected_in_menu = it
        if isinstance(it, Node):
            self.menu_node.exec_(self.view.mapToGlobal(point))
        elif isinstance(it, EdgeGraphical):
            self.menu_edge.exec_(self.view.mapToGlobal(point))

    def show_chunk_pictures_label(self, chunk):
        self.hide_chunk_pictures_widget()
        widget = self.chunk_detail_widget_vertical if self.show_vertically else self.chunk_detail_widget_horizontal

        region_chunk = self.loader.chunks_region_chunks[chunk]
        frames = list(range(chunk[0].frame_, chunk[1].frame_ + 1))
        freq, none = QtGui.QInputDialog.getInt(self, 'Input Dialog', 'Enter frequency:', value=1, min=1)

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        im = self.img_manager.get_whole_img(frames[0])
        alpha = np.zeros((im.shape[0], im.shape[1]), dtype=np.int32)
        alpha2 = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.int32)

        centroids = []

        incr = 1
        for frame in frames[::freq]:
            r = region_chunk[frame - region_chunk.start_frame()]
            centroids.append(r.centroid())

            img = self.img_manager.get_crop(frame, r,  width=self.width, height=self.height, relative_margin=self.relative_margin)

            alpha[r.pts()[:, 0], r.pts()[:, 1]] += 1
            alpha2[r.pts()[:, 0], r.pts()[:, 1], 1] += 1
            alpha2[r.pts()[:, 0], r.pts()[:, 1], 2] += 1
            alpha2[r.pts()[:, 0], r.pts()[:, 1], 0] = incr

            incr += 1

            pixmap = cvimg2qtpixmap(img)
            label = QtGui.QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            widget.layout().addWidget(label)

        # alpha2[:, :, 2] = alpha2[:, :, 2]**0.5

        centroids = np.array(centroids)

        plt.figure(1)
        plt.imshow(alpha)
        plt.set_cmap('viridis')

        plt.figure(2)
        plt.imshow(alpha)
        plt.set_cmap('jet')

        centr_step = 3
        centroids = centroids[::centr_step, :]

        plt.scatter(centroids[:, 1], centroids[:, 0], s=8, c=range(len(centroids)), edgecolors='None', cmap=mpl.cm.afmhot)
        # plt.figure(2)
        # plt.imshow(alpha2)
        plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
        plt.show()

        scroll = self.chunk_detail_scroll_vertical if self.show_vertically else self.chunk_detail_scroll_horizontal
        scroll.show()

    def hide_chunk_pictures_widget(self):
        widget = self.chunk_detail_widget_vertical if self.show_vertically else self.chunk_detail_widget_horizontal
        for child in widget.findChildren(QLabel):
            widget.layout().removeWidget(child)
            child.hide()
        self.chunk_detail_scroll_horizontal.hide()
        self.chunk_detail_scroll_vertical.hide()

    def swap_chunk_pictures_widgets(self):
        widget_a = self.chunk_detail_widget_horizontal if self.show_vertically else self.chunk_detail_widget_vertical
        widget_b = self.chunk_detail_widget_vertical if self.show_vertically else self.chunk_detail_widget_horizontal
        for child in widget_a.findChildren(QLabel):
            widget_a.layout().removeWidget(child)
            widget_b.layout().addWidget(child)
        if self.show_vertically:
            self.chunk_detail_scroll_horizontal.hide()
            self.chunk_detail_scroll_vertical.show()
        else:
            self.chunk_detail_scroll_vertical.hide()
            self.chunk_detail_scroll_horizontal.show()

    def scene_clicked(self, click_pos):
        item = self.scene.itemAt(click_pos)
        if item is None:
            self.selected = []
            self.node_zoom_manager.remove_all()
            self.info_manager.remove_info_all()
        else:
            if isinstance(item, EdgeGraphical):
                self.info_manager.add(item)
                if isinstance(item, ChunkGraphical):
                 self.show_chunk_pictures_label(item.core_obj)
            elif isinstance(item, Node):
                self.node_zoom_manager.add(item)
                self.info_manager.add(item)
            self.selected.append(item)

    def stretch(self):
        global SPACE_BETWEEN_HOR
        SPACE_BETWEEN_HOR *= 1.5
        self.redraw()

    def shrink(self):
        global SPACE_BETWEEN_HOR
        SPACE_BETWEEN_HOR *= 0.5
        self.redraw()

    def compute_positions(self):
        for edge in self.edges:
            if edge[2] == "chunk":
                self.find_suitable_position_chunk(edge)
            elif edge[2] == "line":
                self.find_suitable_position_line(edge)
            else:
                self.find_suitable_position_partial(edge)

    def add_sole_nodes(self):
        for node in self.regions:
            col = self.frames_columns[node.frame_]
            if col.contains(node):
                continue
            else:
                position = 0
                while not self.frames_columns[node.frame_].is_free(position, node):
                    position += 1
                else:
                    self.add_node_to_column(node, node.frame_, position)

    def add_node_to_column(self, node, column_frame, position):
        self.frames_columns[column_frame].add_object(node, position)

    def find_suitable_position_chunk(self, edge):
        node_1 = edge[0]
        node_2 = edge[1]
        col1 = self.frames_columns[node_1.frame_]
        col2 = self.frames_columns[node_2.frame_]
        if col1.contains(node_1) or col2.contains(node_2):
            self.find_suitable_position_semi_placed_chunk(edge, col1, col2, node_1, node_2)
        else:
            self.find_suitable_position_fresh_chunk(edge, node_1, node_2)

    def find_suitable_position_fresh_chunk(self, edge, node_1, node_2):
        # import time
        # time2 = time.time()
        position = 0
        while True:
            if self.is_line_free(position, node_1.frame_, node_2.frame_):
                # time1 = time.time()
                start = node_1.frame_
                end = node_2.frame_
                frame = start
                while frame <= end:
                    column = self.get_next_to_column(frame - 1, "right")
                    frame = column.get_end_frame()
                    if frame == start or frame == end or position == self.frames_columns[start].get_position_item(node_1):
                        column.add_object(edge, position)
                    frame += 1
                # print("inside {0}.".format(time.time() - time1))
                break
            else:
                position += 1
        # print("outside {0}.".format(time.time() - time2))

    def find_suitable_position_semi_placed_chunk(self, edge, col1, col2, node_1, node_2):
        """Finds the best position for semi-placed chunk. This situation should never happen!
        """
        # if col1.contains(node_1) and col2.contains(node_2):
        #     return
        if col1.contains(node_1) and not col2.contains(node_2):
            position1, position2 = self.find_nearest_free_slot(node_1, node_2)
            col2.add_object(edge, position2)
        if not col1.contains(node_1) and col2.contains(node_2):
            position1, position2 = self.find_nearest_free_slot(node_2, node_1)
            col1.add_object(edge, position2)

    def is_line_free(self, position, start_frame, end_frame):
        # import time
        # time1 = time.time()
        while start_frame <= end_frame:
            column = self.get_next_to_column(start_frame - 1, "right")
            start_frame = column.get_end_frame() + 1
            if not column.is_free(position):
                # print("fa {0}.".format(time.time() - time1))
                return False
        # print("f {0}.".format(time.time() - time1))
        return True

    def find_suitable_position_line(self, edge):
        node_1 = edge[0]
        node_2 = edge[1]
        contains_1 = self.frames_columns[node_1.frame_].contains(node_1)
        contains_2 = self.frames_columns[node_2.frame_].contains(node_2)
        if contains_1 and contains_2:
            return
        elif contains_1:
            position_1, position_2 = self.find_nearest_free_slot(node_1, node_2)
        elif contains_2:
            position_2, position_1 = self.find_nearest_free_slot(node_2, node_1)
        else:
            position_1 = -1
            position_2 = None
            offset_list = [0, 1, -1, 2, -2]
            while position_2 is None:
                position_1 += 1
                for num in offset_list:
                    if node_2.frame_ - node_1.frame_ == 1:
                        if self.frames_columns[node_1.frame_].is_free(position_1, node_1) and \
                         self.frames_columns[node_2.frame_].is_free(position_1 + num, node_2):
                            position_2 = position_1 + num
                            break
                    elif self.is_line_free(position_1, node_1.frame_, node_2.frame_) and \
                            self.is_line_free(position_1 + num, node_1.frame_, node_2.frame_):
                        position_2 = position_1 + num
                        break

        self.add_node_to_column(node_1, node_1.frame_, position_1)
        self.add_node_to_column(node_2, node_2.frame_, position_2)

    def find_suitable_position_partial(self, edge):
        if edge[0] is None or not edge[0] in self.regions:
            node = edge[1]
            direction = "left"
        else:
            node = edge[0]
            direction = "right"

        if not self.frames_columns.has_key(node.frame_):
            return
        column = self.frames_columns[node.frame_]
        # next_column = self.get_next_to_column(node.frame_, direction)

        if not column.contains(node):
            position = 0
            occupied = True
            while occupied:
                if column.is_free(position):
                        # and next_column.is_free(position):
                    occupied = False
                position += 1

            column.add_object(node, position)

    def find_nearest_free_slot(self, node_placed, node_free):
        position = self.frames_columns[node_placed.frame_].get_position_item(node_placed)
        offset = 0
        occupied = True
        column_free = self.frames_columns[node_free.frame_]
        while occupied:
            if column_free.is_free(position + offset, node_free):
                occupied = False
            else:
                # offset = (-1) * offset if offset < 0 else (-1)*(offset + 1)
                offset += 1
        return position, position + offset

    def get_next_to_column(self, frame_from, direction):
        # import time
        # time1 = time.time()
        frame_offset = (1 if direction == "right" else -1)
        frame = frame_offset + frame_from
        if frame in self.frames_columns.keys():
            return self.frames_columns[frame]
        else:
            # frames = self.frames_columns.keys()
            # tuples = [tup for tup in frames if isinstance(tup, tuple)]
            # for tup in tuples:
            #     if tup[0 if direction == "right" else 1] == frame:
            #         print("b {0}".format(time.time() - time1))
            #         return self.frames_columns[tup]
            start = frame
            end = frame + frame_offset
            while end not in self.frames_columns.keys():
                end += frame_offset
            return self.frames_columns[((start, end - frame_offset) if direction == "right" else (end - frame_offset, start))]

    def prepare_columns(self, frames):
        from gui.graph_view.column import Column
        empty_frame_count = 0
        for x in range(frames[0], frames[len(frames) - 1] + 1):
            if x in frames:
                if empty_frame_count > 0:
                    if empty_frame_count == 1:
                        column = Column(x - 1, self.scene, self.img_manager, self.relative_margin, self.width, self.height, True)
                        self.frames_columns[x - 1] = column
                        self.columns.append(column)
                    else:
                        column = Column(((x - empty_frame_count), x - 1), self.scene, self.img_manager, self.relative_margin, self.width, self.height, True)
                        self.frames_columns[((x - empty_frame_count), x - 1)] = column
                        self.columns.append(column)
                    self.scene_width += WIDTH / 2 + SPACE_BETWEEN_HOR
                column = Column(x, self.scene, self.img_manager, self.relative_margin, self.width, self.height)
                self.frames_columns[x] = column
                self.columns.append(column)
                self.scene_width += WIDTH + SPACE_BETWEEN_HOR

                empty_frame_count = 0
            else:
                empty_frame_count += 1

    def add_objects(self, added_regions, added_edges):
        print("Preparing objects...")
        frames = set()
        for node in added_regions:
            if node not in self.regions:
                self.regions.add(node)
                if node.frame_ not in frames:
                    frames.add(node.frame_)
        for edge in added_edges:
            self.edges.add(edge)

        frames = list(frames)
        frames.sort()
        first_frame, last_frame = frames[0], frames[len(frames) - 1]

        self.prepare_columns(frames)
        self.edges = comp.sort_edges(self.edges, frames)
        self.compute_positions()
        self.add_sole_nodes()
        print("Visualizing...")
        self.showMaximized()
        self.redraw(first_frame, last_frame)

    def draw_columns(self, first_frame, last_frame, minimum):
        next_x = 0
        if self.dynamically:
            event_loaded = threading.Event()
            thread_load = threading.Thread(group=None, target=self.load, args=(minimum, event_loaded))
        QApplication.processEvents()
        for column in self.columns:
            # uncomment to achieve node-by-node loading, decreases performance, remember to comment
            # every other call of QApplication in this function
            # QApplication.processEvents()
            # import time
            # time1 = time.time()
            # print("Drawing column {0}...".format(column.frame))
            self.load_indicator_wheel()
            column.set_x(next_x)
            next_x = self.increment_x(column, next_x)

            frame_a = frame_b = column.frame
            if isinstance(column.frame, tuple):
                frame_a, frame_b = column.frame[0], column.frame[1]
            if not (frame_a < first_frame or frame_b > last_frame):
                col_index = self.columns.index(column)
                if col_index <= minimum:
                        column.add_crop_to_col()
                        QApplication.processEvents()
                elif self.dynamically:
                    if col_index == minimum + 1:
                        QApplication.processEvents()
                        event_loaded.clear()
                        thread_load.start()
                        event_loaded.wait()
                    if col_index not in self.loaded:
                            event_loaded.clear()
                            event_loaded.wait()
                    if col_index % minimum == 0:
                            QApplication.processEvents()
                column.draw(self.compress_axis, self.show_vertically, self.frames_columns)
            # time2 = time.time()
            # print("The drawing of column took {0}".format(time2 - time1))

    def increment_x(self, column, next_x):
        if column.empty and isinstance(column.frame, tuple) and not self.compress_axis:
            next_x += WIDTH * (column.frame[1] - column.frame[0])
        else:
            next_x += WIDTH / 2 if column.empty else WIDTH
        next_x += SPACE_BETWEEN_HOR
        return next_x

    def load(self, minimum, event_loaded):
        i = minimum + 1
        while i < len(self.columns):
            col = self.columns[i]
            col.prepare_images()
            self.loaded.add(i)
            if i % minimum == 1:
                event_loaded.set()
            i += 1
        event_loaded.set()

    def draw_lines(self, first_frame, last_frame):
        for edge in self.edges:
                if edge[2] == "line" or edge[2] == "chunk":
                    if first_frame <= edge[0].frame_ and edge[1].frame_ <= last_frame:
                        col = self.frames_columns[edge[1].frame_]
                        col.show_edge(edge, self.frames_columns, self.show_vertically)
                elif edge[2] == "partial":
                    if(edge[0] or edge[1]) and (edge[0] in self.regions or edge[1] in self.regions):
                        direction = "left" if (edge[0] is None or not (edge[0] in self.regions)) else "right"
                        node = edge[1] if direction == "left" else edge[0]
                        if first_frame <= node.frame_ <= last_frame:
                            col = self.frames_columns[node.frame_]
                            col.show_edge(edge, self.frames_columns, self.show_vertically, direction, node)

    def toggle_show_vertically(self):
        self.show_vertically = False if self.show_vertically else True
        self.swap_chunk_pictures_widgets()
        self.flash()

    def compress_axis_toggle(self):
        self.compress_axis = False if self.compress_axis else True
        self.flash()

    def flash(self):
        self.view.setEnabled(False)
        self.scene.setForegroundBrush(QtCore.Qt.white)
        self.info_manager.remove_info_all()
        self.redraw(self.columns[0].frame, self.columns[len(self.columns) - 1].frame)
        self.scene.setForegroundBrush(QtCore.Qt.transparent)
        self.view.setEnabled(True)

    def redraw(self, first_frame=None, last_frame=None):
        if not first_frame:
            first_frame = self.columns[0].frame
        if not last_frame:
            last_frame = self.columns[len(self.columns) - 1].frame
        self.view.centerOn(0, 0)
        self.load_indicator_init()

        # to ensure that graphics scene has correct size
        rect = self.add_rect_to_scene()

        self.draw_columns(first_frame, last_frame, MINIMUM)
        self.draw_lines(first_frame, last_frame)
        self.load_indicator_hide()
        self.scene.removeItem(rect)
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.view.centerOn(0, 0)

    def add_rect_to_scene(self):
        width = self.scene_width if self.compress_axis else (WIDTH * self.columns[len(self.columns) - 1].frame +
                                                             (self.columns[len(self.columns) - 1].frame - 1) * SPACE_BETWEEN_VER)
        height = self.compute_height()
        if self.show_vertically:
            width, height = height, width
        rect = QtGui.QGraphicsRectItem(QtCore.QRectF(QtCore.QPointF(0, 0), QtCore.QPointF(width, height)))
        rect.setBrush(QtCore.Qt.transparent)
        rect.setPen(QtCore.Qt.transparent)
        self.scene.addItem(rect)
        return rect

    def compute_height(self):
        height = 0
        for col in self.columns:
            if len(col.objects) > height:
                height = len(col.objects)
        height = GAP + self.height * height + SPACE_BETWEEN_VER * (height - 1)
        return height

    def get_selected(self):
        return self.selected

    def load_indicator_init(self):
        self.text.setText("Loading")

    def load_indicator_hide(self):
        self.text.setText(DEFAULT_TEXT)

    def load_indicator_wheel(self):
        self.text.setText(self.wheel_count * "." + "Loading" + self.wheel_count * ".")
        self.wheel_count += 1
        if self.wheel_count % 3 is 0:
            self.wheel_count = 1

