import threading
from PyQt6 import QtCore, QtGui, QtWidgets

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QSizePolicy

from . import computer as comp
from gui.graph_widget.control_panel import ControlPanel
from gui.graph_widget.graph_line import LineType
from gui.graph_widget_loader import GAP, SPACE_BETWEEN_HOR, \
    MINIMUM, SPACE_BETWEEN_VER, WIDTH, FROM_TOP, HEIGHT
from gui.graph_widget.edge import EdgeGraphical, ChunkGraphical
from gui.graph_widget.info_manager import InfoManager
from gui.graph_widget.node import Node
from gui.graph_widget.node_zoom_manager import NodeZoomManager
from gui.gui_utils import cvimg2qtpixmap
from gui.img_controls.my_scene import MyScene
from gui.img_controls.my_view_zoomable import MyViewZoomable

__author__ = 'Simon Mandlik'


class GraphVisualizer(QtWidgets.QWidget):
    """
    Requires list of regions and list of edge-tuples (node1, node2, type - chunk, line or partial, sureness).
    Those can be passed in constructor or using a method add_objects
    """

    def __init__(self, loader, img_manager, show_tracklet_callback=None, show_vertically=False, compress_axis=True, dynamically=True):
        super(GraphVisualizer, self).__init__()
        self.regions = set()
        self.regions_list = []
        self.edges = set()
        self.edges_list = []
        self.frames_columns = {}
        self.first_frame = None
        self.last_frame = None
        self.columns = []
        self.img_manager = img_manager
        self.relative_margin = loader.relative_margin
        self.width = loader.width
        self.height = loader.height
        self.loader = loader
        self.dynamically = dynamically
        self.compress_axis = compress_axis
        self.show_vertically = show_vertically
        self.show_tracklet_callback = show_tracklet_callback
        self.node_zoom_manager = NodeZoomManager()
        self.info_manager = InfoManager(loader)

        self.setWindowTitle("graph")
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(0, 11, 0, 11)

        self.chunk_detail_scroll_horizontal = QtWidgets.QScrollArea()
        self.chunk_detail_scroll_vertical = QtWidgets.QScrollArea()
        self.chunk_detail_scroll_horizontal.setWidgetResizable(True)
        self.chunk_detail_scroll_vertical.setWidgetResizable(True)

        # hscroll_shint = self.chunk_detail_scroll_horizontal.horizontalScrollBar().sizeHint()
        self.chunk_detail_scroll_horizontal.setFixedHeight(self.height * 2)
        # vscroll_shint = self.chunk_detail_scroll_vertical.verticalScrollBar().sizeHint()
        self.chunk_detail_scroll_vertical.setFixedWidth(self.width * 2)

        self.chunk_detail_widget_horizontal = QtWidgets.QWidget()
        self.chunk_detail_widget_horizontal.setLayout(QtWidgets.QHBoxLayout())
        self.chunk_detail_widget_vertical = QtWidgets.QWidget()
        self.chunk_detail_widget_vertical.setLayout(QtWidgets.QVBoxLayout())
        self.chunk_detail_scroll_horizontal.setContentsMargins(0, 0, 0, 0)
        self.chunk_detail_scroll_vertical.setContentsMargins(0, 0, 0, 0)

        self.chunk_detail_scroll_horizontal.setWidget(self.chunk_detail_widget_horizontal)
        self.chunk_detail_scroll_vertical.setWidget(self.chunk_detail_widget_vertical)

        self.upper_part = QtWidgets.QWidget()
        self.upper_part.setLayout(QtWidgets.QHBoxLayout())
        self.upper_part.setContentsMargins(0, 0, 0, 0)

        self.view = MyViewZoomable(self)
        self.view.setContentsMargins(0, 0, 0, 0)
        self.view.setMouseTracking(True)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scene = MyScene()
        self.scene_width = 0
        self.view.setScene(self.scene)
        self.view.setLayout(QtWidgets.QHBoxLayout())
        self.scene.clicked.connect(self.scene_clicked)
        self.view.setStyleSheet("QGraphicsView { border-style: none;}")

        self.upper_part.layout().addWidget(self.view)
        self.upper_part.layout().addWidget(self.chunk_detail_scroll_vertical)

        self.buttons = ControlPanel(self)

        self.layout().addWidget(self.upper_part)
        self.layout().addWidget(self.chunk_detail_scroll_horizontal)
        self.layout().addWidget(self.buttons)

        self.chunk_detail_scroll_horizontal.hide()
        self.chunk_detail_scroll_vertical.hide()

        self.load_ind = QtWidgets.QLabel()
        self.load_ind.setAlignment(QtCore.Qt.AlignCenter)
        stylesheet = "font: 18px"
        self.load_ind.setStyleSheet(stylesheet)
        self.layout().addWidget(self.load_ind)
        self.load_ind.hide()

        self.selected = []

        self.wheel_count = 1
        self.loaded = set()

        self.selected_in_menu = None
        self.menu_node = QtWidgets.QMenu(self)
        self.menu_edge = QtWidgets.QMenu(self)
        self.show_info_menu_action = QtGui.QAction('Show Info', self)
        self.show_info_menu_action.triggered.connect(self.show_info_action_method)
        self.show_zoom_menu_action = QtGui.QAction('Show Zoom', self)
        self.show_zoom_menu_action.triggered.connect(self.show_zoom_action_method)
        self.hide_zoom_menu_action = QtGui.QAction('Hide Zoom', self)
        self.hide_zoom_menu_action.triggered.connect(self.hide_zoom_action_method)
        self.hide_info_menu_action = QtGui.QAction('Hide Info', self)
        self.hide_info_menu_action.triggered.connect(self.hide_info_action_method)
        self.show_detail_menu_action = QtGui.QAction('Show detail', self)
        self.show_detail_menu_action.triggered.connect(self.show_chunk_pictures_label)
        self.view_results_menu_action = QtGui.QAction('View results', self)
        self.view_results_menu_action.triggered.connect(self.view_results_tracklet)
        self.menu_node.addAction(self.show_info_menu_action)
        self.menu_node.addAction(self.hide_info_menu_action)
        self.menu_node.addAction(self.show_zoom_menu_action)
        self.menu_node.addAction(self.hide_zoom_menu_action)
        self.menu_edge.addAction(self.show_info_menu_action)
        self.menu_edge.addAction(self.hide_info_menu_action)
        self.menu_edge.addAction(self.show_detail_menu_action)
        self.menu_edge.addAction(self.view_results_menu_action)
        self.view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.view.customContextMenuRequested[QPoint].connect(self.menu)

        if len(self.loader.edges) + len(self.loader.regions) > 0:
            self.add_objects(self.loader.regions, self.loader.edges)

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

    def view_results_tracklet(self):
        if self.selected_in_menu and isinstance(self.selected_in_menu, EdgeGraphical):
            if self.show_tracklet_callback is not None:
                self.show_tracklet_callback(self.loader.get_chunk_by_id(self.selected_in_menu.graph_line[5]))

    def menu(self, point):
        it = self.scene.itemAt(self.view.mapToScene(point))
        self.selected_in_menu = it
        if isinstance(it, Node):
            self.menu_node.exec_(self.view.mapToGlobal(point))
        elif isinstance(it, EdgeGraphical):
            self.menu_edge.exec_(self.view.mapToGlobal(point))

    def show_chunk_pictures_label(self):
        chunk = self.selected_in_menu.core_obj
        self.hide_chunk_pictures_widget()
        widget = self.chunk_detail_widget_vertical if self.show_vertically else self.chunk_detail_widget_horizontal

        region_chunk = self.loader.chunks_region_chunks[chunk]
        frames = list(range(chunk.region_from.frame_, chunk.region_to.frame_ + 1))
        freq, none  = QtWidgets.QInputDialog.getInt(self, 'Chunk Detail',
            'Enter frequency:', value=1, min=1)

        for frame in frames[::freq]:
            img = self.img_manager.get_crop(frame, region_chunk[frame - region_chunk.start_frame()], width=self.width,
                                            height=self.height, relative_margin=self.relative_margin)
            pixmap = cvimg2qtpixmap(img)
            label = QtWidgets.QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            widget.layout().addWidget(label)

        scroll = self.chunk_detail_scroll_vertical if self.show_vertically else self.chunk_detail_scroll_horizontal
        scroll.show()

    def hide_chunk_pictures_widget(self):
        widget = self.chunk_detail_widget_vertical if self.show_vertically else self.chunk_detail_widget_horizontal
        for child in widget.findChildren(QWidget):
            widget.layout().removeWidget(child)
            child.hide()
        self.chunk_detail_scroll_horizontal.hide()
        self.chunk_detail_scroll_vertical.hide()

    def swap_chunk_pictures_widgets(self):
        widget_a = self.chunk_detail_widget_horizontal if self.show_vertically else self.chunk_detail_widget_vertical
        widget_b = self.chunk_detail_widget_vertical if self.show_vertically else self.chunk_detail_widget_horizontal
        for child in widget_a.findChildren(QWidget):
            widget_a.layout().removeWidget(child)
            widget_b.layout().addWidget(child)
        if self.show_vertically:
            self.chunk_detail_scroll_horizontal.hide()
        else:
            self.chunk_detail_scroll_vertical.hide()

    def scene_clicked(self, click_pos):
        item = self.scene.itemAt(click_pos)
        if item is None:
            self.selected = []
            self.node_zoom_manager.remove_all()
            self.info_manager.remove_info_all()
            self.hide_chunk_pictures_widget()
        else:
            if isinstance(item, EdgeGraphical):
                self.info_manager.add(item)
                if isinstance(item, ChunkGraphical):
                    # moved to menu
                    # self.show_chunk_pictures_label(item.core_obj)
                    pass
                else:
                    self.hide_chunk_pictures_widget()
            elif isinstance(item, Node):
                item.create_pixmap()
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

    def zoom_to_chunk_event(self):
        ch_id, ok = QtWidgets.QInputDialog.getInt(self, 'Zoom to chunk',
            'Enter chunk\'s id:', value=0, min=0)

        if ok:
            try:
                self.zoom_to_chunk(ch_id)
            except AttributeError as e:
                dialog = QtWidgets.QDialog(self)
                dialog.setLayout(QtWidgets.QVBoxLayout())
                dialog.layout().addWidget(QtWidgets.QLabel(str(e)))
                dialog.show()
            else:
                pass

    def zoom_to_chunk(self, ch_id):
        ch = self.loader.project.chm[ch_id]
        if ch is None:
            raise AttributeError("Chunk with id {0} does not exist!".format(ch_id))
        start_frame = ch.start_frame()
        end_frame = ch.end_frame()
        if start_frame > self.last_frame or end_frame < self.first_frame:
            raise AttributeError("Chunk with id {0} out of range!".format(ch_id))
        if start_frame < self.first_frame:
            start_frame = self.first_frame
        if start_frame == end_frame:
            self.view.centerOn(self.frames_columns[start_frame].x, 0)
        elif start_frame < self.last_frame:
            col = self.frames_columns[start_frame]
            x = col.x
            position = col.get_position_with_chunk_id(ch_id)
            y = GAP + FROM_TOP + position * self.height + SPACE_BETWEEN_VER * position
            if self.show_vertically:
                self.view.centerOn(y, x + QtWidgets.QWidget.normalGeometry(self).width() / 2)
            else:
                self.view.centerOn(x + QtWidgets.QWidget.normalGeometry(self).width() / 2, y)
            x += WIDTH
            y += HEIGHT / 2
            if self.show_vertically:
                x, y = y, x
            g_item = self.scene.itemAt(x, y)
            g_item.setSelected(True)

    def update_lines(self):
        self.loader.update_colours(self.edges)

    def show_node_images(self):
        for it in list(self.scene.items()):
            if isinstance(it, Node):
                it.create_pixmap()

    def compute_positions(self):
        for edge in self.edges:
            if edge.type == LineType.TRACKLET or edge.type == LineType.PARTIAL_TRACKLET:
                self.find_suitable_position_chunk(edge)
            elif edge.type == LineType.LINE:
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
        node_1 = edge.region_from
        node_2 = edge.region_to
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
                    if frame == start or frame == end or position == self.frames_columns[start].get_position_item(
                            node_1):
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
        node_1 = edge.region_from
        node_2 = edge.region_to
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
        if edge.region_from is None or not edge.region_from in self.regions:
            node = edge.region_to
            direction = "left"
        else:
            node = edge.region_from
            direction = "right"

        if node.frame_ not in self.frames_columns:
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
        if frame in self.frames_columns:
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
            while end not in self.frames_columns:
                end += frame_offset
            return self.frames_columns[
                ((start, end - frame_offset) if direction == "right" else (end - frame_offset, start))]

    def prepare_columns(self, frames):
        from gui.graph_widget.column import Column
        empty_frame_count = 0
        for x in range(frames[0], frames[len(frames) - 1] + 1):
            if x in frames:
                if empty_frame_count > 0:
                    if empty_frame_count == 1:
                        column = Column(x - 1, self.scene, self.img_manager, self.relative_margin, self.width,
                                        self.height, True)
                        self.frames_columns[x - 1] = column
                        self.columns.append(column)
                    else:
                        column = Column(((x - empty_frame_count), x - 1), self.scene, self.img_manager,
                                        self.relative_margin, self.width, self.height, True)
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

        self.prepare_columns(frames)
        self.edges = comp.sort_edges(self.edges, frames)
        self.compute_positions()
        self.add_sole_nodes()
        self.first_frame, self.last_frame = frames[0], frames[len(frames) - 1]

    def draw_columns_full(self, first_frame, last_frame, minimum):
        next_x = 0
        if self.dynamically:
            event_loaded = threading.Event()
            thread_load = threading.Thread(group=None, target=self.load, args=(minimum, event_loaded))
            QApplication.processEvents()
        for column in self.columns:
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

    def draw_columns_light(self, first_frame, last_frame):
        """
        Faster than the one above, intended for use with imageless nodes
        """
        next_x = 0
        refresh_step = last_frame - first_frame / 3
        r = first_frame
        for column in self.columns:
            self.load_indicator_wheel()
            column.set_x(next_x)
            next_x = self.increment_x(column, next_x)
            frame_a = frame_b = column.frame
            if isinstance(column.frame, tuple):
                frame_a, frame_b = column.frame[0], column.frame[1]
            if not (frame_a < first_frame or frame_b > last_frame):
                column.delete_scene()
                column.draw(self.compress_axis, self.show_vertically, self.frames_columns)
                if frame_a > r:
                    self.load_indicator_wheel()
                    QApplication.processEvents()
                    r += refresh_step
        QApplication.processEvents()

    def load_columns(self):
        for column in self.columns:
            column.prepare_images()
        print("Columns loaded")

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


    def draw_lines(self, first_frame=None, last_frame=None):
        if first_frame is None:
            first_frame = self.first_frame
        if last_frame is None:
            last_frame = self.last_frame
        for edge in self.edges:
            region_from = edge.region_from
            region_to = edge.region_to
            # if edge.type == LineType.LINE or edge.type == LineType.TRACKLET:
            #     if first_frame <= region_from.frame_ and region_to.frame_ <= last_frame:
            col = self.frames_columns[region_to.frame_]
            col.show_edge(edge, self.frames_columns, self.show_vertically)
            # elif edge.type == LineType.PARTIAL:
            #     if (region_from or region_to) and (region_from in self.regions or region_to in self.regions):
            #         direction = "left" if (region_from is None or not (region_from in self.regions)) else "right"
            #         node = region_to if direction == "left" else region_from
            #         if first_frame <= node.frame_ <= last_frame:
            #             col = self.frames_columns[node.frame_]
            #             col.show_edge(edge, self.frames_columns, self.show_vertically, direction, node)

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

    def redraw(self, first_frame=None, last_frame=None, columns=True):
        if not first_frame:
            first_frame = self.first_frame
        if not last_frame:
            last_frame = self.last_frame
        self.view.centerOn(0, 0)
        self.load_indicator_init()

        # to ensure that graphics scene has correct size
        rect = self.add_rect_to_scene()
        if columns:
            self.draw_columns_light(first_frame, last_frame)
        self.draw_lines(first_frame, last_frame)
        self.load_indicator_hide()
        self.scene.removeItem(rect)
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        # self.view.centerOn(0, 0)

    def add_rect_to_scene(self):
        width = self.scene_width if self.compress_axis else (WIDTH * self.columns[len(self.columns) - 1].frame +
                                                             (self.columns[
                                                                  len(self.columns) - 1].frame - 1) * SPACE_BETWEEN_VER)
        height = self.compute_height()
        if self.show_vertically:
            width, height = height, width
        rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(QtCore.QPointF(0, 0), QtCore.QPointF(width, height)))
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
        self.buttons.hide()
        self.load_ind.show()

    def load_indicator_hide(self):
        self.load_ind.hide()
        self.buttons.show()

    def load_indicator_wheel(self):
        self.load_ind.setText(self.wheel_count * "." + "Loading" + self.wheel_count * ".")
        self.wheel_count += 1
        if self.wheel_count % 3 is 0:
            self.wheel_count = 1
