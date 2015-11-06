import threading
from PyQt4.QtGui import QApplication
from gui.graph_view.node import Node
from gui.img_controls.my_scene import MyScene
from PyQt4 import QtGui, QtCore
import computer as comp
from core.project.project import Project
from gui.img_controls.my_view_zoomable import MyViewZoomable
from utils.img_manager import ImgManager
from gui.graph_view.edge import EdgeGraphical

# the size of a node
STEP = 40
# distance of the whole visualization from the top of the widget
FROM_TOP = 20
# space between each of the columns
SPACE_BETWEEN_HOR = 30
# space between nodes in column
SPACE_BETWEEN_VER = 10
# gap between frame_numbers and first node in columns
GAP = 50
# number of columns to be displayed before dynamic loading, 0 means dynamic loading for all
MINIMUM = 20
# number of columns processed in one chunk sent to dummy thread, for debbuging purpose
COLUMNS_TO_LOAD = 10
# default text to display
DEFAULT_TEXT = "V - toggle vertical display; C - compress axis; I, O - zoom in or out; Q, W - shrink, stretch"

__author__ = 'Simon Mandlik'


class GraphVisualizer(QtGui.QWidget):
    """
    Requires list of regions and list of edge-tuples (node1, node2, type - chunk, line or partial, sureness).
    Those can be passed in constructor or using a method add_objects
    """

    def __init__(self, regions, edges, img_manager, show_vertically=False, compress_axis=True, dynamically=True):
        super(GraphVisualizer, self).__init__()
        self.regions = set()
        self.regions_list = []
        self.edges = set()
        self.edges_list = []
        self.frames_columns = {}
        self.columns = []
        self.img_manager = img_manager

        self.view = MyViewZoomable(self)
        self.setLayout(QtGui.QVBoxLayout())
        self.view.setMouseTracking(True)
        self.scene = MyScene()
        self.scene_width = 0
        self.view.setScene(self.scene)
        self.scene.clicked.connect(self.scene_clicked)
        self.layout().addWidget(self.view)

        self.text = QtGui.QLabel(DEFAULT_TEXT)
        self.text.setAlignment(QtCore.Qt.AlignCenter)
        stylesheet = "font: 25px"
        self.text.setStyleSheet(stylesheet)
        self.layout().addWidget(self.text)


        self.menu_node = QtGui.QMenu(self)
        self.menu_edge = QtGui.QMenu(self)
        # TODO add your actions
        self.test_action_node = QtGui.QAction('test_action_node', self)
        self.test_action_edge = QtGui.QAction('test_action_edge', self)
        self.menu_node.addAction(self.test_action_node)
        self.menu_edge.addAction(self.test_action_edge)
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

        self.selected = []
        self.selected_edge = None
        self.toggled = []
        self.wheel_count = 1
        self.loaded = set()

        if len(edges) + len(regions) > 0:
            self.add_objects(regions, edges)

    def menu(self, point):
        it = self.scene.itemAt(self.view.mapToScene(point))
        if isinstance(it, Node):
            self.menu_node.exec_(self.view.mapToGlobal(point))
        elif isinstance(it, EdgeGraphical):
            self.menu_edge.exec_(self.view.mapToGlobal(point))

    def scene_clicked(self, click_pos):
        item = self.scene.itemAt(click_pos)
        if item is None:
            self.selected = []
            while self.toggled:
                self.toggled.pop().toggle()
        else:
            self.selected.append(item)

        if isinstance(item, EdgeGraphical):
            self.selected_edge = item
        elif isinstance(item, Node):
            item.toggle()
            self.toggled.append(item)

    def stretch(self):
        global SPACE_BETWEEN_HOR
        SPACE_BETWEEN_HOR *= 1.2
        self.redraw(self.columns[0].frame, self.columns[len(self.columns) - 1].frame)

    def shrink(self):
        global SPACE_BETWEEN_HOR
        SPACE_BETWEEN_HOR *= 0.8
        self.redraw(self.columns[0].frame, self.columns[len(self.columns) - 1].frame)

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
        position = 0
        while True:
            if self.is_line_free(position, node_1.frame_, node_2.frame_):
                start = node_1.frame_
                end = node_2.frame_
                while start <= end:
                    if start in self.frames_columns.keys():
                        column = self.frames_columns[start]
                    else:
                        column = self.get_next_to_column(start - 1, "right")
                        start = column.frame[1]
                    if start is node_1.frame_ or start is end or \
                                    position is self.frames_columns[node_1.frame_].get_position_item(node_1):
                        column.add_object(edge, position)
                    start += 1
                break
            else:
                position += 1

    def find_suitable_position_semi_placed_chunk(self, edge, col1, col2, node_1, node_2):
        if col1.contains(node_1) and col2.contains(node_2):
            return
        if col1.contains(node_1) and not col2.contains(node_2):
            position1, position2 = self.find_nearest_free_slot(node_1, node_2)
            col2.add_object(edge, position2)
        if not col1.contains(node_1) and col2.contains(node_2):
            position1, position2 = self.find_nearest_free_slot(node_2, node_1)
            col1.add_object(edge, position2)

    def is_line_free(self, position, start_frame, end_frame):
        while start_frame <= end_frame:
            if start_frame in self.frames_columns.keys():
                column = self.frames_columns[start_frame]
                start_frame += 1
            else:
                column = self.get_next_to_column(start_frame - 1, "right")
                start_frame = column.frame[1] + 1
            if not column.is_free(position):
                return False
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
                offset = (-1) * offset if offset < 0 else (-1)*(offset + 1)
        return position, position + offset

    def get_next_to_column(self, frame_from, direction):
        frame_offset = (1 if direction == "right" else -1) + frame_from
        if frame_offset in self.frames_columns.keys():
            return self.frames_columns[frame_offset]
        else:
            frames = self.frames_columns.keys()
            tuples = [tup for tup in frames if isinstance(tup, tuple)]
            for tup in tuples:
                if tup[0 if direction == "right" else 1] == frame_offset:
                    return self.frames_columns[tup]

    def prepare_columns(self, frames):
        from gui.graph_view.column import Column
        empty_frame_count = 0
        for x in range(frames[0], frames[len(frames) - 1] + 1):
            if x in frames:
                if empty_frame_count > 0:
                    if empty_frame_count == 1:
                        column = Column(x - 1, self.scene, self.img_manager, True)
                        self.frames_columns[x - 1] = column
                        self.columns.append(column)
                    else:
                        column = Column(((x - empty_frame_count), x - 1), self.scene, self.img_manager, True)
                        self.frames_columns[((x - empty_frame_count), x - 1)] = column
                        self.columns.append(column)
                    self.scene_width += STEP / 2 + SPACE_BETWEEN_HOR
                column = Column(x, self.scene, self.img_manager)
                self.frames_columns[x] = column
                self.columns.append(column)
                self.scene_width += STEP + SPACE_BETWEEN_HOR

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
        first_frame, last_frame = frames[0], frames[len(frames) - 1]

        self.prepare_columns(frames)
        self.edges = comp.sort_edges(self.edges, frames)
        self.compute_positions()
        self.add_sole_nodes()
        self.redraw(first_frame, last_frame)

    def draw_columns(self, first_frame, last_frame, minimum):
        event_loaded = threading.Event()
        thread_load = threading.Thread(group=None, target=self.load, args=(minimum, event_loaded))
        next_x = 0
        for column in self.columns:
            QApplication.processEvents()
            self.load_indicator_wheel()
            column.set_x(next_x)
            next_x = self.increment_x(column, next_x)

            frame_a = frame_b = column.frame
            if isinstance(column.frame, tuple):
                frame_a, frame_b = column.frame[0], column.frame[1]
            if not (frame_a < first_frame or frame_b > last_frame):
                if self.columns.index(column) == minimum:
                    event_loaded.clear()
                    thread_load.start()
                    event_loaded.wait()
                elif self.columns.index(column) < minimum:
                    column.add_crop_to_col()
                else:
                    if column not in self.loaded:
                        event_loaded.clear()
                        event_loaded.wait()
                column.draw(self.compress_axis, self.show_vertically, self.frames_columns)

    def increment_x(self, column, next_x):
        if column.empty and isinstance(column.frame, tuple) and not self.compress_axis:
            next_x += STEP * (column.frame[1] - column.frame[0])
        else:
            next_x += STEP / 2 if column.empty else STEP
        next_x += SPACE_BETWEEN_HOR
        return next_x

    def load(self, minimum, event_loaded):
        columns = list(self.columns[minimum::])
        while len(columns) > 0:
            columns_stripped = columns[:COLUMNS_TO_LOAD:]
            columns = columns[COLUMNS_TO_LOAD::]
            for col in columns_stripped:
                col.prepare_images()
                self.loaded.add(col)
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
        self.flash()

    def compress_axis_toggle(self):
        self.compress_axis = False if self.compress_axis else True
        self.flash()

    def flash(self):
        self.view.setEnabled(False)
        self.scene.setForegroundBrush(QtCore.Qt.white)
        self.redraw(self.columns[0].frame, self.columns[len(self.columns) - 1].frame)
        self.scene.setForegroundBrush(QtCore.Qt.transparent)
        self.view.setEnabled(True)

    def redraw(self, first_frame, last_frame):
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
        width = self.scene_width if self.compress_axis else (STEP * self.columns[len(self.columns) - 1].frame +
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
        height = GAP + STEP * height + SPACE_BETWEEN_VER * (height - 1)
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


if __name__ == '__main__':
    p = Project()
    p.load('/home/ferda/PROJECTS/eight_22/eight22.fproj')
    im_manager = ImgManager(p)

    solver = p.saved_progress['solver']
    n = solver.g.nodes()
    e = solver.g.edges(data=True)

    import sys
    app = QtGui.QApplication(sys.argv)
    g = GraphVisualizer([], [], im_manager, show_vertically=False, compress_axis=True, dynamically=True)
    g.showMaximized()
    g.show()
    g.add_objects(n, e)
    app.exec_()