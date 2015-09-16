import thread
import threading
import time
from gui.graph_view.node import Node
from gui.img_controls.my_scene import MyScene
from PyQt4 import QtGui, QtCore
import computer as comp
from core.project.project import Project
from gui.img_controls.my_view_zoomable import MyViewZoomable
from utils.img_manager import ImgManager
from gui.graph_view.edge import Edge_Graphical

__author__ = 'Simon Mandlik'

# the size of a node
STEP = 50
# distance of the whole visualization from the top of the widget
FROM_TOP = 0
# space between each of the columns
SPACE_BETWEEN_HOR = 20
# space between nodes in column
SPACE_BETWEEN_VER = 5
# gap between frame_numbers and first node in columns
GAP = 50
# number of columns to be displayed before dynamic loading, -1 means no dynamicall loading
MINIMUM = 10

class GraphVisualizer(QtGui.QWidget):

    def __init__(self, regions, edges, img_manager, show_vertically=False, compress_axis = True, dynamically=True):
        super(GraphVisualizer, self).__init__()
        self.regions = set()
        self.edges = set()
        self.frames_columns = {}
        self.columns = []

        self.img_manager = img_manager

        self.view = MyViewZoomable(self)
        self.setLayout(QtGui.QVBoxLayout())
        self.view.setMouseTracking(True)
        self.scene = MyScene()
        self.view.setScene(self.scene)
        self.scene.clicked.connect(self.scene_clicked)
        self.layout().addWidget(self.view)

        self.menu_node = QtGui.QMenu(self)
        self.menu_edge = QtGui.QMenu(self)
        # add your actions
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

        self.selected = []
        self.selected_edge = None
        self.toggled = []

        self.wheel = None
        self.wheel_count = 1

        self.add_objects(regions, edges)

    def menu(self, point):
        it = self.scene.itemAt(self.view.mapToScene(point))
        if isinstance(it, Node):
            self.menu_node.exec_(self.view.mapToGlobal(point))
        elif isinstance(it, Edge_Graphical):
            self.menu_edge.exec_(self.view.mapToGlobal(point))

    def scene_clicked(self, click_pos):
        item = self.scene.itemAt(click_pos)
        if item is None:
            self.selected = []
            while self.toggled:
                self.toggled.pop().toggle()

        if isinstance(item, Edge_Graphical):
            self.selected_edge = item
        elif isinstance(item, Node):
            item.toggle()
            self.toggled.append(item)

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

    def find_suitable_position_chunk(self, edge):
        node_1 = edge[0]
        node_2 = edge[1]
        position = 0
        while True:
            if self.is_line_free(position, node_1.frame_, node_2.frame_):
                start = node_1.frame_
                end = node_2.frame_
                while start <= end:
                    try:
                        column = self.frames_columns[start]
                    except:
                        column = self.get_next_to_column(start - 1, "right")
                        start = column.frame[1]
                    column.add_object(edge, position)
                    start += 1
                break
            else:
                position += 1

    def is_line_free(self, position, start_frame, end_frame):
        while start_frame <= end_frame:
            try:
                column = self.frames_columns[start_frame]
                start_frame += 1
            except:
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
                        # if self.is_line_free(position_1, node_1.frame_, node_2.frame_) and \
                        #         self.is_line_free(position_1 + num, node_1.frame_, node_2.frame_):
                        #     position_2 = position_1 + num
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

        column = self.frames_columns[node.frame_]
        next_column = self.get_next_to_column(node.frame_, direction)

        if not column.contains(node):
            position = 0
            occupied = True
            while occupied:
                if not column.is_free(position) and not next_column.is_free(position):
                    occupied = False
                position += 1

            column.add_object(node, position)

    def find_nearest_free_slot(self, node_placed, node_free):
        position = self.frames_columns[node_placed.frame_].get_position_object(node_placed)
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
        try:
            return self.frames_columns[frame_from + (1 if direction == "right" else -1)]
        except:
            frames = self.frames_columns.keys()
            tuples = [tup for tup in frames if isinstance(tup, tuple)]
            for tup in tuples:
                if tup[0 if direction == "right" else 1] == frame_from + (1 if direction == "right" else -1):
                    return self.frames_columns[tup]

    def draw_columns(self, first_frame, last_frame, minimum):
        event = threading.Event()
        thread = threading.Thread(group=None, target=self.load, args=(minimum, event))
        next_x = 0
        self.scene.setForegroundBrush(QtCore.Qt.lightGray)
        for column in self.columns:
            app.processEvents()
            column.set_x(next_x)

            x = next_x
            y = 0
            if self.show_vertically:
                x, y = y, x
            self.view.centerOn(QtCore.QPointF(x, y))

            if column.empty and isinstance(column.frame, tuple) and not self.compress_axis:
                next_x += STEP * (column.frame[1] - column.frame[0])
            else:
                next_x += STEP / 2 if column.empty else STEP
            next_x += SPACE_BETWEEN_HOR

            frame_a = frame_b = column.frame
            if isinstance(column.frame, tuple):
                frame_a, frame_b = column.frame[0], column.frame[1]

            if not (frame_a < first_frame or frame_b > last_frame):
                    if self.columns.index(column) == minimum:
                        event.clear()
                        thread.start()
                        self.scene.setForegroundBrush(QtCore.Qt.transparent)
                        event.wait()
                    elif self.columns.index(column) < minimum:
                        column.add_crop_to_col()
                        column.draw(self.compress_axis, self.show_vertically)
                    else:
                        if thread.is_alive():
                            event.clear()
                        column.draw(self.compress_axis, self.show_vertically)
                        event.wait()
                    self.wheel_turn()

    def load(self, minimum, event):
        for col in self.columns[minimum - 1::]:
            col.prepare_imgs(event)

    def draw_lines(self, first_frame, last_frame):
        for edge in self.edges:
                if edge[2] == "line" or edge[2] == "chunk":
                    if first_frame <= edge[0].frame_ and edge[1].frame_ <= last_frame:
                        col = self.frames_columns[edge[1].frame_]
                        col.show_edge(edge, self.frames_columns, self.show_vertically)
                elif edge[2] == "partial":
                    direction = "left" if (edge[0] is None or not (edge[0] in self.regions)) else "right"
                    node = edge[1] if direction == "left" else edge[0]
                    if first_frame <= node.frame_ <= last_frame:
                        col = self.frames_columns[node.frame_]
                        col.show_edge(edge, self.frames_columns, self.show_vertically, direction, node)

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

                column = Column(x, self.scene, self.img_manager)
                self.frames_columns[x] = column
                self.columns.append(column)

                empty_frame_count = 0
            else:
                empty_frame_count += 1

    def add_objects(self, added_nodes, added_edges):
        print("Sorting and preparing data")
        frames = set()
        for node in added_nodes:
            if node not in self.regions:
                self.regions.add(node)
                if node.frame_ not in frames:
                    frames.add(node.frame_)
        for edge in added_edges:
            if 'chunk_ref' in edge[2].keys():
                type_edge = "chunk"
            elif edge[0] is None or edge[1] is None or not edge[0] in self.regions or not edge[1] in self.regions:
                type_edge = "partial"
            else:
                type_edge = "line"

            # TODO sureness - dodelat, zatim se generuje nahodne
            import random
            sureness = random.randint(0, 100) / float(100)
            new_tuple = (edge[0], edge[1]) + (type_edge, sureness)
            self.edges.add(new_tuple)

        frames = list(frames)
        frames.sort()
        first_frame, last_frame = frames[0], frames[len(frames) - 1]

        self.prepare_columns(frames)
        self.edges = comp.sort_edges(self.edges, self.regions, frames)
        print("Computing positions for edges")
        self.compute_positions()
        print("Adding remaining nodes")
        self.add_sole_nodes()
        print("Drawing")
        self.redraw(first_frame, last_frame)
        print("Viewing")

    def add_node_to_column(self, node, column_frame, position):
        self.frames_columns[column_frame].add_object(node, position)

    def toggle_show_vertically(self):
        self.show_vertically = False if self.show_vertically else True
        self.redraw(self.columns[0].frame, self.columns[len(self.columns) - 1].frame)

    def compress_axis_toggle(self):
        self.compress_axis = False if self.compress_axis else True
        self.redraw(self.columns[0].frame, self.columns[len(self.columns) - 1].frame)

    def redraw(self, first_frame, last_frame):
        self.view.centerOn(QtCore.QPointF(0,0))
        self.view.setEnabled(False)
        # self.scene.setForegroundBrush(QtCore.Qt.lightGray)
        self.wheel_init()
        self.draw_columns(first_frame, last_frame, MINIMUM)
        self.draw_lines(first_frame, last_frame)
        self.hide_wheel()
        # self.scene.setForegroundBrush(QtCore.Qt.transparent)
        self.view.setEnabled(True)
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.view.centerOn(0, 0)

    def get_selected(self):
        return self.selected

    def wheel_init(self):
        # TODO make it prettier
        if self.wheel is None:
            self.wheel = QtGui.QLabel()
            self.wheel.setAlignment(QtCore.Qt.AlignCenter)
            stylesheet = "font: 25px"
            self.wheel.setStyleSheet(stylesheet)
        self.layout().addWidget(self.wheel)
        self.wheel.show()

    def wheel_turn(self):
        self.wheel_count += 1 if self.wheel_count is not 3 else -2
        text = self.wheel_count * ". " + "Loading " + self.wheel_count * " ."
        self.wheel.setText(text)

    def hide_wheel(self):
        self.layout().removeWidget(self.wheel)
        self.wheel.hide()

if __name__ == '__main__':
    p = Project()
    p.load('/home/ferda/PROJECTS/eight_22/eight22.fproj')

    im_manager = ImgManager(p)

    import cv2
    solver = p.saved_progress['solver']
    n = solver.g.nodes()
    e = solver.g.edges(data=True)

    # n = n[:100:]
    # e = e[:100:]

    import sys
    app = QtGui.QApplication(sys.argv)
    g = GraphVisualizer(n, e, im_manager, show_vertically=False, compress_axis=True, dynamically=True)
    g.show()
    app.exec_()


