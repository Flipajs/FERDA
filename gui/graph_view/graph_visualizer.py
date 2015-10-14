from gui.graph_view.edge import Edge_Graphical

__author__ = 'Simon Mandlik'

STEP = 50

from gui.img_controls.my_scene import MyScene
from PyQt4 import QtGui, Qt, QtCore
import computer as comp
from core.project.project import Project
from gui.graph_view.column import Column
from utils.img_manager import ImgManager

class GraphVisualizer(QtGui.QWidget):

    def __init__(self, regions, edges, img_manager, show_vertically=False, node_size=30):
        super(GraphVisualizer, self).__init__()
        self.regions = []
        self.edges = []
        self.frames_columns = {}

        self.img_manager = img_manager

        self.view = QtGui.QGraphicsView(self)
        self.setLayout(QtGui.QVBoxLayout())
        self.view.setMouseTracking(True)
        self.scene = MyScene()
        self.view.setScene(self.scene)
        self.scene.clicked.connect(self.scene_clicked)
        self.layout().addWidget(self.view)

        self.show_vertically = show_vertically
        self.selected = []

        self.add_objects(regions, edges)

    def scene_clicked(self, click_pos):
        item = self.scene.itemAt(click_pos)

        if item is None:
            self.selected = []
        else:
            self.selected.append(item.graph_obj)

        if isinstance(item, Edge_Graphical):
            pass
        # if isinstance(item, Pixmap_Selectable):
            pass
        elif isinstance(item, QtGui.QGraphicsPixmapItem):
            # toggled item...
            return
        else:
            # self.clear_all_button_function()
            self.suggest_node = False
            self.update_view(None, None)
            self.suggest_node = True

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
            for col in self.frames_columns.values():
                if col.contains(node):
                   break
            else:
                position = 0
                while not self.frames_columns[node.frame_].is_free(position):
                    position += 1
                else:
                    self.add_node_to_column(node, node.frame_, position)
                    continue
            break

    def find_suitable_position_chunk(self, edge):
        node_1 = edge[0]
        node_2 = edge[1]
        position = 0
        while True:
            if self.is_line_free(position, node_1.frame_, node_2.frame_):
                start = node_1.frame_
                end = node_2.frame_
                while start < end:
                    try:
                        column = self.frames_columns[start]
                    except:
                        column = self.get_next_to_column(start - 1, "right")
                        start = column.frame[1] + 1
                    column.add_object(edge, position)
                    start += 1
                self.add_node_to_column(node_1, node_1.frame_, position)
                self.add_node_to_column(node_2, node_2.frame_, position)
                break
            else:
                position += 1

    def is_line_free(self, position, start_frame, end_frame):
        free = True
        while start_frame < end_frame:
            try:
                column = self.frames_columns[start_frame]
                start_frame += 1
            except:
                column = self.get_next_to_column(start_frame - 1, "right")
                start_frame = column.frame[1] + 1

            if not column.is_free(position):
                free = False
                break
        return free

    def find_suitable_position_line(self, edge):
        node_1 = edge[0]
        node_2 = edge[1]
        if self.frames_columns[node_1.frame_].contains(node_1):
            position_1, position_2 = self.find_nearest_free_slot(node_1, node_2)
        elif self.frames_columns[node_2.frame_].contains(node_2):
            position_1, position_2 = self.find_nearest_free_slot(node_2, node_1)
        else:
            position_1 = 0
            position_2 = None
            offset_list = [0, 1, -1]
            while position_2 is None:
                for num in offset_list:
                    if self.is_line_free(position_1, node_1.frame_, node_2.frame_) and \
                        self.is_line_free(position_1 + num, node_1.frame_, node_2.frame_):
                        position_2 = position_1 + num

                position_1 += 1

        self.add_node_to_column(node_1, node_1.frame_, position_1)
        self.add_node_to_column(node_2, node_2.frame_, position_2)

    def find_suitable_position_partial(self, edge):
        node, direction = None, None
        if edge[0] is None or not edge[0] in self.regions :
            node = edge[1]
            direction = "left"
        else:
            node = edge[0]
            direction = "right"

        column = self.frames_columns[node.frame_]
        next_column = self.get_next_to_column(node.frame_, direction)

        if column.contains(node):
            next_column.add_object(edge, column.get_position_object(node))
        else:
            position = 0
            occupied = True
            while occupied:
                if not column.is_free(position) and not next_column.is_free(position):
                    occupied = False
                position += 1

            column.add_object(node, position)
            next_column.add_object(edge, position)

    def find_nearest_free_slot(self, node_placed, node_free):
        position = self.frames_columns[node_placed.frame_].get_position_object(node_placed)
        offset = 0
        occupied = True
        column_free = self.frames_columns[node_free.frame_]
        while occupied:
            if column_free.is_free(position + offset):
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

    def draw_columns(self, first_frame, last_frame, scene):
        next_x = 0
        for frame, column in self.frames_columns.items():
            if not (frame < first_frame or frame > last_frame):
                column.set_x(next_x)
                print(frame)
                # column.add_crop_to_col(im_manager, STEP)
                print("pixmap added")
                column.draw(self.show_vertically, scene, self.frames_columns)
                print("column done")
                next_x += STEP/2 if column.empty else STEP
                next_x += STEP/4 #At je mezi cols nejaka pauza

    def prepare_columns(self, frames):
        empty_frame_count = 0
        for x in range(frames[0], frames[len(frames) - 1] + 1):
            # if not self.frames_columns.get(x, None) is None:
            #     continue
            if x in frames:
                if empty_frame_count == 0:
                    column = Column(x)
                    self.frames_columns[x] = column
                else:
                    if empty_frame_count == 1:
                        column = Column(x - 1, True)
                        self.frames_columns[(x - 1)] = column
                    else:
                        column = Column(((x - empty_frame_count), x - 1), True)
                        self.frames_columns[((x - empty_frame_count), x - 1)] = column
                    for y in range(x - empty_frame_count, x):
                        self.frames_columns[x] = column
                    column = Column(x)
                    self.frames_columns[x] = column
                empty_frame_count = 0
            else:
                empty_frame_count += 1

    def add_objects(self, nodes, edges):
        print("Sorting and preparing data")
        for node in nodes:
            if node not in self.regions:
                self.regions.append(node)

        for edge in edges:
            if edge not in self.edges:
                self.edges.append(edge)

        frames = []
        for node in self.regions:
            if not node.frame_ in frames:
                frames.append(node.frame_)
        frames.sort()

        first_frame, last_frame = frames[0], frames[len(frames) -  1]

        self.prepare_columns(frames)
        self.edges = comp.sort_edges(self.edges, self.regions, frames)
        print("Computing positions for edges")
        self.compute_positions()
        print("Adding remaining nodes")
        self.add_sole_nodes()
        print("Drawing")
        self.draw_columns(first_frame, last_frame, self.scene)

        #prepocitat, pouze od urcite polohy prekreslit, zeptat se Filipa, neprepocitavat po kazdem pridanem objektu

    def add_node_to_column(self, node, column_frame, position):
        self.frames_columns[column_frame].add_object(node, position)

    def toggle_show_vertically(self):
        self.show_vertically = False if self.show_vertically else True
        self.draw_columns(0, len(self.frames_columns))

    def get_selected(self):
        return self.selected

if __name__ == '__main__':
    p = Project()
    p.load('/home/ferda/PROJECTS/eight_22/eight22.fproj')

    im_manager = ImgManager(p)

    import cv2
    solver = p.saved_progress['solver']
    nodes = solver.g.nodes()
    edges = solver.g.edges()
    edges_4_tuple = []
    import random
    for edge in edges:
        type = random.choice(["chunk", "line"])
        if type == "chunk":
            sureness = 1
        else:
            sureness = random.randint(0, 101) / float(100)

        new_tuple = edge + (type, sureness)
        edges_4_tuple.append(new_tuple)

    import sys
    app = QtGui.QApplication(sys.argv)
    g = GraphVisualizer(nodes, edges_4_tuple, im_manager)
    g.show()
    app.exec_()
    cv2.waitKey(0)

