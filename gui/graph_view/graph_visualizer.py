from gui.graph_view.edge import Edge_Graphical

__author__ = 'Simon Mandlik'

STEP = 20

from gui.img_controls.my_scene import MyScene
from PyQt4 import QtGui
import computer as comp
from core.project.project import Project
from gui.graph_view.column import Column
from utils.img_manager import ImgManager

class GraphVisualizer():

    def __init__(self, regions, edges, img_manager, show_vertically=False, node_size=30):
        self.regions = []
        self.edges = []
        self.add_objects(regions, edges)

        self.img_manager = img_manager
        self.frames_columns = {}

        self.widget = QtGui.QWidget()
        self.view = QtGui.QGraphicsView(self.widget)
        self.widget.setLayout(QtGui.QVBoxLayout())
        self.scene = MyScene()
        self.view.setScene(self.scene)
        self.view.setMouseTracking(True)
        self.scene.clicked.connect(self.scene_clicked)

        self.show_vertically = show_vertically
        self.selected = []

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

    def prepare_data(self):
        frames = []
        for node in self.regions:
            frames.append(node._frame)
        frames.sort()

        self.prepare_columns(frames)
        self.edges = comp.sort_edges()

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
               while not self.frames_columns[node._frame].is_free(position):
                   position += 1
               else:
                   self.add_node_to_column(node, node._frame)
           break

    def find_suitable_position_chunk(self, edge):
        node_1 = edge[0]
        node_2 = edge[1]
        position = 0
        while True:
            if self.is_line_free(position, node_1._frame, node_2._frame):
                for x in range(node_1._frame + 1, node_2._frame):
                    try:
                        column = self.frames_columns[x]
                    except:
                        column = self.get_next_column(x, "right")
                        x = column.frame + 1
                    column.add_object(edge, position)
                self.add_node_to_column(node_1, node_1._frame, position)
                self.add_node_to_column(node_2, node_2._frame, position)
                break
            else:
                position += 1

    def is_line_free(self, position, start_frame, end_frame):
        free = True
        for x in range(start_frame, end_frame + 1):
            try:
                column = self.frames_columns[x]
            except:
                column = self.get_next_column(x, "right")
                x = column.frame + 1

            if not column.is_free(position):
                free = False
        return  free

    def find_suitable_position_line(self, edge):
        node_1 = edge[0]
        node_2 = edge[1]
        if self.frames_columns[node_1._frame].contains(node_1):
            position_1, position_2 = self.find_nearest_free_slot(node_1, node_2)
        elif self.frames_columns[node_2._frame].contains(node_2):
            position_1, position_2 = self.find_nearest_free_slot(node_2, node_1)
        else:
            position_1 = 0
            offset_list = [0, 1, -1]
            while True:
                for num in offset_list:
                    if self.is_line_free(position_1, node_1._frame, node_2._frame) and \
                        self.is_line_free(position_1 + num, node_1._frame, node_2._frame):
                        position_2 = position_1 + num
                        break
                position_1 += 1

        self.add_node_to_column(node_1, node_1._frame, position_1)
        self.add_node_to_column(node_2, node_2._frame, position_2)

    def find_suitable_position_partial(self, edge):
        node, direction = None
        if edge[0] is None or not edge[0] in self.regions :
            node = edge[1]
            direction = "left"
        else:
            node = edge[0]
            direction = "right"

        column = self.frames_columns[node._frame]
        next_column = self.get_next_column(node._frame, direction)

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
        position = self.frames_columns[node_placed._frame].get_position_object(node_placed)
        offset = 0
        occupied = True
        column_free = self.frames_columns[node_free._frame]
        while occupied:
            if column_free.is_free(position + offset):
                occupie = False
            offset += (-1) * (offset + 1)
        return position, position + offset

    def get_next_column(self, frame, direction):
        try:
            return self.frames_columns[frame + (1 if direction == "right" else -1)]
        except:
            frames = self.frames_columns.keys()
            frames = [tuple for tuple in frames if isinstance(tuple, tuple)]
        for frame in frames:
            if [0 if direction == "right" else 1] == frame + (1 if direction == "right" else -1):
                return self.frames_columns[frame]
                break

    def draw_columns(self, first_frame, last_frame, scene):
        next_x = 0
        for frame, column in self.frames_columns.items():
            if not (frame < first_frame or frame > last_frame):
                column.set_x(next_x)
                column.add_crop_to_col(im_manager)
                column.draw(self.show_vertically, scene, self.frames_columns)
                next_x += STEP/2 if column.empty else STEP

    def prepare_columns(self, frames):

        empty_frame_count = 0
        for x in range(frames[0], frames[len(frames)] + 1):
            if not self.frames_columns[x] is None:
                continue
            elif x in frames:
                if empty_frame_count == 0:
                    column = Column(x)
                    self.frames_columns[x] = column
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
        first_frame, last_frame = 0
        for node in nodes:
            if node not in self.regions:
                self.regions.append(node)
                if node._frame < first_frame:
                    first_frame = node._frame
                elif node._frame > last_frame:
                    last_frame = node._frame
        for edge in edges:
            if edge not in self.edges:
                self.edges.append(edge)
                for x in range(2):
                    node = edge[x]
                    if node._frame < first_frame:
                        first_frame = node._frame
                    elif node._frame > last_frame:
                        last_frame = node._frame

        self.prepare_data()
        self.compute_positions()
        self.add_sole_nodes()
        self.draw_columns(first_frame, last_frame)

        #prepocitat, pouze od urcite polohy prekreslit, zeptat se Filipa, neprepocitavat po kazdem pridanem objektu

    def add_node_to_column(self, node, column_frame, position):
        self.frames_columns[column_frame].add_object(node, position)

    def toggle_show_vertically(self):
        self.show_vertically = False if self.show_vertically else True
        self.draw_columns(0, len(self.frames_columns))

    def get_selected(self):
        return self.selected

if __name__ == '__main__':
    execfile('/home/ferda/FERDA/scripts/fix_project.py')

    from core.project.project import Project


    p = Project()
    p.load('/home/ferda/PROJECTS/eight_22/eight22.fproj')

    p.video_paths = ['/home/ferda/PROJECTS/eight.m4v']
    p.working_directory = '/home/ferda/FERDA'

    for it in p.log.data_:
        print it.action_name, it.data

    p.save()
    p = Project()
    p.load('/home/ferda/PROJECTS/eight_22/eight22.fproj')

    im_manager = ImgManager(p)

    import cv2
    solver = p.saved_progress['solver']
    nodes = solver.g.nodes()
    edges = solver.g.edges()

    import random
    for edge in edges:
        sureness = random.randint(0, 101)
        # type = random.choice(["chunk", "partial", "line"])
        edge.append("", sureness)


    g = GraphVisualizer(nodes, edges, im_manager)
    cv2.waitKey(0)

