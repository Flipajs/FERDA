from core.region.region import Region

__author__ = 'Simon Mandlik'

from gui.img_controls.my_scene import MyScene
from PyQt4 import QtGui, QtCore, Qt
import numpy as np
from column import Column

STEP = 20

class GraphVisualizer():

    def __init__(self, regions, edges, img_manager):
        self.regions = regions
        self.nodes = []
        self.edges = edges
        self.img_manager = img_manager
        self.columns = []
        self.frames_columns = {}
        self.prepare_data()

        self.widget = QtGui.QWidget()
        self.view = QtGui.QGraphicsView(self.widget)
        self.widget.setLayout(QtGui.QVBoxLayout())
        self.scene = MyScene()
        self.view.setScene(self.scene)
        self.view.setMouseTracking(True)
        self.scene.clicked.connect(self.scene_clicked)

    def prepare_data(self):
        frames = []
        for node in self.regions:
            frames.append(node._frame)
        frames.sort()

        self.prepare_columns(frames)
        self.edges = sort_edges_by_length_ascending(self.edges, frames)

    def compute_positions(self):
        for edge in self.edges:

            position = 0
            while True:
                free = True
                for x in range(edge[0]._frame, edge[1]._frame + 1):
                    if not self.frames_columns[x].is_free(position):
                        free = False
                    if free:
                        for x in range(edge[0]._frame + 1, edge[1]._frame):
                            self.frames_columns[x].add_object(edge, position)
                            self.frames_columns[edge[0]._frame].add_object(edge[0], position)
                            self.frames_columns[edge[1]._frame].add_object(edge[1], position)
                        break
                    else:
                        position += 1

    def add_images(self):
        #lze?

    def draw_columns(self):
        for column in self.columns:
            column.draw()

    def draw_edges(self):
        for edge in self.edges:
            if edge[0] is None or edge[1] is None:
                #nakresli partial
        #ziskat sureness, rozlisit je

    def prepare_columns(self, frames):
        empty_frame_count = 0
        for x in range(frames[0], frames[len(frames)] + 1):
            if x in frames:
                if empty_frame_count == 0:
                    column = Column(x)
                    self.columns.append(column)
                    self.frames_columns[x] = column
                else:
                    column = Column(((x - empty_frame_count), x - 1), True)
                    self.columns.append(column)
                    for y in range(x - empty_frame_count, x):
                        self.frames_columns[x] = column
                    column = Column(x)
                    self.columns.append(column)
                    self.frames_columns[x] = column
                empty_frame_count = 0
            else:
                empty_frame_count += 1

    def add_objects(self, *args):
        for object in args:
            if isinstance(object, Region):
                self.regions.apend(object)
            elif isinstance(object, tuple):
                if isinstance(object[0], Region) and isinstance(object[1], Region):
                    self.edges.apend(object)

        #prekreslit, prepocitat, pouze od urcite polohy, zeptat se Filipa, neprepocitavat po kazdem pridanem objektu




    # def prepare_positions(self):

        # for edge in self.edges:
        #     n1, n2
        #     #ziskat z edge nody
        #
        #     add_to_graph()

    # def add_to_graph(self, node):
    #     column = self.frames_columns(node._frame)
    #     if column is None:
    #         column = Column(node._frame)
    #         self.frames_columns[node._frame] = column
    #     else:
    #         column.add_node(node)

    # def sort_columns_by_frame(self, columns):
    #     frames = self.frames_columns.keys()
    #     frames.sort()
    #     self.columns = []
    #     for frame in frames:
    #         self.columns.append(self.frames_columns.get(frame))

    # def insert_blank_spaces(self):
    #     blank = []
    #
    #     for a, b,  in zip(self.columns, self.columns[1:]):
    #         if (b.frame - a.frame) > 1:
    #             blank.append((self.columns.index(b), a.frame))
    #
    #     for index, frame in blank:
    #         self.columns.insert(Column(frame + 1, True), index)
    #         self.

def sort_edges_by_length_ascending(edges, used_frames_sorted):
    result = []
    for edge in edges:
        length = used_frames_sorted.index(edge[0]._frame) - used_frames_sorted.index(edge[1]._frame)
        result.insert(edge, length)
    return result

if __name__ == '__main__':

