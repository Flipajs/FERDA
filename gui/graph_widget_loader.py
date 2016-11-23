import random
from PyQt4 import QtGui

import sys

import cv2

from core.graph.region_chunk import RegionChunk
from core.project.project import Project
from core.settings import Settings as S_
from gui.graph_widget.graph_line import LineType, GraphLine
from utils.img_manager import ImgManager

__author__ = 'Simon Mandlik'

# the width of a node
WIDTH = 35
# the width of a node, should be same as width for the best result
HEIGHT = WIDTH
# relative margin of a node
RELATIVE_MARGIN = 0.9
# distance of the whole visualization from the top of the widget
FROM_TOP = WIDTH
# space between each of the columns
SPACE_BETWEEN_HOR = WIDTH
# space between nodes in column
SPACE_BETWEEN_VER = HEIGHT
# gap between frame_numbers and first node in columns
GAP = WIDTH + 10
# number of columns to be displayed before dynamic loading, 0 means dynamic loading for all
MINIMUM = 5
# Opacity of the colors
OPACITY = 255


class GraphWidgetLoader:
    def __init__(self, project=None, width=WIDTH, height=HEIGHT, relative_margin=RELATIVE_MARGIN):
        self.project = project

        self.graph_manager = None
        self.graph = None
        self.region_manager = None
        self.update_structures()

        self.vertices = set()
        self.edges = set()
        self.regions = set()

        # self.chunks_region_chunks = {}
        self.regions_vertices = {}

        self.g = None

        self.relative_margin = relative_margin
        self.height = height
        self.width = width

    def set_project(self, project):
        self.project = project

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def set_relative_margin(self, margin):
        self.relative_margin = margin

    def update_structures(self):
        if self.project is not None:
            self.graph_manager = self.project.gm
            self.region_manager = self.project.rm
            self.graph = self.graph_manager.g
        else:
            print "No project set!"

    def prepare_vertices(self, frames):
        if frames is None:
            self.vertices = set(self.graph_manager.get_all_relevant_vertices())
        else:
            self.vertices = []

            for f in frames:
                self.vertices.extend(self.graph_manager.get_vertices_in_t(f))

            self.vertices = set(self.vertices)

    def prepare_nodes(self):
        for vertex in self.vertices:
            region = self.graph_manager.region(vertex)
            self.regions_vertices[region] = vertex
            self.regions.add(region)

    def prepare_edges(self):
        for vertex in self.vertices:
            v = self.graph.vertex(vertex)
            self.prepare_tuples(v.in_edges())
            self.prepare_tuples(v.out_edges())

    def update_colours(self, edges):
        for edge in edges:
            if edge.type == LineType.TRACKLET:
                chunk = self.project.chm[edge.id]
                c = self.assign_color(chunk)
                # import random
                # c = QtGui.QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                edge.color = c
        self.g.draw_lines()

    def assign_color(self, chunk):
        if chunk.is_only_one_id_assigned(len(self.project.animals)):
            id_ = list(chunk.P)[0]
            c_ = self.project.animals[id_].color_
            c = QtGui.QColor(c_[2], c_[1], c_[0], 255)
        else:
            # default
            c = QtGui.QColor(0, 0, 0, 120)
            # old version
            # c = self.project.chm[source_start_id].color
        return c

    def prepare_tuples(self, edges):
        for edge in edges:
            source = edge.source()
            target = edge.target()
            r1 = self.project.gm.region(source)
            r2 = self.project.gm.region(target)
            source_start_id = self.graph_manager.g.vp["chunk_start_id"][source]
            target_end_id = self.graph_manager.g.vp["chunk_end_id"][target]
            sureness = self.graph_manager.g.ep['score'][edge]

            if source_start_id == target_end_id and source_start_id != 0:
                type_of_line = LineType.TRACKLET
            else:
                type_of_line = LineType.LINE
            if not (r1 in self.regions and r2 in self.regions):
                type_of_line = LineType.PARTIAL

            if type_of_line == LineType.TRACKLET:
                tracklet = self.project.chm[source_start_id]
                c = self.assign_color(tracklet)
                line = GraphLine(tracklet.id(), r1, r2, type_of_line, sureness, c)
            else:
                line = GraphLine(0, r1, r2, type_of_line, sureness)

            # self.chunks_region_chunks[line] = RegionChunk(self.project.chm[source_start_id], self.graph_manager,
            #                                                    self.region_manager)

            self.edges.add(line)

    def get_node_info(self, region):
        n = self.regions_vertices[region]

        vertex = self.project.gm.g.vertex(int(n))
        best_out_score, _ = self.project.gm.get_2_best_out_vertices(vertex)
        best_in_score, _ = self.project.gm.get_2_best_in_vertices(vertex)

        ch = self.project.gm.is_chunk(vertex)
        ch_info = str(ch)

        # TODO
        # antlikeness = self.project.stats.antlikeness_svm.get_prob(region)[1]
        antlikeness = 0

        # TODO
        # return "Area = %i\nCentroid = %s\nMargin = %i\nBest in = %s\nBest out = %s\nChunk info = %s" % (region.area(), str(region.centroid()),
        #         region.margin_, str(best_in_score[0])+', '+str(best_in_score[1]), str(best_out_score[0])+', '+str(best_out_score[1]), ch_info)
        return "Centroid = %s\nArea = %i\nAntlikeness = %.3f\nMargin = %i\nBest in = %s\nBest out = %s\nChunk info = %s\nVertex id = %s" % \
               (str(region.centroid()), region.area(), antlikeness, region.margin_,
                str(best_in_score[0]) + ', ' + str(best_in_score[1]),
                str(best_out_score[0]) + ', ' + str(best_out_score[1]), ch_info, str(n))

    def get_edge_info(self, edge):
        return "Type = {0}\nSureness = {1}\nTracklet id: {2}".format(edge.type, edge.sureness, edge.id)

    def get_widget(self, frames=None, show_tracklet_callback=None):
        self.prepare_vertices(frames)
        # print("Preparing nodes...")
        self.prepare_nodes()
        # print("Preparing edges...")
        self.prepare_edges()
        # print("Preparing visualizer...")
        img_manager = ImgManager(self.project, max_size_mb=S_.cache.img_manager_size_MB)
        from gui.graph_widget.graph_visualizer import GraphVisualizer
        self.g = GraphVisualizer(self, img_manager, show_tracklet_callback)
        return self.g

    def get_chunk_by_id(self, id):
        return self.project.chm[id]



if __name__ == '__main__':
    project = Project()

    # sn_id = 2
    # name = 'Cam1_'
    # snapshot = {'chm': '/home/sheemon/FERDA/projects/'+name+'/.auto_save/'+str(sn_id)+'__chunk_manager.pkl',
    # 'gm': '/home/sheemon/FERDA/projects/'+name+'/.auto_save/'+str(sn_id)+'__graph_manager.pkl'}

    project.load('/home/simon/FERDA/projects/Cam1_/cam1.fproj')

    app = QtGui.QApplication(sys.argv)
    l1 = GraphWidgetLoader(project)
    l1.set_relative_margin(1)
    # l1.set_width(60)
    # l1.set_height(60)

    # g = l1.get_widget(range(300, 500))
    g = l1.get_widget()
    g.redraw()
    g.show()

    app.exec_()
    cv2.waitKey(0)
