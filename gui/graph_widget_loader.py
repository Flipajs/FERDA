from __future__ import print_function
from __future__ import unicode_literals
from builtins import str
from builtins import range
import sys
import cv2
from PyQt4 import QtGui

from core.graph.region_chunk import RegionChunk
from core.project.project import Project
from gui.graph_widget.graph_line import LineType, GraphLine, Overlap
from gui.settings import Settings as S_
from core.config import config
from utils.img_manager import ImgManager
from gui.generated.ui_graph_widget import Ui_graph_widget
from utils.video_manager import get_auto_video_manager

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

from gui.graph_widget.graph_visualizer import GraphVisualizer


class GraphWidgetLoader(QtGui.QWidget):
    def __init__(self, project=None, width=WIDTH, height=HEIGHT, relative_margin=RELATIVE_MARGIN, tracklet_callback=None):
        super(GraphWidgetLoader, self).__init__()

        self.project = project

        self.graph_manager = None
        self.graph = None
        self.region_manager = None
        self.update_structures()

        self.vertices = set()
        self.edges = set()
        self.regions = set()

        self.chunks_region_chunks = {}
        self.regions_vertices = {}

        self.g = None

        self.relative_margin = relative_margin
        self.height = height
        self.width = width
        self.tracklet_callback = tracklet_callback

        # Set up the user interface from Designer.
        self.ui = Ui_graph_widget()
        self.ui.setupUi(self)
        self.ui.spinBoxTo.setValue(100)
        self.ui.spinBoxTo.setMaximum(get_auto_video_manager(self.project).total_frame_count() - 1)
        self.update_range_minmax()
        self.ui.spinBoxTo.valueChanged.connect(self.update_range_minmax)
        self.ui.spinBoxFrom.valueChanged.connect(self.update_range_minmax)
        self.ui.pushButtonDrawGraph.clicked.connect(self.draw_graph)
        self.graph_visualizer = self.ui.graph_visualizer
        self.draw_graph()

    def update_range_minmax(self):
        self.ui.spinBoxTo.setMinimum(self.ui.spinBoxFrom.value() + 1)
        self.ui.spinBoxFrom.setMaximum(self.ui.spinBoxTo.value() - 1)

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
            print("No project set!")

    def prepare_vertices(self, frames):
        if frames is None:
            self.vertices = set(self.graph_manager.get_all_relevant_vertices())
        else:
            self.vertices = []

            for f in frames:
                self.vertices.extend(self.graph_manager.get_vertices_in_t(f))

            self.vertices = set(self.vertices)

    def prepare_nodes(self):
        # TODO: asking for regions in one single run will be much faster
        for vertex in self.vertices:
            region = self.graph_manager.region(vertex)
            self.regions_vertices[region] = vertex
            self.regions.add(region)

    def prepare_lines(self, first_frame, last_frame):
        for vertex in self.vertices:
            v = self.graph.vertex(vertex)
            # self.prepare_tuples(v.in_edges()) # duplicates in output!
            self.prepare_tuples(v.out_edges(), first_frame, last_frame)

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

    def prepare_tracklets(self, frames):
        tracklets = self.project.chm.chunks_in_interval(frames[0], frames[-1])
        for tracklet in tracklets:
            region_chunk = RegionChunk(tracklet, self.graph_manager, self.region_manager)
            # replace tracklet ends with their ends in range
            # only tracklets can be partial
            r1, r2 = region_chunk[0], region_chunk[-1]
            left_overlap = tracklet.start_frame(self.graph_manager) < frames[0]
            right_overlap = tracklet.end_frame(self.graph_manager) > frames[-1]
            c = self.assign_color(tracklet)

            if left_overlap or right_overlap:
                type_of_line = LineType.PARTIAL_TRACKLET
                overlap = Overlap(left=left_overlap, right=right_overlap)

                if left_overlap:
                    r1 = region_chunk.region_in_t(frames[0])
                    self.regions.add(r1)
                if right_overlap:
                    r2 = region_chunk.region_in_t(frames[-1])
                    self.regions.add(r2)

                line = GraphLine(tracklet.id(), r1, r2, type_of_line, overlap=overlap, color=c)
            else:
                self.regions.add(r1)
                self.regions.add(r2)
                type_of_line = LineType.TRACKLET
                line = GraphLine(tracklet.id(), r1, r2, type_of_line, color=c)
            self.chunks_region_chunks[line] = RegionChunk(tracklet, self.graph_manager, self.region_manager)
            # print tracklet.id(), type_of_line, left_overlap, right_overlap

            self.edges.add(line)

    def prepare_tuples(self, edges, first_frame, last_frame):
        for edge in edges:
            source = edge.source()
            target = edge.target()
            r1 = self.project.gm.region(source)
            r2 = self.project.gm.region(target)
            if r1.frame_ == last_frame or r2.frame_ == first_frame:
                continue # out of snaps
            source_start_id = self.graph_manager.g.vp["chunk_start_id"][source]
            target_end_id = self.graph_manager.g.vp["chunk_end_id"][target]

            appearance_score = self.graph_manager.g.ep['score'][edge]
            try:
                movement_score = self.graph_manager.g.ep['movement_score'][edge]
            except:
                movement_score = 0

            if source_start_id != target_end_id or source_start_id == 0:
                line = GraphLine(0, r1, r2, LineType.LINE, appearance_score=appearance_score, movement_score=movement_score)
                self.edges.add(line)

            # self.chunks_region_chunks[line] = RegionChunk(self.project.chm[source_start_id], self.graph_manager,
            #                                                   self.region_manager)

    def get_node_info(self, region):
        n = self.regions_vertices[region]

        vertex = self.project.gm.g.vertex(int(n))
        best_out_score, _ = self.project.gm.get_2_best_out_vertices(vertex)
        best_in_score, _ = self.project.gm.get_2_best_in_vertices(vertex)

        ch, _ = self.project.gm.is_chunk(vertex)
        ch_info = str(ch)

        # TODO
        # antlikeness = self.project.stats.antlikeness_svm.get_prob(region)[1]
        antlikeness = 0

        # TODO
        # return "Area = %i\nCentroid = %s\nMargin = %i\nBest in = %s\nBest out = %s\nChunk info = %s" % (region.area(), str(region.centroid()),
        #         region.margin_, str(best_in_score[0])+', '+str(best_in_score[1]), str(best_out_score[0])+', '+str(best_out_score[1]), ch_info)
        return "Centroid = %s\nArea = %i\nAntlikeness = %.3f\nMargin = %i\nBest in = %s\nBest out = %s\nChunk info = %s\nVertex/region id = %s/%s" % \
               (str(region.centroid()), region.area(), antlikeness, region.margin_,
                str(best_in_score[0]) + ', ' + str(best_in_score[1]),
                str(best_out_score[0]) + ', ' + str(best_out_score[1]), ch_info, str(n), str(region.id()))

    def get_edge_info(self, edge):
        return "Type = {}\nAppearance score = {}\nMovement score={}\nScore product={}\nTracklet id: {}".format(edge.type, edge.appearance_score, edge.movement_score, edge.appearance_score * edge.movement_score, edge.id)

    def draw_graph(self):
        frames = list(range(self.ui.spinBoxFrom.value(), self.ui.spinBoxTo.value()))
        self.prepare_vertices(frames)
        # print("Preparing nodes...")
        self.prepare_nodes()
        # print("Preparing edges...")
        self.prepare_lines(frames[0], frames[-1])
        self.prepare_tracklets(frames)
        # print("Preparing visualizer...")
        img_manager = ImgManager(self.project, max_size_mb=config['cache']['img_manager_size_MB'])
        self.ui.verticalLayout.removeWidget(self.graph_visualizer)
        self.graph_visualizer = GraphVisualizer(self, img_manager, self.tracklet_callback)
        self.ui.verticalLayout.addWidget(self.graph_visualizer)
        self.graph_visualizer.redraw()

    def get_chunk_by_id(self, id):
        return self.project.chm[id]


if __name__ == '__main__':
    project = Project()

    # sn_id = 2
    # name = 'Cam1_'
    # snapshot = {'chm': '/home/sheemon/FERDA/projects/'+name+'/.auto_save/'+str(sn_id)+'__chunk_manager.pkl',
    # 'gm': '/home/sheemon/FERDA/projects/'+name+'/.auto_save/'+str(sn_id)+'__graph_manager.pkl'}

    project.load('/home/simon/FERDA/projects/clusters_gt/Cam1_/cam1.fproj')

    app = QtGui.QApplication(sys.argv)
    l1 = GraphWidgetLoader(project)
    l1.set_relative_margin(1)
    # l1.set_width(60)
    # l1.set_height(60)

    g = l1.get_widget(list(range(300, 400)))
    # g = l1.get_widget()
    g.redraw()
    g.show()

    app.exec_()
    cv2.waitKey(0)
