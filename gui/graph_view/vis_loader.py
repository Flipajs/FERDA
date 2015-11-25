from core.project.project import Project
from utils.img_manager import ImgManager
from PyQt4 import QtGui

__author__ = 'Simon Mandlik'

# the width of a node
WIDTH = 40
# the width of a node
HEIGHT = 40
# relative margin of a node
RELATIVE_MARGIN = 1
# distance of the whole visualization from the top of the widget
FROM_TOP = 20
# space between each of the columns
SPACE_BETWEEN_HOR = 30
# space between nodes in column
SPACE_BETWEEN_VER = 30
# gap between frame_numbers and first node in columns
GAP = 50
# number of columns to be displayed before dynamic loading, 0 means dynamic loading for all
MINIMUM = 20
# number of columns processed in one chunk sent to dummy thread, for debbuging purpose
COLUMNS_TO_LOAD = 3
# Opacity of the colors
OPACITY = 255
# default text to display
DEFAULT_TEXT = "V - toggle vertical display; C - compress axis; I, O - zoom in or out; Q, W - shrink, " \
               "stretch; A, S show/hide info for selected; T - show toggled node"


class VisLoader:

    def __init__(self, project=None, width=WIDTH, height=HEIGHT, relative_margin=RELATIVE_MARGIN):
        self.project = project
        self.graph_manager = None
        self.graph = None
        self.region_manager = None
        self.update_structures()

        self.vertices = []
        self.edges = set()
        self.regions = set()

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

    def prepare_vertices(self):
        self.vertices = self.graph_manager.get_all_relevant_vertices()

    def prepare_nodes(self):
        for v in self.vertices:
            region = self.graph_manager.region(v)
            self.regions_vertices[region] = v
            self.regions.add(region)

    def prepare_edges(self):
        for vertex in self.vertices:
            v = self.graph.vertex(vertex)
            self.prepare_tuples(v.in_edges())
            self.prepare_tuples(v.out_edges())

    def prepare_tuples(self, edges):
        for edge in edges:
            source = edge.source()
            target = edge.target()
            r1 = self.project.gm.region(source)
            r2 = self.project.gm.region(target)
            source_start_id = self.graph_manager.g.vp["chunk_start_id"][source]
            target_end_id = self.graph_manager.g.vp["chunk_end_id"][target]
            sureness = self.graph_manager.g.ep['score'][edge]
            type_of_line = "chunk" if source_start_id == target_end_id and source_start_id != 0 else "line"
            if not(r1 in self.regions and r2 in self.regions):
                type_of_line = "partial"

            new_tuple = (r1, r2, type_of_line, sureness)
            self.edges.add(new_tuple)

    def get_node_info(self, region):
        n = self.regions_vertices[region]

        # antlikeness = self.parent.solver.project.stats.antlikeness_svm.get_prob(region)[1]
        # virtual = False
        #
        # try:
        #     if region.is_virtual:
        #         antlikeness = 1.0
        #         virtual = True
        # except:
        #     pass

        vertex = self.project.gm.g.vertex(int(n))
        best_out_score, _ = self.project.gm.get_2_best_out_vertices(vertex)
        best_in_score, _ = self.project.gm.get_2_best_in_vertices(vertex)

        ch = self.project.gm.is_chunk(vertex)
        ch_info = str(ch)

        return "Area = %i\nCentroid = %s\nMargin = %i\nBest in = %s\nBest out = %s\nChunk info = %s" % (region.area(), str(region.centroid()),
                region.margin_, str(best_in_score[0])+', '+str(best_in_score[1]), str(best_out_score[0])+', '+str(best_out_score[1]), ch_info)

    def get_edge_info(self, edge):
        return "Type = {0}\nSureness = {1}".format(edge[2], edge[3])

    def visualise(self):
        self.prepare_vertices()
        self.prepare_nodes()
        self.prepare_edges()
        img_manager = ImgManager(self.project)

        from graph_visualizer import GraphVisualizer
        self.g = GraphVisualizer(self, img_manager)
        self.g.show()

if __name__ == '__main__':
    from scripts import fix_project
    p = Project()
    p.load('/home/sheemon/FERDA/projects/eight_new_issue/eight.fproj')

    import cv2, sys
    app = QtGui.QApplication(sys.argv)
    l1 = VisLoader(p)
    l1.set_relative_margin(0.3)
    l1.set_width(40)
    l1.set_height(40)
    l1.visualise()
    app.exec_()
    cv2.waitKey(0)
