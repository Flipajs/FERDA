import pickle
from core.project.project import Project
from core.region.region_manager import RegionManager
from utils.img_manager import ImgManager
from PyQt4 import QtGui

__author__ = 'Simon Mandlik'

# the width of a node
WIDTH = 40
# the width of a node
HEIGHT = 40
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


class VisLoader:

    def __init__(self, project=None, ):
        self.project = project
        self.graph_manager = None
        self.graph = None
        self.region_manager = None
        self.update_structures()

        self.vertices = []
        self.edges = set()
        self.regions = set()

    def set_project(self, project):
        self.project = project

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
            self.regions.add(self.graph_manager.region(v))

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
            # TODO
            # self.graph_manager.g.vp["chunk_start_id"]
            # visualizer requires tuple of length 4
            sureness = self.graph_manager.g.ep['score'][self.graph_manager.g.edge(source, target)]
            type_of_line = "chunk" if abs(sureness) == 1 else "line"
            if not(r1 in self.regions and r2 in self.regions):
                type_of_line = "partial"

            new_tuple = (r1, r2, type_of_line, sureness)
            self.edges.add(new_tuple)

    def visualise(self):
        self.prepare_vertices()
        self.prepare_nodes()
        self.prepare_edges()
        img_manager = ImgManager(p)
        from graph_visualizer import GraphVisualizer
        g = GraphVisualizer(self.regions, self.edges, img_manager)
        g.show()

if __name__ == '__main__':
    from scripts import fix_project

    p = Project()
    p.load('/home/sheemon/FERDA/projects/eight_new/eight.fproj')

    import cv2, sys
    app = QtGui.QApplication(sys.argv)
    im_manager = ImgManager(p)
    l = VisLoader(p)
    l.visualise()
    app.exec_()
    cv2.waitKey(0)
