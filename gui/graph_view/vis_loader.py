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
DEFAULT_TEXT = "V - toggle vertical display; C - compress axis; I, O - zoom in or out; Q, W - shrink, stretch; A show info for selected; T - show toggled node"


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

            source_start_id = self.graph_manager.g.vp["chunk_start_id"][source]
            target_end_id = self.graph_manager.g.vp["chunk_end_id"][target]

            sureness = self.graph_manager.g.ep['score'][edge]

            type_of_line = "chunk" if source_start_id == target_end_id and source_start_id != 0 else "line"
            if not(r1 in self.regions and r2 in self.regions):
                type_of_line = "partial"

            new_tuple = (r1, r2, type_of_line, sureness)
            self.edges.add(new_tuple)

    def visualise(self):
        self.prepare_vertices()
        self.prepare_nodes()
        self.prepare_edges()
        img_manager = ImgManager(self.project)

        from graph_visualizer import GraphVisualizer
        self.g = GraphVisualizer(self.regions, self.edges, img_manager, self.relative_margin, self.width, self.height)
        self.g.show()

if __name__ == '__main__':
    from scripts import fix_project
    p = Project()
    p.load('/home/sheemon/FERDA/projects/eight_new/eight.fproj')
    # from core.graph.graph_manager import GraphManager
    #
    # for i in range(1):
    #     rm_old = RegionManager(db_wd=p.working_directory + '/temp',
    #                                db_name='part' + str(i) + '_rm.sqlite3')
    #
    #     with open(p.working_directory + '/temp/part' + str(i) + '.pkl', 'rb') as f:
    #             up = pickle.Unpickler(f)
    #             g_ = up.load()
    #             relevant_vertices = up.load()
    #             chm_ = up.load()
    #
    #     p.chm = chm_
    #     p.rm = rm_old
    #     p.gm = GraphManager(p, None)
    #     p.gm.g = g_
    #
    #     for v_id in relevant_vertices:
    #         v = p.gm.g.vertex(v_id)
    #         r = p.rm[p.gm.g.vp['region_id'][v]]
    #         p.gm.vertices_in_t.setdefault(r.frame(), []).append(v_id)

    import cv2, sys
    app = QtGui.QApplication(sys.argv)
    # im_manager = ImgManager(p, max_size_mb=0, max_num_of_instances=0)
    # l = VisLoader(p)
    # l.visualise()
    l1 = VisLoader(p)
    l1.set_relative_margin(0.3)
    l1.set_width(40)
    l1.set_height(40)
    l1.visualise()
    app.exec_()
    cv2.waitKey(0)
