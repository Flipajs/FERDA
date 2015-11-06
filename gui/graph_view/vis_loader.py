import pickle
from core.project.project import Project
from core.region.region_manager import RegionManager
from utils.img_manager import ImgManager
from graph_visualizer import GraphVisualizer
from PyQt4 import QtGui

__author__ = 'Simon Mandlik'


class VisLoader:

    def __init__(self, project=None):
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
        g = GraphVisualizer(self.regions, self.edges, img_manager)
        g.show()

if __name__ == '__main__':
    p = Project()
    p.load('/Users/flipajs/Documents/wd/eight_new/eight.fproj')
    from core.graph.graph_manager import GraphManager

    for i in range(1):
        rm_old = RegionManager(db_wd=p.working_directory + '/temp',
                                   db_name='part' + str(i) + '_rm.sqlite3')

        with open(p.working_directory + '/temp/part' + str(i) + '.pkl', 'rb') as f:
                up = pickle.Unpickler(f)
                g_ = up.load()
                relevant_vertices = up.load()
                chm_ = up.load()

        p.chm = chm_
        p.rm = rm_old
        p.gm = GraphManager(p, None)
        p.gm.g = g_

        for v_id in relevant_vertices:
            v = p.gm.g.vertex(v_id)
            r = p.rm[p.gm.g.vp['region_id'][v]]
            p.gm.vertices_in_t.setdefault(r.frame(), []).append(v_id)

    import cv2, sys
    app = QtGui.QApplication(sys.argv)
    im_manager = ImgManager(p, max_size_mb=0, max_num_of_instances=0)
    l = VisLoader(p)
    l.visualise()
    app.exec_()
    cv2.waitKey(0)
