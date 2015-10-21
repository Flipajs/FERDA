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
        self.edges = []
        self.regions = []

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
        # for vertex in self.vertices:
        #     # v = self.graph.vertex(vertex)
        self.regions = self.region_manager[self.vertices]

    def prepare_edges(self):
        for vertex in self.vertices:
            v = self.graph.vertex(vertex)
            for edge in v.out_edges():
                source = edge.source()
                target = edge.target()
                r1 = self.region_manager[source]
                r2 = self.region_manager[target]

                # visualizer requires tuple of length 4
                import random
                type = random.choice(["chunk", "line"])
                if type == "chunk":
                    sureness = 1
                else:
                    sureness = random.randint(0, 101) / float(100)

                new_tuple = (r1, r2, type, sureness)
                self.edges.append(new_tuple)

    def visualise(self):
        self.prepare_vertices()
        self.prepare_edges()
        self.prepare_nodes()
        img_manager = ImgManager(p)
        g = GraphVisualizer(self.regions, self.edges, img_manager)
        g.show()

if __name__ == '__main__':
    from scripts import fix_project
    execfile("/home/sheemon/FERDA/ferda/scripts/fix_project.py")

    p = Project()
    p.load('/home/sheemon/FERDA/projects/eight_new/eight.fproj')

    # # test
    # p.rm = RegionManager(db_wd=p.working_directory+'/temp', db_name='regions_part_'+str(id)+'.sqlite3')
    # f = open('/home/sheemon/Downloads/c5regions.pkl', 'r+b')
    # up = pickle.Unpickler(f)
    # regions = up.load()
    # for r in regions:
    #     r.pts_rle_ = None
    # f.close()
    #
    # p.rm = RegionManager(db_wd="/home/dita", cache_size_limit=1)
    # p.rm.add(regions)

    import cv2
    # solver = p.saved_progress['solver']
    # nodes = solver.g.nodes()
    # edges = solver.g.edges()

    # vertices = p.gm.get_all_relevant_vertices()
    # edges = []
    # for vertex in vertices:
    #     v = p.gm.g.vertex(vertex)
    #     for edge in v.out_edges():
    #         edges.append(edge)
    # regions = []
    # for vertex in vertices:
    #     regions.append(p.rm[vertex])
    # pass

    # edges_4_tuple = []
    # import random
    # for edge in edges:
    #     type = random.choice(["chunk", "line"])
    #     if type == "chunk":
    #         sureness = 1
    #     else:
    #         sureness = random.randint(0, 101) / float(100)
    #
    #     new_tuple = edge + (type, sureness)
    #     edges_4_tuple.append(new_tuple)

    import sys
    app = QtGui.QApplication(sys.argv)
    im_manager = ImgManager(p)
    l = VisLoader(p)
    l.visualise()
    app.exec_()
    cv2.waitKey(0)
