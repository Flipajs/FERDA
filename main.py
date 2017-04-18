import sys
import cPickle as pickle
from PyQt4 import QtGui

from gui import main_window
from core.settings import Settings as S_
from utils.misc import is_flipajs_pc
import time
from core.project.project import Project
import timeit
import numpy as np


app = QtGui.QApplication(sys.argv)
ex = main_window.MainWindow()
ex.setFocus()

t_ = time.time()

project = Project()

S_.general.print_log = False


# def test1():
#     for n in project.gm.g.vertices():
#         pass
#
#
# def test2():
#     for i, n in enumerate(project.gm.g.vertices()):
#         pass
#
#
# def test3():
#     a = project.gm.g.get_vertices()
#     for i, n in enumerate(np.nditer(project.gm.g.get_vertices())):
#         pass

def test1():
    for n in project.gm.g.edges():
        pass

def test2():
    for i, n in enumerate(project.gm.g.edges()):
        pass

def test3():
    a = project.gm.g.get_vertices()
    for i, n in enumerate(np.nditer(project.gm.g.get_edges())):
        pass

# This is development speed up process (kind of fast start). Runs only on developers machines...
# if is_flipajs_pc() and False:
if is_flipajs_pc():
    # wd = '/Users/iflipajs/Documents/wd/FERDA/Cam1_rf'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/test6'
    # wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_rfs'
    wd = '/Users/flipajs/Documents/wd/FERDA/rep1-cam3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'

    # wd = '/Users/flipajs/Documents/wd/FERDA/test'

    project.load(wd)

    # start_time = timeit.default_timer()
    # test1()
    # print(timeit.default_timer() - start_time)
    #
    # start_time = timeit.default_timer()
    # test2()
    # print(timeit.default_timer() - start_time)
    #
    # start_time = timeit.default_timer()
    # test3()
    # print(timeit.default_timer() - start_time)

    # project.chm.add_single_vertices_chunks(project)
    # project.save()

    # project.load_semistate(wd, 'edge_cost_updated', update_t_nodes=True)
    # project.load_semistate(wd, 'first_tracklets')
    # project.load_semistate(wd, 'lp_id_SEG_IDCR_0')
    # project.load_semistate(wd, 'lp_HIL_INIT3_0')

    # project.gm.update_nodes_in_t_refs()


    # from core.region.region_manager import RegionManager
    # from core.graph.chunk_manager import ChunkManager
    #
    # project.rm = RegionManager(project.working_directory + '/temp', db_name='part0_rm.sqlite3')
    # with open(project.working_directory + '/temp/part0.pkl', 'rb') as f:
    #     up = pickle.Unpickler(f)
    #     g_ = up.load()
    #
    # project.gm.g = g_
    # project.gm.rm = project.rm
    # project.chm = ChunkManager()
    #
    # print "#edges: {}".format(project.gm.g.num_edges())
    # from core.graph.solver import Solver
    # project.gm.update_nodes_in_t_refs()
    # solver = Solver(project)
    #
    # num_changed2 = solver.simplify(rules=[solver.adaptive_threshold])
    # num_changed1 = solver.simplify(rules=[solver.update_costs])
    # num_changed2 = solver.simplify(rules=[solver.adaptive_threshold])
    #
    # print "#edges: {}".format(project.gm.g.num_edges())
    # project.chm.add_single_vertices_chunks(project)

    try:
        # WORKAROUND:
        for t in project.chm.chunk_gen():
            if not hasattr(t, 'N'):
                t.N = set()
                t.P = set()
    except AttributeError:
        pass

    from utils.color_manager import colorize_project
    colorize_project(project)
    ex.widget_control('load_project', project)
    # ex.move(-500, -500)
    ex.showMaximized()

print "FERDA is READY, loaded in {:.3}s".format(time.time()-t_)

app.exec_()
app.deleteLater()
sys.exit()
