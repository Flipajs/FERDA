import sys
import cPickle as pickle
from PyQt4 import QtGui

from gui import main_window
from core.settings import Settings as S_
from utils.misc import is_flipajs_pc, is_matejs_pc
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


# This is development speed up process (kind of fast start). Runs only on developers machines...
# if is_flipajs_pc() and False:
wd = None
if is_flipajs_pc():
    # wd = '/Users/iflipajs/Documents/wd/FERDA/Cam1_rf'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/test6'
    # wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_rfs2'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1'
    wd = '/Users/flipajs/Documents/wd/FERDA/rep1-cam2'
    # wd = '/Users/flipajs/Documents/wd/FERDA/rep1-cam3'

    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'

    # wd = '/Users/flipajs/Documents/wd/FERDA/test'

if is_matejs_pc():
    # wd = '/home/matej/prace/ferda/10-15/'
    # wd = '/home/matej/prace/ferda/10-15 (copy)/'
    pass

if wd is not None:
    project.load(wd)

    # from scripts.regions_stats import decide_one2one
    # decide_one2one(project)

    # project.chm.add_single_vertices_chunks(project)
    # project.save()

    # project.load_semistate(wd, 'edge_cost_updated', update_t_nodes=True)
    # project.load_semistate(wd, 'first_tracklets')
    # project.load_semistate(wd, 'lp_id_SEG_IDCR_0')
    # project.load_semistate(wd, 'lp_HIL_INIT3_0')

    # project.gm.update_nodes_in_t_refs()

    try:
        # old projects WORKAROUND:
        for t in project.chm.chunk_gen():
            if not hasattr(t, 'N'):
                t.N = set()
                t.P = set()
    except AttributeError:
        pass

    ex.widget_control('load_project', project)
    # colorize_project(project)
    # ex.move(-500, -500)
    ex.showMaximized()

print "FERDA is READY, loaded in {:.3}s".format(time.time()-t_)

app.exec_()
app.deleteLater()
sys.exit()
