import sys
import cPickle as pickle
from PyQt4 import QtGui

from gui import main_window
from core.settings import Settings as S_
from utils.misc import is_flipajs_pc
import time
from core.project.project import Project


app = QtGui.QApplication(sys.argv)
ex = main_window.MainWindow()
ex.setFocus()

t_ = time.time()

project = Project()

S_.general.print_log = False

# This is development speed up process (kind of fast start). Runs only on developers machines...
if is_flipajs_pc() and False:
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_rf'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'

    # project.load_semistate(wd, 'edge_cost_updated', update_t_nodes=True)
    # project.load_semistate(wd, 'id_classified_HIL_init_0')
    # project.load_semistate(wd, 'lp_id_SEG_IDCR_0')
    project.load_semistate(wd, 'lp_HIL_INIT3_0')

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
    ex.move(-500, -500)
    ex.showMaximized()

print "FERDA is READY, loaded in {:.3}s".format(time.time()-t_)

app.exec_()
app.deleteLater()
sys.exit()
