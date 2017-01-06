import sys
import cPickle as pickle
from PyQt4 import QtGui

from gui import main_window
from core.settings import Settings as S_
from utils.misc import is_flipajs_pc
import time

app = QtGui.QApplication(sys.argv)
ex = main_window.MainWindow()
# ex.showMaximized()
ex.setFocus()

t_ = time.time()

from core.project.project import Project
project = Project()

S_.general.print_log = False

# This is development speed up process (kind of fast start). Runs only on developers machines...
if is_flipajs_pc():
    sn_id = 875
    cam_ = 1

    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_rf'
    wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'


    # project.load_semistate(wd, 'edge_cost_updated', update_t_nodes=True)
    # project.load_semistate(wd, 'id_classified_no_HIL_1')
    project.load_semistate(wd, 'tracklets_s_classified2')

    # project.load(wd)
    # project.save_semistate('init_state')

    # project.chm.reset_itree(project.gm)

    # project.load_snapshot(snapshot)

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

    # from core.id_detection.feature_manager import FeatureManager
    # fm = FeatureManager(project.working_directory, db_name='fm_idtracker_i_d50.sqlite3')


    ex.widget_control('load_project', project)



print "FERDA is READY, loaded in {:.3}s".format(time.time()-t_)
ex.move(-500, -500)
ex.showMaximized()
app.exec_()
app.deleteLater()
sys.exit()
