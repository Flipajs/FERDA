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

    # name = 'S9T91min_'
    # name = 'Cam'+str(cam_)+' copy'

    name = 'Cam'+str(cam_)
    wd = '/Users/flipajs/Documents/wd/gt/'
    # wd = '/Users/flipajs/Documents/wd/'
    snapshot = {'chm': wd+name+'/.auto_save/'+str(sn_id)+'__chunk_amanager.pkl',
                'gm': wd+name+'/.auto_save/'+str(sn_id)+'__graph_manager.pkl'}

    project.load(wd+name)
    # project.load_snapshot(snapshot)

    try:
        full_set = set(range(6))
        # WORKAROUND:
        for t in project.chm.chunk_gen():
            if not hasattr(t, 'N'):
                t.N = set()
                t.P = set()

    except AttributeError:
        pass

    ex.widget_control('load_project', project)

print "FERDA is READY, loaded in {:.3}s".format(time.time()-t_)
ex.move(-500, -500)
ex.showMaximized()
app.exec_()
app.deleteLater()
sys.exit()
