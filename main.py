import sys

from PyQt4 import QtGui

from gui import main_window
from core.settings import Settings as S_
from utils.misc import is_flipajs_pc

app = QtGui.QApplication(sys.argv)
ex = main_window.MainWindow()
# ex.showMaximized()
ex.setFocus()

from core.project.project import Project
project = Project()

S_.general.print_log = False

# This is development speed up process (kind of fast start). Runs only on developers machines...
if is_flipajs_pc() and False:
    sn_id = 2
    name = 'big_lenses'
    wd = '/Users/flipajs/Documents/wd/GT/'
    # snapshot = {'chm': wd+name+'/.auto_save/'+str(sn_id)+'__chunk_manager.pkl',
    #             'gm': wd+name+'/.auto_save/'+str(sn_id)+'__graph_manager.pkl'}

    project.load(wd+name+'/big_lenses.fproj')
    ex.widget_control('load_project', project)


app.exec_()
app.deleteLater()
sys.exit()
