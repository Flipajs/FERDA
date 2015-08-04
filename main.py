import sys
from PyQt4 import QtGui
from gui import main_window
from core.settings import Settings as S_
from utils.misc import is_flipajs_pc

app = QtGui.QApplication(sys.argv)
ex = main_window.MainWindow()
ex.showMaximized()
ex.setFocus()

from core.project import Project
project = Project()

S_.general.print_log = False

# This is development speed up process (kind of fast start). Runs only on developers machines...
if is_flipajs_pc():
    project.load('/Users/flipajs/Documents/wd/colonies_crop1/colonies.fproj')
    S_.general.print_log = False
    S_.parallelization.processes_num = 3
    S_.parallelization.frames_in_row = 50
    ex.widget_control('load_project', project)

app.exec_()
app.deleteLater()
sys.exit()