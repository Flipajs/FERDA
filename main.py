import sys

from PyQt4 import QtGui

from gui import main_window
from core.settings import Settings as S_
from utils.misc import is_flipajs_pc
from utils.video_manager import get_auto_video_manager

app = QtGui.QApplication(sys.argv)
ex = main_window.MainWindow()
# ex.showMaximized()
ex.setFocus()

from core.project.project import Project
project = Project()

S_.general.print_log = False

# This is development speed up process (kind of fast start). Runs only on developers machines...
if is_flipajs_pc():
    project.load('/Users/flipajs/Documents/wd/F1C51/f1c51.fproj')
    # project.load('/Users/flipajs/Documents/wd/eight/eight.fproj')
    # #
    ex.widget_control('load_project', project)

    pass

app.exec_()
app.deleteLater()
sys.exit()