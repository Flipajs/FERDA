import sys
from PyQt4 import QtGui, QtCore
from gui import ferda_window
from gui import main_window
from core.settings import Settings as S_
from utils.misc import is_flipajs_pc

app = QtGui.QApplication(sys.argv)
# ex = ferda_window.FerdaControls()
ex = main_window.MainWindow()
ex.showMaximized()
ex.setFocus()

from core.project import Project
proj = Project()

S_.general.print_log = False
if is_flipajs_pc():
    proj.load('/Users/flipajs/Documents/wd/eight/eight.fproj')
    S_.general.print_log = False
    ex.widget_control('load_project', proj)

    # proj.load('/Users/flipajs/Documents/wd/crop_1h00/c1.fproj')
    # ex.widget_control('load_project', proj)

app.exec_()
app.deleteLater()
sys.exit()