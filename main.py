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

# project.load('/home/dita/PycharmProjects/c5__/c5__.fproj')
# ex.widget_control('load_project', project)
# This is development speed up process (kind of fast start). Runs only on developers machines...
if is_flipajs_pc():
    # project.load('/Users/flipajs/Documents/wd/F1C51/f1c51.fproj')
    # project.load('/Users/flipajs/Documents/wd/video_bounds_test/test.fproj')
    # project.load('/Users/flipajs/Documents/wd/small_lenses_colony1/small_lenses    .fproj')
    # project.load('/Users/flipajs/Documents/wd/colonies1_30m/colonies30m.fproj')
    # project.load('/Users/flipajs/Documents/wd/eight/eight.fproj')

    # project.load('/Users/flipajs/Documents/wd/eight_new/eight.fproj')
    # # project.load('/Users/flipajs/Documents/wd/c4/c4.fproj')
    # # project.load('/Users/flipajs/Documents/wd/c2/c2.fproj')
    # # # # # # #
    # ex.widget_control('load_project', project)

    # project.load('/Users/flipajs/Documents/wd/C210_2/C210.fproj')
    # ex.widget_control('load_project', project)

    project.load('/Users/flipajs/Documents/wd/GT/C210_5000/C210.fproj')
    ex.widget_control('load_project', project)

    # project.load('/Users/flipajs/Documents/wd/eight/eight.fproj')
    # ex.widget_control('load_project', project)
    pass


app.exec_()
app.deleteLater()
sys.exit()