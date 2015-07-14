import sys
from PyQt4 import QtGui, QtCore
from gui import ferda_window
from gui import main_window

app = QtGui.QApplication(sys.argv)
# ex = ferda_window.FerdaControls()
ex = main_window.MainWindow()
ex.showMaximized()
ex.setFocus()

from core.project import Project
proj = Project()
proj.load('/Users/flipajs/Documents/wd/eight/eight.fproj')
ex.widget_control('load_project', proj)

app.exec_()
app.deleteLater()
sys.exit()