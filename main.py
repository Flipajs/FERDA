import sys
from PyQt4 import QtGui, QtCore
from gui import ferda_window
from gui import main_window

app = QtGui.QApplication(sys.argv)
# ex = ferda_window.FerdaControls()
ex = main_window.MainWindow()
ex.showMaximized()

# from core.project import Project
# proj = Project()
# proj.load('/home/flipajs/Documents/wd/c6_2/c6.fproj')
# ex.widget_control('load_project', proj)

app.exec_()
app.deleteLater()
sys.exit()