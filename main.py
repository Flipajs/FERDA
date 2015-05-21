import sys
from PyQt4 import QtGui
from gui import ferda_window
from gui import main_window

app = QtGui.QApplication(sys.argv)
# ex = ferda_window.FerdaControls()
ex = main_window.MainWindow()

app.exec_()
app.deleteLater()
sys.exit()