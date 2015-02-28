import sys
from PyQt4 import QtGui
from gui import ferda_window

app = QtGui.QApplication(sys.argv)
ex = ferda_window.FerdaControls()

app.exec_()
app.deleteLater()
sys.exit()/home/tomas