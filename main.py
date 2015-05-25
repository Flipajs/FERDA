import sys
from PyQt4 import QtGui, QtCore
from gui import ferda_window
from gui import main_window

s = QtCore.QSettings('FERDA')
settings = {}
for k in s.allKeys():
    settings[str(k)] = str(s.value(k, 0, str))

print settings

app = QtGui.QApplication(sys.argv)
# ex = ferda_window.FerdaControls()
ex = main_window.MainWindow()
ex.showMaximized()

app.exec_()
app.deleteLater()
sys.exit()