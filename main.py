import sys
from PyQt4 import QtGui
from gui.init_window import init_window
from gui import ferda_window


# app = QtGui.QApplication(sys.argv)
# ex = init_window.InitWindow()
#
# sys.exit(app.exec_())

#TEST

app = QtGui.QApplication(sys.argv)
ex = ferda_window.FerdaControls()

app.exec_()
app.deleteLater()
sys.exit()