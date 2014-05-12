import sys
from PyQt4 import QtGui
from gui import init_window


app = QtGui.QApplication(sys.argv)
ex = init_window.InitWindow()

sys.exit(app.exec_())