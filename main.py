import sys
from PyQt4 import QtGui
from gui import source_window

app = QtGui.QApplication(sys.argv)
ex = source_window.SourceWindow()

sys.exit(app.exec_())