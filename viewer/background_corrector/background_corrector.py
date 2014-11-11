import sys
import background_corrector_core
from PyQt4 import QtGui


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = background_corrector_core.BackgroundCorrector()
    sys.exit(app.exec_())