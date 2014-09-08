__author__ = 'flipajs'

import sys, os
from tests import auto_run_ferda
from PyQt4 import QtCore, QtGui

app = QtGui.QApplication(sys.argv)

try:
    os.makedirs('/home/flipajs/dump/test2/aaa')
except:
    pass

auto_run_ferda.run_ferda_with_params('a')
auto_run_ferda.run_ferda_with_params('a')


sys.exit()