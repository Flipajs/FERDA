# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'img_controls.ui'
#
# Created: Mon Jul 14 10:52:04 2014
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(967, 628)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.viewPositions = QtGui.QPushButton(self.centralwidget)
        self.viewPositions.setGeometry(QtCore.QRect(0, 0, 98, 27))
        self.viewPositions.setObjectName(_fromUtf8("viewPositions"))
        self.positionsLabel = QtGui.QLabel(self.centralwidget)
        self.positionsLabel.setGeometry(QtCore.QRect(0, 30, 91, 281))
        self.positionsLabel.setObjectName(_fromUtf8("positionsLabel"))
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.viewPositions.setText(_translate("MainWindow", "view positions", None))
        self.positionsLabel.setText(_translate("MainWindow", "TextLabel", None))

