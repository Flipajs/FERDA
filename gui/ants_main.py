# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ants_main.ui'
#
# Created: Wed Nov 20 12:24:37 2013
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(266, 115)
        Dialog.setStyleSheet(_fromUtf8(""))
        self.choose_video_button = QtGui.QPushButton(Dialog)
        self.choose_video_button.setGeometry(QtCore.QRect(10, 10, 141, 27))
        self.choose_video_button.setObjectName(_fromUtf8("choose_video_button"))
        self.start_button = QtGui.QPushButton(Dialog)
        self.start_button.setGeometry(QtCore.QRect(170, 80, 87, 27))
        self.start_button.setObjectName(_fromUtf8("start_button"))
        self.file_name_label = QtGui.QLabel(Dialog)
        self.file_name_label.setGeometry(QtCore.QRect(13, 43, 441, 17))
        self.file_name_label.setObjectName(_fromUtf8("file_name_label"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "source selection", None, QtGui.QApplication.UnicodeUTF8))
        self.choose_video_button.setText(QtGui.QApplication.translate("Dialog", "choose video file", None, QtGui.QApplication.UnicodeUTF8))
        self.start_button.setText(QtGui.QApplication.translate("Dialog", "START", None, QtGui.QApplication.UnicodeUTF8))
        self.file_name_label.setText(QtGui.QApplication.translate("Dialog", "..", None, QtGui.QApplication.UnicodeUTF8))

