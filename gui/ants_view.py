# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ants_view.ui'
#
# Created: Thu Jan  9 02:34:07 2014
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
        Dialog.resize(342, 367)
        self.groupBox = QtGui.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(10, 260, 321, 101))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.b_choose_path = QtGui.QPushButton(self.groupBox)
        self.b_choose_path.setGeometry(QtCore.QRect(10, 30, 100, 27))
        self.b_choose_path.setObjectName(_fromUtf8("b_choose_path"))
        self.i_file_name = QtGui.QLineEdit(self.groupBox)
        self.i_file_name.setGeometry(QtCore.QRect(112, 30, 178, 27))
        self.i_file_name.setObjectName(_fromUtf8("i_file_name"))
        self.b_save_file = QtGui.QPushButton(self.groupBox)
        self.b_save_file.setGeometry(QtCore.QRect(192, 59, 98, 27))
        self.b_save_file.setObjectName(_fromUtf8("b_save_file"))
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(290, 35, 66, 20))
        self.label.setObjectName(_fromUtf8("label"))
        self.groupBox_2 = QtGui.QGroupBox(Dialog)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 10, 320, 80))
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.b_play = QtGui.QPushButton(self.groupBox_2)
        self.b_play.setGeometry(QtCore.QRect(10, 30, 90, 27))
        self.b_play.setObjectName(_fromUtf8("b_play"))
        self.b_stap_by_step = QtGui.QPushButton(self.groupBox_2)
        self.b_stap_by_step.setGeometry(QtCore.QRect(110, 30, 101, 27))
        self.b_stap_by_step.setObjectName(_fromUtf8("b_stap_by_step"))
        self.label_2 = QtGui.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(216, 0, 48, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.l_frame = QtGui.QLabel(self.groupBox_2)
        self.l_frame.setGeometry(QtCore.QRect(264, 0, 58, 17))
        self.l_frame.setObjectName(_fromUtf8("l_frame"))
        self.groupBox_3 = QtGui.QGroupBox(Dialog)
        self.groupBox_3.setGeometry(QtCore.QRect(9, 89, 321, 101))
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.ch_mser_collection = QtGui.QCheckBox(self.groupBox_3)
        self.ch_mser_collection.setGeometry(QtCore.QRect(10, 20, 201, 22))
        self.ch_mser_collection.setChecked(False)
        self.ch_mser_collection.setObjectName(_fromUtf8("ch_mser_collection"))
        self.ch_ants_collection = QtGui.QCheckBox(self.groupBox_3)
        self.ch_ants_collection.setGeometry(QtCore.QRect(10, 40, 161, 22))
        self.ch_ants_collection.setChecked(True)
        self.ch_ants_collection.setObjectName(_fromUtf8("ch_ants_collection"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "experiment controls", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("Dialog", "Save results to .mat file", None, QtGui.QApplication.UnicodeUTF8))
        self.b_choose_path.setText(QtGui.QApplication.translate("Dialog", "choose path", None, QtGui.QApplication.UnicodeUTF8))
        self.i_file_name.setText(QtGui.QApplication.translate("Dialog", "mat_name.mat", None, QtGui.QApplication.UnicodeUTF8))
        self.b_save_file.setText(QtGui.QApplication.translate("Dialog", "save", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", ".mat", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_2.setTitle(QtGui.QApplication.translate("Dialog", "Controls", None, QtGui.QApplication.UnicodeUTF8))
        self.b_play.setText(QtGui.QApplication.translate("Dialog", "play", None, QtGui.QApplication.UnicodeUTF8))
        self.b_stap_by_step.setText(QtGui.QApplication.translate("Dialog", "step by step", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "frame:", None, QtGui.QApplication.UnicodeUTF8))
        self.l_frame.setText(QtGui.QApplication.translate("Dialog", "1", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_3.setTitle(QtGui.QApplication.translate("Dialog", "Display", None, QtGui.QApplication.UnicodeUTF8))
        self.ch_mser_collection.setText(QtGui.QApplication.translate("Dialog", "msers collection", None, QtGui.QApplication.UnicodeUTF8))
        self.ch_ants_collection.setText(QtGui.QApplication.translate("Dialog", "ants collection", None, QtGui.QApplication.UnicodeUTF8))

