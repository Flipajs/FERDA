# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/landing_tab.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
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

class Ui_landingForm(object):
    def setupUi(self, landingForm):
        landingForm.setObjectName(_fromUtf8("landingForm"))
        landingForm.resize(400, 300)
        self.horizontalLayout = QtGui.QHBoxLayout(landingForm)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.newProjectButton = QtGui.QPushButton(landingForm)
        self.newProjectButton.setObjectName(_fromUtf8("newProjectButton"))
        self.verticalLayout.addWidget(self.newProjectButton)
        spacerItem = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        self.verticalLayout.addItem(spacerItem)
        self.loadProjectButton = QtGui.QPushButton(landingForm)
        self.loadProjectButton.setObjectName(_fromUtf8("loadProjectButton"))
        self.verticalLayout.addWidget(self.loadProjectButton)
        spacerItem1 = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        self.verticalLayout.addItem(spacerItem1)
        self.settingsButton = QtGui.QPushButton(landingForm)
        self.settingsButton.setObjectName(_fromUtf8("settingsButton"))
        self.verticalLayout.addWidget(self.settingsButton)
        spacerItem2 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.label = QtGui.QLabel(landingForm)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_2.addWidget(self.label)
        self.recentProjectsList = QtGui.QListWidget(landingForm)
        self.recentProjectsList.setObjectName(_fromUtf8("recentProjectsList"))
        self.verticalLayout_2.addWidget(self.recentProjectsList)
        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.retranslateUi(landingForm)
        QtCore.QMetaObject.connectSlotsByName(landingForm)

    def retranslateUi(self, landingForm):
        landingForm.setWindowTitle(_translate("landingForm", "Form", None))
        self.newProjectButton.setText(_translate("landingForm", "New Project", None))
        self.loadProjectButton.setText(_translate("landingForm", "Load Project", None))
        self.settingsButton.setText(_translate("landingForm", "Settings", None))
        self.label.setText(_translate("landingForm", "Recent Projects:", None))

