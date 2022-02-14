# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/landing_tab.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtCore.QCoreApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtCore.QCoreApplication.translate(context, text, disambig)

class Ui_landingForm(object):
    def setupUi(self, landingForm):
        landingForm.setObjectName("landingForm")
        landingForm.resize(400, 300)
        self.horizontalLayout = QtWidgets.QHBoxLayout(landingForm)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.newProjectButton = QtWidgets.QPushButton(landingForm)
        self.newProjectButton.setObjectName("newProjectButton")
        self.verticalLayout.addWidget(self.newProjectButton)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout.addItem(spacerItem)
        self.loadProjectButton = QtWidgets.QPushButton(landingForm)
        self.loadProjectButton.setObjectName("loadProjectButton")
        self.verticalLayout.addWidget(self.loadProjectButton)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout.addItem(spacerItem1)
        self.settingsButton = QtWidgets.QPushButton(landingForm)
        self.settingsButton.setObjectName("settingsButton")
        self.verticalLayout.addWidget(self.settingsButton)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(landingForm)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.recentProjectsList = QtWidgets.QListWidget(landingForm)
        self.recentProjectsList.setObjectName("recentProjectsList")
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

