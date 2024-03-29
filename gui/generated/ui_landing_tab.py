# Form implementation generated from reading ui file 'gui/landing_tab.ui'
#
# Created by: PyQt6 UI code generator 6.2.3
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


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
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        self.verticalLayout.addItem(spacerItem)
        self.loadProjectButton = QtWidgets.QPushButton(landingForm)
        self.loadProjectButton.setObjectName("loadProjectButton")
        self.verticalLayout.addWidget(self.loadProjectButton)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        self.verticalLayout.addItem(spacerItem1)
        self.settingsButton = QtWidgets.QPushButton(landingForm)
        self.settingsButton.setObjectName("settingsButton")
        self.verticalLayout.addWidget(self.settingsButton)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
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
        _translate = QtCore.QCoreApplication.translate
        landingForm.setWindowTitle(_translate("landingForm", "Form"))
        self.newProjectButton.setText(_translate("landingForm", "New Project"))
        self.loadProjectButton.setText(_translate("landingForm", "Load Project"))
        self.settingsButton.setText(_translate("landingForm", "Settings"))
        self.label.setText(_translate("landingForm", "Recent Projects:"))
