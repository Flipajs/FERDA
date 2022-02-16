# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/main_tab.ui'
#
# Created by: PyQt4 UI code generator 4.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt6 import QtCore, QtGui, QtWidgets

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtCore.QCoreApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtCore.QCoreApplication.translate(context, text, disambig)

class Ui_MainWidget(object):
    def setupUi(self, MainWidget):
        MainWidget.setObjectName("MainWidget")
        MainWidget.resize(539, 449)
        self.horizontalLayout = QtWidgets.QHBoxLayout(MainWidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.newProjectButton = QtWidgets.QPushButton(MainWidget)
        self.newProjectButton.setObjectName("newProjectButton")
        self.verticalLayout.addWidget(self.newProjectButton)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        self.verticalLayout.addItem(spacerItem)
        self.loadProjectButton = QtWidgets.QPushButton(MainWidget)
        self.loadProjectButton.setObjectName("loadProjectButton")
        self.verticalLayout.addWidget(self.loadProjectButton)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        self.verticalLayout.addItem(spacerItem1)
        self.settingsButton = QtWidgets.QPushButton(MainWidget)
        self.settingsButton.setObjectName("settingsButton")
        self.verticalLayout.addWidget(self.settingsButton)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        spacerItem3 = QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(MainWidget)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.recentProjectsListWidget = QtWidgets.QListWidget(MainWidget)
        self.recentProjectsListWidget.setObjectName("recentProjectsListWidget")
        self.verticalLayout_2.addWidget(self.recentProjectsListWidget)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.label.setBuddy(self.recentProjectsListWidget)

        self.retranslateUi(MainWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWidget)
        MainWidget.setTabOrder(self.newProjectButton, self.loadProjectButton)
        MainWidget.setTabOrder(self.loadProjectButton, self.settingsButton)
        MainWidget.setTabOrder(self.settingsButton, self.recentProjectsListWidget)

    def retranslateUi(self, MainWidget):
        MainWidget.setWindowTitle(_translate("MainWidget", "Form", None))
        self.newProjectButton.setText(_translate("MainWidget", "New Project", None))
        self.loadProjectButton.setText(_translate("MainWidget", "Load Project", None))
        self.settingsButton.setText(_translate("MainWidget", "Settings", None))
        self.label.setText(_translate("MainWidget", "Recen&t Projects:", None))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWidget = QtWidgets.QWidget()
    ui = Ui_MainWidget()
    ui.setupUi(MainWidget)
    MainWidget.show()
    sys.exit(app.exec())

