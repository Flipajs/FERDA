# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/main_tab.ui'
#
# Created by: PyQt4 UI code generator 4.12.1
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

class Ui_MainWidget(object):
    def setupUi(self, MainWidget):
        MainWidget.setObjectName(_fromUtf8("MainWidget"))
        MainWidget.resize(539, 449)
        self.horizontalLayout = QtGui.QHBoxLayout(MainWidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.newProjectButton = QtGui.QPushButton(MainWidget)
        self.newProjectButton.setObjectName(_fromUtf8("newProjectButton"))
        self.verticalLayout.addWidget(self.newProjectButton)
        spacerItem = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        self.verticalLayout.addItem(spacerItem)
        self.loadProjectButton = QtGui.QPushButton(MainWidget)
        self.loadProjectButton.setObjectName(_fromUtf8("loadProjectButton"))
        self.verticalLayout.addWidget(self.loadProjectButton)
        spacerItem1 = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        self.verticalLayout.addItem(spacerItem1)
        self.settingsButton = QtGui.QPushButton(MainWidget)
        self.settingsButton.setObjectName(_fromUtf8("settingsButton"))
        self.verticalLayout.addWidget(self.settingsButton)
        spacerItem2 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        spacerItem3 = QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.label = QtGui.QLabel(MainWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_2.addWidget(self.label)
        self.recentProjectsListWidget = QtGui.QListWidget(MainWidget)
        self.recentProjectsListWidget.setObjectName(_fromUtf8("recentProjectsListWidget"))
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
    app = QtGui.QApplication(sys.argv)
    MainWidget = QtGui.QWidget()
    ui = Ui_MainWidget()
    ui.setupUi(MainWidget)
    MainWidget.show()
    sys.exit(app.exec_())

