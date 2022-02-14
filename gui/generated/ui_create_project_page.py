# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/create_project_page.ui'
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

class Ui_createProjectPage(object):
    def setupUi(self, createProjectPage):
        createProjectPage.setObjectName("createProjectPage")
        createProjectPage.resize(559, 594)
        createProjectPage.setSubTitle("")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(createProjectPage)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(createProjectPage)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.videoFileEdit = QtWidgets.QLineEdit(createProjectPage)
        self.videoFileEdit.setObjectName("videoFileEdit")
        self.horizontalLayout.addWidget(self.videoFileEdit)
        self.videoFileButton = QtWidgets.QToolButton(createProjectPage)
        self.videoFileButton.setObjectName("videoFileButton")
        self.horizontalLayout.addWidget(self.videoFileButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.videoFileWarning = QtWidgets.QLabel(createProjectPage)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.videoFileWarning.setFont(font)
        self.videoFileWarning.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.videoFileWarning.setText("")
        self.videoFileWarning.setTextFormat(QtCore.Qt.AutoText)
        self.videoFileWarning.setIndent(6)
        self.videoFileWarning.setObjectName("videoFileWarning")
        self.verticalLayout_2.addWidget(self.videoFileWarning)
        self.label_2 = QtWidgets.QLabel(createProjectPage)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.projectFolderEdit = QtWidgets.QLineEdit(createProjectPage)
        self.projectFolderEdit.setObjectName("projectFolderEdit")
        self.horizontalLayout_2.addWidget(self.projectFolderEdit)
        self.projectFolderButton = QtWidgets.QToolButton(createProjectPage)
        self.projectFolderButton.setObjectName("projectFolderButton")
        self.horizontalLayout_2.addWidget(self.projectFolderButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.projectFolderWarning = QtWidgets.QLabel(createProjectPage)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.projectFolderWarning.setFont(font)
        self.projectFolderWarning.setText("")
        self.projectFolderWarning.setIndent(6)
        self.projectFolderWarning.setObjectName("projectFolderWarning")
        self.verticalLayout_2.addWidget(self.projectFolderWarning)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_3 = QtWidgets.QLabel(createProjectPage)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_4.addWidget(self.label_3)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.label_4 = QtWidgets.QLabel(createProjectPage)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.projectNameEdit = QtWidgets.QLineEdit(createProjectPage)
        self.projectNameEdit.setObjectName("projectNameEdit")
        self.verticalLayout_2.addWidget(self.projectNameEdit)
        spacerItem1 = QtWidgets.QSpacerItem(361, 7, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_2.addItem(spacerItem1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(createProjectPage)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.label_6 = QtWidgets.QLabel(createProjectPage)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_5.addWidget(self.label_6)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.projectDescriptionEdit = QtWidgets.QPlainTextEdit(createProjectPage)
        self.projectDescriptionEdit.setObjectName("projectDescriptionEdit")
        self.verticalLayout_2.addWidget(self.projectDescriptionEdit)
        spacerItem3 = QtWidgets.QSpacerItem(20, 99, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem3)
        self.label_2.setBuddy(self.projectFolderEdit)
        self.label_3.setBuddy(self.projectNameEdit)
        self.label_5.setBuddy(self.projectDescriptionEdit)

        self.retranslateUi(createProjectPage)
        QtCore.QMetaObject.connectSlotsByName(createProjectPage)

    def retranslateUi(self, createProjectPage):
        createProjectPage.setTitle(_translate("createProjectPage", "New Project Wi&zard", None))
        self.label.setText(_translate("createProjectPage", "Video File:", None))
        self.videoFileButton.setText(_translate("createProjectPage", "...", None))
        self.label_2.setText(_translate("createProjectPage", "Pro&ject Folder:", None))
        self.projectFolderButton.setText(_translate("createProjectPage", "...", None))
        self.label_3.setText(_translate("createProjectPage", "Projec&t Name:", None))
        self.label_4.setText(_translate("createProjectPage", "<html><head/><body><p><span style=\" font-size:8pt; color:#babdb6;\">Optional</span></p></body></html>", None))
        self.label_5.setText(_translate("createProjectPage", "Project Descriptio&n:", None))
        self.label_6.setText(_translate("createProjectPage", "<html><head/><body><p><span style=\" font-size:8pt; color:#babdb6;\">Optional</span></p></body></html>", None))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    createProjectPage = QtWidgets.QWizardPage()
    ui = Ui_createProjectPage()
    ui.setupUi(createProjectPage)
    createProjectPage.show()
    sys.exit(app.exec_())

