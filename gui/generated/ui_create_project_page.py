# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/create_project_page.ui'
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

class Ui_createProjectPage(object):
    def setupUi(self, createProjectPage):
        createProjectPage.setObjectName(_fromUtf8("createProjectPage"))
        createProjectPage.resize(559, 594)
        createProjectPage.setSubTitle(_fromUtf8(""))
        self.verticalLayout_2 = QtGui.QVBoxLayout(createProjectPage)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.label = QtGui.QLabel(createProjectPage)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_2.addWidget(self.label)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.videoFileEdit = QtGui.QLineEdit(createProjectPage)
        self.videoFileEdit.setObjectName(_fromUtf8("videoFileEdit"))
        self.horizontalLayout.addWidget(self.videoFileEdit)
        self.videoFileButton = QtGui.QToolButton(createProjectPage)
        self.videoFileButton.setObjectName(_fromUtf8("videoFileButton"))
        self.horizontalLayout.addWidget(self.videoFileButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.videoFileWarning = QtGui.QLabel(createProjectPage)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.videoFileWarning.setFont(font)
        self.videoFileWarning.setFrameShape(QtGui.QFrame.NoFrame)
        self.videoFileWarning.setText(_fromUtf8(""))
        self.videoFileWarning.setTextFormat(QtCore.Qt.AutoText)
        self.videoFileWarning.setIndent(6)
        self.videoFileWarning.setObjectName(_fromUtf8("videoFileWarning"))
        self.verticalLayout_2.addWidget(self.videoFileWarning)
        self.label_2 = QtGui.QLabel(createProjectPage)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout_2.addWidget(self.label_2)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.projectFolderEdit = QtGui.QLineEdit(createProjectPage)
        self.projectFolderEdit.setObjectName(_fromUtf8("projectFolderEdit"))
        self.horizontalLayout_2.addWidget(self.projectFolderEdit)
        self.projectFolderButton = QtGui.QToolButton(createProjectPage)
        self.projectFolderButton.setObjectName(_fromUtf8("projectFolderButton"))
        self.horizontalLayout_2.addWidget(self.projectFolderButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.projectFolderWarning = QtGui.QLabel(createProjectPage)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.projectFolderWarning.setFont(font)
        self.projectFolderWarning.setText(_fromUtf8(""))
        self.projectFolderWarning.setIndent(6)
        self.projectFolderWarning.setObjectName(_fromUtf8("projectFolderWarning"))
        self.verticalLayout_2.addWidget(self.projectFolderWarning)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_3 = QtGui.QLabel(createProjectPage)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_4.addWidget(self.label_3)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.label_4 = QtGui.QLabel(createProjectPage)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout_4.addWidget(self.label_4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.projectNameEdit = QtGui.QLineEdit(createProjectPage)
        self.projectNameEdit.setObjectName(_fromUtf8("projectNameEdit"))
        self.verticalLayout_2.addWidget(self.projectNameEdit)
        spacerItem1 = QtGui.QSpacerItem(361, 7, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        self.verticalLayout_2.addItem(spacerItem1)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.label_5 = QtGui.QLabel(createProjectPage)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.horizontalLayout_5.addWidget(self.label_5)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.label_6 = QtGui.QLabel(createProjectPage)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.horizontalLayout_5.addWidget(self.label_6)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.projectDescriptionEdit = QtGui.QPlainTextEdit(createProjectPage)
        self.projectDescriptionEdit.setObjectName(_fromUtf8("projectDescriptionEdit"))
        self.verticalLayout_2.addWidget(self.projectDescriptionEdit)
        spacerItem3 = QtGui.QSpacerItem(20, 99, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
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
    app = QtGui.QApplication(sys.argv)
    createProjectPage = QtGui.QWizardPage()
    ui = Ui_createProjectPage()
    ui.setupUi(createProjectPage)
    createProjectPage.show()
    sys.exit(app.exec_())

