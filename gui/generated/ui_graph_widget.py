# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/graph_widget.ui'
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

class Ui_graph_widget(object):
    def setupUi(self, graph_widget):
        graph_widget.setObjectName(_fromUtf8("graph_widget"))
        graph_widget.resize(421, 319)
        self.verticalLayout = QtGui.QVBoxLayout(graph_widget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.verticalLayoutSetupGroup = QtGui.QVBoxLayout()
        self.verticalLayoutSetupGroup.setObjectName(_fromUtf8("verticalLayoutSetupGroup"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.groupBox = QtGui.QGroupBox(graph_widget)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label = QtGui.QLabel(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_2.addWidget(self.label)
        self.spinBoxFrom = QtGui.QSpinBox(self.groupBox)
        self.spinBoxFrom.setPrefix(_fromUtf8(""))
        self.spinBoxFrom.setObjectName(_fromUtf8("spinBoxFrom"))
        self.horizontalLayout_2.addWidget(self.spinBoxFrom)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.label_2 = QtGui.QLabel(self.groupBox)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_2.addWidget(self.label_2)
        self.spinBoxTo = QtGui.QSpinBox(self.groupBox)
        self.spinBoxTo.setProperty("showGroupSeparator", False)
        self.spinBoxTo.setObjectName(_fromUtf8("spinBoxTo"))
        self.horizontalLayout_2.addWidget(self.spinBoxTo)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.pushButtonDrawGraph = QtGui.QPushButton(self.groupBox)
        self.pushButtonDrawGraph.setObjectName(_fromUtf8("pushButtonDrawGraph"))
        self.horizontalLayout_2.addWidget(self.pushButtonDrawGraph)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addWidget(self.groupBox)
        self.verticalLayoutSetupGroup.addLayout(self.horizontalLayout)
        self.verticalLayout.addLayout(self.verticalLayoutSetupGroup)
        spacerItem2 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.graph_visualizer = QtGui.QWidget(graph_widget)
        self.graph_visualizer.setObjectName(_fromUtf8("graph_visualizer"))
        self.verticalLayout.addWidget(self.graph_visualizer)
        self.label.setBuddy(self.spinBoxFrom)
        self.label_2.setBuddy(self.spinBoxTo)

        self.retranslateUi(graph_widget)
        QtCore.QMetaObject.connectSlotsByName(graph_widget)
        graph_widget.setTabOrder(self.spinBoxFrom, self.spinBoxTo)

    def retranslateUi(self, graph_widget):
        graph_widget.setWindowTitle(_translate("graph_widget", "Form", None))
        self.groupBox.setTitle(_translate("graph_widget", "Tracking Graph Range", None))
        self.label.setText(_translate("graph_widget", "From", None))
        self.label_2.setText(_translate("graph_widget", "&To", None))
        self.pushButtonDrawGraph.setText(_translate("graph_widget", "Draw Graph", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    graph_widget = QtGui.QWidget()
    ui = Ui_graph_widget()
    ui.setupUi(graph_widget)
    graph_widget.show()
    sys.exit(app.exec_())

