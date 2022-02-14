# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/graph_widget.ui'
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

class Ui_graph_widget(object):
    def setupUi(self, graph_widget):
        graph_widget.setObjectName("graph_widget")
        graph_widget.resize(421, 319)
        self.verticalLayout = QtWidgets.QVBoxLayout(graph_widget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayoutSetupGroup = QtWidgets.QVBoxLayout()
        self.verticalLayoutSetupGroup.setObjectName("verticalLayoutSetupGroup")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(graph_widget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.spinBoxFrom = QtWidgets.QSpinBox(self.groupBox)
        self.spinBoxFrom.setPrefix("")
        self.spinBoxFrom.setObjectName("spinBoxFrom")
        self.horizontalLayout_2.addWidget(self.spinBoxFrom)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.spinBoxTo = QtWidgets.QSpinBox(self.groupBox)
        self.spinBoxTo.setProperty("showGroupSeparator", False)
        self.spinBoxTo.setObjectName("spinBoxTo")
        self.horizontalLayout_2.addWidget(self.spinBoxTo)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.pushButtonDrawGraph = QtWidgets.QPushButton(self.groupBox)
        self.pushButtonDrawGraph.setObjectName("pushButtonDrawGraph")
        self.horizontalLayout_2.addWidget(self.pushButtonDrawGraph)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addWidget(self.groupBox)
        self.verticalLayoutSetupGroup.addLayout(self.horizontalLayout)
        self.verticalLayout.addLayout(self.verticalLayoutSetupGroup)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.graph_visualizer = QtWidgets.QWidget(graph_widget)
        self.graph_visualizer.setObjectName("graph_visualizer")
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
    app = QtWidgets.QApplication(sys.argv)
    graph_widget = QtWidgets.QWidget()
    ui = Ui_graph_widget()
    ui.setupUi(graph_widget)
    graph_widget.show()
    sys.exit(app.exec_())

