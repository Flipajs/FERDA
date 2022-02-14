# Form implementation generated from reading ui file 'gui/graph_widget.ui'
#
# Created by: PyQt6 UI code generator 6.2.3
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.spinBoxFrom = QtWidgets.QSpinBox(self.groupBox)
        self.spinBoxFrom.setPrefix("")
        self.spinBoxFrom.setObjectName("spinBoxFrom")
        self.horizontalLayout_2.addWidget(self.spinBoxFrom)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.spinBoxTo = QtWidgets.QSpinBox(self.groupBox)
        self.spinBoxTo.setProperty("showGroupSeparator", False)
        self.spinBoxTo.setObjectName("spinBoxTo")
        self.horizontalLayout_2.addWidget(self.spinBoxTo)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.pushButtonDrawGraph = QtWidgets.QPushButton(self.groupBox)
        self.pushButtonDrawGraph.setObjectName("pushButtonDrawGraph")
        self.horizontalLayout_2.addWidget(self.pushButtonDrawGraph)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addWidget(self.groupBox)
        self.verticalLayoutSetupGroup.addLayout(self.horizontalLayout)
        self.verticalLayout.addLayout(self.verticalLayoutSetupGroup)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
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
        _translate = QtCore.QCoreApplication.translate
        graph_widget.setWindowTitle(_translate("graph_widget", "Form"))
        self.groupBox.setTitle(_translate("graph_widget", "Tracking Graph Range"))
        self.label.setText(_translate("graph_widget", "From"))
        self.label_2.setText(_translate("graph_widget", "&To"))
        self.pushButtonDrawGraph.setText(_translate("graph_widget", "Draw Graph"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    graph_widget = QtWidgets.QWidget()
    ui = Ui_graph_widget()
    ui.setupUi(graph_widget)
    graph_widget.show()
    sys.exit(app.exec())
