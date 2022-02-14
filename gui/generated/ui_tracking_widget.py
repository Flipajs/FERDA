# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/tracking_widget.ui'
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

class Ui_tracking_widget(object):
    def setupUi(self, tracking_widget):
        tracking_widget.setObjectName("tracking_widget")
        tracking_widget.resize(582, 396)
        tracking_widget.setMinimumSize(QtCore.QSize(582, 298))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(tracking_widget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame = QtWidgets.QFrame(tracking_widget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_6 = QtWidgets.QLabel(self.frame)
        self.label_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_6.setTextFormat(QtCore.Qt.RichText)
        self.label_6.setWordWrap(True)
        self.label_6.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.verticalLayout_2.addWidget(self.frame)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_4 = QtWidgets.QLabel(tracking_widget)
        self.label_4.setEnabled(False)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.pbar_regions_classification = QtWidgets.QProgressBar(tracking_widget)
        self.pbar_regions_classification.setEnabled(False)
        self.pbar_regions_classification.setProperty("value", 0)
        self.pbar_regions_classification.setObjectName("pbar_regions_classification")
        self.gridLayout.addWidget(self.pbar_regions_classification, 2, 1, 1, 1)
        self.label = QtWidgets.QLabel(tracking_widget)
        self.label.setEnabled(False)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(tracking_widget)
        self.label_2.setEnabled(False)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.pbar_graph = QtWidgets.QProgressBar(tracking_widget)
        self.pbar_graph.setEnabled(False)
        self.pbar_graph.setProperty("value", 0)
        self.pbar_graph.setObjectName("pbar_graph")
        self.gridLayout.addWidget(self.pbar_graph, 1, 1, 1, 1)
        self.pbar_segmentation = QtWidgets.QProgressBar(tracking_widget)
        self.pbar_segmentation.setEnabled(False)
        self.pbar_segmentation.setAutoFillBackground(False)
        self.pbar_segmentation.setProperty("value", 0)
        self.pbar_segmentation.setObjectName("pbar_segmentation")
        self.gridLayout.addWidget(self.pbar_segmentation, 0, 1, 1, 1)
        self.pbar_descriptors = QtWidgets.QProgressBar(tracking_widget)
        self.pbar_descriptors.setEnabled(False)
        self.pbar_descriptors.setProperty("value", 0)
        self.pbar_descriptors.setObjectName("pbar_descriptors")
        self.gridLayout.addWidget(self.pbar_descriptors, 3, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(tracking_widget)
        self.label_3.setEnabled(False)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(tracking_widget)
        self.label_5.setEnabled(False)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.pbar_reid = QtWidgets.QProgressBar(tracking_widget)
        self.pbar_reid.setEnabled(False)
        self.pbar_reid.setProperty("value", 0)
        self.pbar_reid.setObjectName("pbar_reid")
        self.gridLayout.addWidget(self.pbar_reid, 4, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.start_tracking_button = QtWidgets.QPushButton(tracking_widget)
        self.start_tracking_button.setEnabled(False)
        self.start_tracking_button.setObjectName("start_tracking_button")
        self.verticalLayout_2.addWidget(self.start_tracking_button)

        self.retranslateUi(tracking_widget)
        QtCore.QMetaObject.connectSlotsByName(tracking_widget)

    def retranslateUi(self, tracking_widget):
        tracking_widget.setWindowTitle(_translate("tracking_widget", "Form", None))
        self.label_6.setText(_translate("tracking_widget", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Cantarell\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Project processing in the GUI is not supported yet. Please exit the application and use the command line tool <span style=\" font-family:\'Courier New,courier\';\">ferda_cli.py</span>. When the tracking is completed, open the project in the application again.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Example:</p>\n"
"<p style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Courier New,courier\';\">$ python ferda_cli.py ../projects/project1/ --run-tracking --reidentification-weights ../weights/ants.h5</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Courier New,courier\';\"><br /></p></body></html>", None))
        self.label_4.setText(_translate("tracking_widget", "Re-identification Descriptors", None))
        self.label.setText(_translate("tracking_widget", "Segmentation", None))
        self.label_2.setText(_translate("tracking_widget", "Graph Construction", None))
        self.label_3.setText(_translate("tracking_widget", "Regions Classification", None))
        self.label_5.setText(_translate("tracking_widget", "Re-identification", None))
        self.start_tracking_button.setText(_translate("tracking_widget", "Start Tracking", None))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    tracking_widget = QtWidgets.QWidget()
    ui = Ui_tracking_widget()
    ui.setupUi(tracking_widget)
    tracking_widget.show()
    sys.exit(app.exec_())

