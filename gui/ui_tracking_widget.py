# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/tracking_widget.ui'
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

class Ui_tracking_widget(object):
    def setupUi(self, tracking_widget):
        tracking_widget.setObjectName(_fromUtf8("tracking_widget"))
        tracking_widget.resize(582, 396)
        tracking_widget.setMinimumSize(QtCore.QSize(582, 298))
        self.verticalLayout_2 = QtGui.QVBoxLayout(tracking_widget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.frame = QtGui.QFrame(tracking_widget)
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.verticalLayout = QtGui.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label_6 = QtGui.QLabel(self.frame)
        self.label_6.setFrameShape(QtGui.QFrame.StyledPanel)
        self.label_6.setTextFormat(QtCore.Qt.RichText)
        self.label_6.setWordWrap(True)
        self.label_6.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.verticalLayout.addWidget(self.label_6)
        self.verticalLayout_2.addWidget(self.frame)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_4 = QtGui.QLabel(tracking_widget)
        self.label_4.setEnabled(False)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.pbar_regions_classification = QtGui.QProgressBar(tracking_widget)
        self.pbar_regions_classification.setEnabled(False)
        self.pbar_regions_classification.setProperty("value", 0)
        self.pbar_regions_classification.setObjectName(_fromUtf8("pbar_regions_classification"))
        self.gridLayout.addWidget(self.pbar_regions_classification, 2, 1, 1, 1)
        self.label = QtGui.QLabel(tracking_widget)
        self.label.setEnabled(False)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtGui.QLabel(tracking_widget)
        self.label_2.setEnabled(False)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.pbar_graph = QtGui.QProgressBar(tracking_widget)
        self.pbar_graph.setEnabled(False)
        self.pbar_graph.setProperty("value", 0)
        self.pbar_graph.setObjectName(_fromUtf8("pbar_graph"))
        self.gridLayout.addWidget(self.pbar_graph, 1, 1, 1, 1)
        self.pbar_segmentation = QtGui.QProgressBar(tracking_widget)
        self.pbar_segmentation.setEnabled(False)
        self.pbar_segmentation.setAutoFillBackground(False)
        self.pbar_segmentation.setProperty("value", 0)
        self.pbar_segmentation.setObjectName(_fromUtf8("pbar_segmentation"))
        self.gridLayout.addWidget(self.pbar_segmentation, 0, 1, 1, 1)
        self.pbar_descriptors = QtGui.QProgressBar(tracking_widget)
        self.pbar_descriptors.setEnabled(False)
        self.pbar_descriptors.setProperty("value", 0)
        self.pbar_descriptors.setObjectName(_fromUtf8("pbar_descriptors"))
        self.gridLayout.addWidget(self.pbar_descriptors, 3, 1, 1, 1)
        self.label_3 = QtGui.QLabel(tracking_widget)
        self.label_3.setEnabled(False)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_5 = QtGui.QLabel(tracking_widget)
        self.label_5.setEnabled(False)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.pbar_reid = QtGui.QProgressBar(tracking_widget)
        self.pbar_reid.setEnabled(False)
        self.pbar_reid.setProperty("value", 0)
        self.pbar_reid.setObjectName(_fromUtf8("pbar_reid"))
        self.gridLayout.addWidget(self.pbar_reid, 4, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.start_tracking_button = QtGui.QPushButton(tracking_widget)
        self.start_tracking_button.setEnabled(False)
        self.start_tracking_button.setObjectName(_fromUtf8("start_tracking_button"))
        self.verticalLayout_2.addWidget(self.start_tracking_button)

        self.retranslateUi(tracking_widget)
        QtCore.QMetaObject.connectSlotsByName(tracking_widget)

    def retranslateUi(self, tracking_widget):
        tracking_widget.setWindowTitle(_translate("tracking_widget", "Form", None))
        self.label_6.setText(_translate("tracking_widget", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Cantarell\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Performing tracking in the GUI is not supported yet. Please exit the application and use the command line tool <span style=\" font-family:\'Courier New,courier\';\">ferda_cli.py</span>. When the tracking is completed, open the project in the application again.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Example:</p>\n"
"<pre style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Courier New,courier\';\">$ python ferda_cli.py ../projects/project1/ --run-tracking --reidentification-weights ../weights/ants.h5</span></pre>\n"
"<pre style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Courier New,courier\';\"><br /></pre></body></html>", None))
        self.label_4.setText(_translate("tracking_widget", "Re-identification Descriptors", None))
        self.label.setText(_translate("tracking_widget", "Segmentation", None))
        self.label_2.setText(_translate("tracking_widget", "Graph Construction", None))
        self.label_3.setText(_translate("tracking_widget", "Regions Classification", None))
        self.label_5.setText(_translate("tracking_widget", "Re-identification", None))
        self.start_tracking_button.setText(_translate("tracking_widget", "Start Tracking", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    tracking_widget = QtGui.QWidget()
    ui = Ui_tracking_widget()
    ui.setupUi(tracking_widget)
    tracking_widget.show()
    sys.exit(app.exec_())

