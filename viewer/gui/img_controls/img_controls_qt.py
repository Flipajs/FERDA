# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'img_controls.ui'
#
# Created: Mon Jul 14 10:52:04 2014
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

from viewer.gui.img_controls.components import VideoSlider

class SelectAllLineEdit(QtGui.QLineEdit):
    def __init__(self, parent=None):
        super(SelectAllLineEdit, self).__init__(parent)
        self.readyToEdit = True

    def mousePressEvent(self, e, Parent=None):
        super(SelectAllLineEdit, self).mousePressEvent(e) #required to deselect on 2e click
        if self.readyToEdit:
            self.selectAll()
            self.readyToEdit = False

    def focusOutEvent(self, e):
        super(SelectAllLineEdit, self).focusOutEvent(e) #required to remove cursor on focusOut
        self.deselect()
        self.readyToEdit = True

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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1030, 770)

        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))

        self.splitter2 = QtGui.QSplitter(QtCore.Qt.Vertical)

        self.lines_layout = QtGui.QVBoxLayout()
        self.lines_layout.setSpacing(0)
        self.lines_layout.setMargin(0)
        self.lines_layout.setContentsMargins(0, 0, 0, 0)

        self.main_line_layout = QtGui.QHBoxLayout()
        self.main_line_layout.setSpacing(0)
        self.main_line_layout.setMargin(0)
        self.main_line_layout.setContentsMargins(0, 0, 0, 0)
        self.main_line_widget = QtGui.QWidget()
        # self.main_line_widget.setMaximumHeight(500)
        # self.main_line_widget.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.main_line_widget.setLayout(self.main_line_layout)

        self.centralwidget.setLayout(self.lines_layout)

        self.splitter2.addWidget(self.main_line_widget)

        # self.lines_layout.addWidget(self.main_line_widget)


        self.bottom_line_widget = QtGui.QWidget()
        self.bottom_line_layout = QtGui.QVBoxLayout()
        self.bottom_line_widget.setLayout(self.bottom_line_layout)

        self.video_widget = QtGui.QWidget()
        self.video_layout = QtGui.QVBoxLayout()
        self.video_widget.setLayout(self.video_layout)

        self.video_control_widget = QtGui.QWidget()
        self.video_control_layout = QtGui.QVBoxLayout()
        self.video_control_widget.setLayout(self.video_control_layout)
        self.video_control_widget.setMaximumHeight(70)

        self.video_control_buttons_widget = QtGui.QWidget()
        self.video_control_buttons_layout = QtGui.QHBoxLayout()
        self.video_control_buttons_widget.setLayout(self.video_control_buttons_layout)

        # self.lines_layout.addWidget(self.bottom_line_widget)
        self.splitter2.addWidget(self.bottom_line_widget)
        self.lines_layout.addWidget(self.splitter2)

        self.menu_panel_widget = QtGui.QWidget()
        self.main_line_layout.addWidget(self.menu_panel_widget)
        self.menu_panel_layout = QtGui.QVBoxLayout()
        self.menu_panel_widget.setLayout(self.menu_panel_layout)

        self.informationLabel = QtGui.QLabel(self.centralwidget)
        self.informationLabel.setAlignment(QtCore.Qt.AlignTop)
        self.informationLabel.setAlignment(QtCore.Qt.AlignHCenter)
        self.informationLabel.setObjectName(_fromUtf8("informationLabel"))

        self.openData = QtGui.QPushButton(self.centralwidget)
        self.openData.setObjectName(_fromUtf8("open data"))

        self.openVideo = QtGui.QPushButton(self.centralwidget)
        self.openVideo.setObjectName(_fromUtf8("open video"))

        self.saveData = QtGui.QPushButton(self.centralwidget)
        self.saveData.setObjectName(_fromUtf8("save data"))

        self.loadChanges = QtGui.QPushButton(self.centralwidget)
        self.loadChanges.setObjectName(_fromUtf8("load changes"))

        self.saveChangesToFile = QtGui.QPushButton(self.centralwidget)
        self.saveChangesToFile.setObjectName(_fromUtf8("save changes"))

        self.undoChange = QtGui.QPushButton(self.centralwidget)
        self.undoChange.setObjectName(_fromUtf8("undo change"))

        self.redoChange = QtGui.QPushButton(self.centralwidget)
        self.redoChange.setObjectName(_fromUtf8("redo change"))

        self.showHistory = QtGui.QPushButton(self.centralwidget)
        self.showHistory.setObjectName(_fromUtf8("show history"))

        self.swapAnts = QtGui.QPushButton(self.centralwidget)
        self.swapAnts.setObjectName(_fromUtf8("swap ants"))

        self.swapTailHead = QtGui.QPushButton(self.centralwidget)
        self.swapTailHead.setObjectName(_fromUtf8("swap tail and head"))

        self.showFaults = QtGui.QPushButton(self.centralwidget)
        self.showFaults.setObjectName(_fromUtf8("show faults"))

        self.nextFault = QtGui.QPushButton(self.centralwidget)
        self.nextFault.setObjectName(_fromUtf8("next fault"))

        self.previousFault = QtGui.QPushButton(self.centralwidget)
        self.previousFault.setObjectName(_fromUtf8("previous fault"))

        self.toggleHighlight = QtGui.QPushButton(self.centralwidget)
        self.toggleHighlight.setObjectName(_fromUtf8("toggle highlight"))

        self.faultNumLabel = QtGui.QLabel(self.centralwidget)
        self.faultNumLabel.setObjectName(_fromUtf8("faultNumLabel"))

        self.faultLabel = QtGui.QLabel(self.centralwidget)
        self.faultLabel.setObjectName(_fromUtf8("faultLabel"))

        self.cancelButton = QtGui.QPushButton(self.centralwidget)
        self.cancelButton.setObjectName(_fromUtf8("cancel correction"))

        self.settingsButton = QtGui.QPushButton(self.centralwidget)
        self.settingsButton.setObjectName(_fromUtf8("settings"))

        self.sequenceButton = QtGui.QPushButton(self.centralwidget)
        self.sequenceButton.setObjectName(_fromUtf8("sequence"))

        self.showMSERsButton = QtGui.QPushButton(self.centralwidget)
        self.showMSERsButton.setObjectName(_fromUtf8("show msers"))

        self.showCertaintyButton = QtGui.QPushButton(self.centralwidget)
        self.showCertaintyButton.setObjectName(_fromUtf8("show certainty"))

        self.speedSlider = QtGui.QSlider(self.centralwidget)
        self.speedSlider.setObjectName(_fromUtf8("speedSlider"))
        self.speedSlider.setOrientation(QtCore.Qt.Horizontal)

        self.backward = QtGui.QPushButton(self.centralwidget)
        self.backward.setObjectName(_fromUtf8("backward"))

        self.playPause = QtGui.QPushButton(self.centralwidget)
        self.playPause.setObjectName(_fromUtf8("play pause"))

        self.forward = QtGui.QPushButton(self.centralwidget)
        self.forward.setObjectName(_fromUtf8("forward"))

        self.frameEdit = SelectAllLineEdit(self.centralwidget)
        self.frameEdit.setObjectName(_fromUtf8("frameEdit"))

        self.showFrame = QtGui.QPushButton(self.centralwidget)
        self.showFrame.setObjectName(_fromUtf8("show frame"))

        self.fpsLabel = QtGui.QLabel(self.centralwidget)
        self.speedSlider.setObjectName(_fromUtf8("speedSlider"))
        self.fpsLabel.setAlignment(QtCore.Qt.AlignRight)

        self.videoSlider = VideoSlider(self.centralwidget)
        self.videoSlider.setObjectName(_fromUtf8("videoSlider"))
        self.videoSlider.setOrientation(QtCore.Qt.Horizontal)
        self.videoSlider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.videoSlider.setMaximumHeight(10)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.assign_icons()
        self.assign_styles()

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.forward.setText(_translate("MainWindow", "forward", None))
        self.backward.setText(_translate("MainWindow", "backward", None))
        self.playPause.setText(_translate("MainWindow", "play", None))
        self.showFrame.setText(_translate("MainWindow", "show frame", None))
        self.openVideo.setText(_translate("MainWindow", "open video", None))
        self.openData.setText(_translate("MainWindow", "open data", None))
        self.undoChange.setText(_translate("MainWindow", "undo change", None))
        self.redoChange.setText(_translate("MainWindow", "redo change", None))
        self.saveData.setText(_translate("MainWindow", "save data", None))
        self.showHistory.setText(_translate("MainWindow", "show history", None))
        self.swapAnts.setText(_translate("MainWindow", "swap ants", None))
        self.showFaults.setText(_translate("MainWindow", "show faults", None))
        self.nextFault.setText(_translate("MainWindow", "next fault", None))
        self.toggleHighlight.setText(_translate("MainWindow", "toggle highlight", None))
        self.faultLabel.setText(_translate("MainWindow", "", None))
        self.faultNumLabel.setText(_translate("MainWindow", "", None))
        self.loadChanges.setText(_translate("MainWindow", "load changes", None))
        self.cancelButton.setText(_translate("MainWindow", "cancel", None))
        self.saveChangesToFile.setText(_translate("MainWindow", "save changes", None))
        self.swapTailHead.setText(_translate("MainWindow", "swap tail and head", None))
        self.settingsButton.setText(_translate("MainWindow", "settings", None))
        self.previousFault.setText(_translate("MainWindow", "previous fault", None))
        self.sequenceButton.setText(_translate("MainWindow", "sequence", None))
        self.showMSERsButton.setText(_translate("MainWindow", "show MSERs", None))
        self.showCertaintyButton.setText(_translate("MainWindow", "plot certainty", None))

    def assign_icons(self):
        self.forward.setIcon(QtGui.QIcon(QtGui.QPixmap('src/forward.png')))
        self.forward.setIconSize(QtCore.QSize(20, 20))
        self.backward.setIcon(QtGui.QIcon(QtGui.QPixmap('src/backward.png')))
        self.backward.setIconSize(QtCore.QSize(20, 20))
        self.playPause.setIcon(QtGui.QIcon(QtGui.QPixmap('src/play.png')))
        self.playPause.setIconSize(QtCore.QSize(20, 20))

    def assign_styles(self):
        self.faultLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.faultNumLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.faultLabel.setWordWrap(True)