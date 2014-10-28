# coding=utf8
__author__ = 'flipajs'

import os
import re
import copy
import pickle
import default_settings

from PyQt4 import QtCore, QtGui
import cv2

from viewer.gui.img_controls import img_controls_qt, utils, markers
from my_view import *
from viewer import video_manager
import visualization_utils
from viewer.identity_manager import IdentityManager
from viewer.gui.img_controls.dialogs import SettingsDialog
from gui.img_sequence import img_sequence_widget
from gui.plot import plot_widget


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s


class ImgControls(QtGui.QMainWindow, img_controls_qt.Ui_MainWindow):
    def __init__(self):

        super(ImgControls, self).__init__()
        self.graphics_view_old_w = 0
        self.graphics_view_old_h = 0
        self.setupUi(self)
        self.graphics_view_old_w = self.video_widget.width()
        self.graphics_view_old_h = self.video_widget.height()

        self.lines_layout.addWidget(self.informationLabel)
        # self.informationLabel.setMinimumHeight(40)

        self.scene = QtGui.QGraphicsScene()

        graphics_view_widget = QtGui.QWidget()
        self.graphics_view = MyView(graphics_view_widget)

        self.splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.splitter1.addWidget(self.video_widget)



        #self.main_line_layout.addWidget(self.video_widget)
        self.video_layout.addWidget(self.graphics_view)
        self.video_layout.addWidget(self.video_control_widget)

        self.video_control_layout.addWidget(self.videoSlider)
        self.video_control_layout.addWidget(self.video_control_buttons_widget)

        self.video_control_buttons_layout.addWidget(self.fpsLabel)
        self.video_control_buttons_layout.addWidget(self.speedSlider)
        self.video_control_buttons_layout.addWidget(self.backward)
        self.video_control_buttons_layout.addWidget(self.playPause)
        self.video_control_buttons_layout.addWidget(self.showFrame)
        self.video_control_buttons_layout.addWidget(self.frameEdit)
        self.video_control_buttons_layout.addWidget(self.forward)

        self.pixMap = None
        self.video = None
        self.items = []
        self.pixMapItem = None
        self.identity_markers = []
        self.history_markers = dict()
        self.forward_markers = dict()
        self.ant_highlighters = []
        self.is_highlighting = True
        self.timer = QTimer(self)
        self.frame_rate = 30
        self.identity_manager = None
        self.autosave_filepath = 'autosave.arr'
        self.change_count = 0
        self.showing_faulty_frames = False
        self.current_fault_index = 0
        self.current_fault = None
        self.ordered_faults = []
        self.settable_buttons = []

        self.init_settable_buttons()

        self.setStyleSheet("font:8pt \"Arial\"")

        self.init_graphic_view()
        self.init_speed_slider()

        self.connect_shortcuts()
        self.connect_GUI()
        self.set_fault_utils_visibility(False)

        # DEBUG COMMANDS
        # self.identity_manager = IdentityManager('data/noplast2262-new_results.arr')
        # self.identity_manager = IdentityManager('/home/flipajs/Desktop/ferda-webcam1_3194_results.arr')
        # self.identity_manager = IdentityManager('/home/flipajs/Downloads/c_bigLense_colormarks3.arr')
        # self.identity_manager = IdentityManager('/home/flipajs/Downloads/corrected_021014.arr.cng')

        # self.delete_history_markers()
        # self.delete_forward_markers()
        # self.init_identity_markers(self.identity_manager.ant_num, self.identity_manager.group_num)
        self.load_video_debug()
        # print tests.test_seek(self.video)
        # END OF DEBUG COMMANDS

        self.sequence_view = img_sequence_widget.ImgSequenceWidget(self.video)

        self.splitter1.addWidget(self.sequence_view)
        self.main_line_layout.addWidget(self.splitter1)

        # self.sequence_view.setMinimumWidth(350)
        self.add_controls_2_scene()

        self.plot_widget = plot_widget.PlotWidget()
        self.bottom_line_layout.addWidget(self.plot_widget)

        self.update()
        self.show()
        self.sequence_view.update_sequence(100, 100)

    def add_controls_2_scene(self):
        el = QtGui.QGraphicsEllipseItem(10, 10, 10, 10)
        el.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.scene.addItem(el)

    def connect_GUI(self):
        """Connects GUI elements to appropriate methods"""
        self.forward.clicked.connect(self.load_next_frame)
        self.backward.clicked.connect(self.load_previous_frame)
        self.playPause.clicked.connect(self.play_pause)
        self.timer.timeout.connect(self.load_next_frame)
        self.speedSlider.valueChanged.connect(self.speed_slider_changed)
        self.showFrame.clicked.connect(self.show_frame)
        self.openVideo.clicked.connect(self.open_video)
        self.openData.clicked.connect(self.open_data)
        self.videoSlider.valueChanged.connect(self.video_slider_changed)
        self.undoChange.clicked.connect(self.undo_change)
        self.saveData.clicked.connect(self.save_data)
        self.redoChange.clicked.connect(self.redo_change)
        self.showHistory.clicked.connect(self.show_history_and_forward)
        self.swapAnts.clicked.connect(self.swap_ants)
        self.showFaults.clicked.connect(self.show_faults)
        self.nextFault.clicked.connect(self.next_fault)
        self.toggleHighlight.clicked.connect(self.toggle_highlight)
        self.frameEdit.returnPressed.connect(self.show_frame)
        self.cancelButton.clicked.connect(self.cancel_show_faults)
        self.loadChanges.clicked.connect(self.load_changes)
        self.saveChangesToFile.clicked.connect(self.save_changes)
        self.swapTailHead.clicked.connect(self.swap_tail_head)
        self.settingsButton.clicked.connect(self.show_settings_dialog)
        self.previousFault.clicked.connect(self.previous_fault)

    def init_settable_buttons(self):
        """Adds those buttons which have user settable shortcuts into self.settable_buttons"""
        self.settable_buttons.append(self.forward)
        self.settable_buttons.append(self.backward)
        self.settable_buttons.append(self.playPause)
        self.settable_buttons.append(self.openData)
        self.settable_buttons.append(self.openVideo)
        self.settable_buttons.append(self.saveData)
        self.settable_buttons.append(self.settingsButton)
        self.settable_buttons.append(self.loadChanges)
        self.settable_buttons.append(self.saveChangesToFile)
        self.settable_buttons.append(self.undoChange)
        self.settable_buttons.append(self.redoChange)
        self.settable_buttons.append(self.showHistory)
        self.settable_buttons.append(self.swapAnts)
        self.settable_buttons.append(self.swapTailHead)
        self.settable_buttons.append(self.showFaults)
        self.settable_buttons.append(self.nextFault)
        self.settable_buttons.append(self.previousFault)
        self.settable_buttons.append(self.toggleHighlight)
        self.settable_buttons.append(self.cancelButton)

    def connect_shortcuts(self):
        """Connects shortcuts from settings to all settable buttons"""
        settings = QSettings("Ants correction tool")
        for button in self.settable_buttons:
            button.setShortcut(settings.value(str(button.objectName()), default_settings.get_default(str(button.objectName())), QtGui.QKeySequence))

    def set_fault_utils_visibility(self, visibility):
        """Sets visibility to all components used only when showing faulty frames"""
        self.toggleHighlight.setVisible(visibility)
        self.nextFault.setVisible(visibility)
        self.faultLabel.setVisible(visibility)
        self.cancelButton.setVisible(visibility)
        self.faultNumLabel.setVisible(visibility)
        self.previousFault.setVisible(visibility)

    def position_bottom_panel(self):
        """Sets correct positions to all components on bottom of the window"""
        settings = QtCore.QSettings("Ants correction tool")
        w = self.width()
        h = self.height()
        button_width = 100
        spacing = 3
        button_height = 3*settings.value('bottom_panel_height', default_settings.get_default('bottom_panel_height'), int)/4
        video_slider_gap = 0


        # self.showFrame.setGeometry(QtCore.QRect(w - 1 * button_width, h - button_height + spacing, button_width - spacing, button_height - spacing))
        # self.frameEdit.setGeometry(QtCore.QRect(w - 2 * button_width, h - button_height + spacing, button_width - spacing, button_height - spacing))
        # self.forward.setGeometry(QtCore.QRect(w - 3 * button_width, h - button_height + spacing, button_width - spacing, button_height - spacing))
        # self.playPause.setGeometry(QtCore.QRect(w - 4 * button_width, h - button_height + spacing, button_width - spacing, button_height - spacing))
        # self.backward.setGeometry(QtCore.QRect(w - 5 * button_width, h - button_height + spacing, button_width - spacing, button_height - spacing))
        # self.speedSlider.setGeometry(QtCore.QRect(w - 8 * button_width, h - button_height + spacing, 3 * (button_width - spacing), button_height - spacing))
        # self.fpsLabel.setGeometry(QtCore.QRect(w - 9 * button_width, h - button_height + spacing + 5, button_width - spacing, button_height - spacing))
        # self.videoSlider.setGeometry(QtCore.QRect(settings.value('side_panel_width', default_settings.get_default('side_panel_width'), int) + video_slider_gap, h - settings.value('bottom_panel_height', default_settings.get_default('bottom_panel_height'), int) + spacing, w - settings.value('side_panel_width', default_settings.get_default('side_panel_width'), int) - 2 * video_slider_gap, settings.value('bottom_panel_height', default_settings.get_default('bottom_panel_height'), int) - button_height - spacing))
        # self.videoSlider.setGeometry()

    def position_side_panel(self, side_panel_width):
        """Sets correct positions to all components on left side of the window"""
        w = self.width()
        h = self.height()
        spacing = 3
        button_height = 30

        self.menu_panel_layout.addWidget(self.openData)
        self.menu_panel_layout.addWidget(self.openVideo)
        self.menu_panel_layout.addWidget(self.saveData)
        self.menu_panel_layout.addWidget(self.settingsButton)
        self.menu_panel_layout.addWidget(self.loadChanges)
        self.menu_panel_layout.addWidget(self.saveChangesToFile)
        self.menu_panel_layout.addWidget(self.undoChange)
        self.menu_panel_layout.addWidget(self.redoChange)
        self.menu_panel_layout.addWidget(self.showHistory)
        self.menu_panel_layout.addWidget(self.swapAnts)
        self.menu_panel_layout.addWidget(self.swapTailHead)
        self.menu_panel_layout.addWidget(self.showFaults)
        self.menu_panel_layout.addWidget(self.nextFault)
        self.menu_panel_layout.addWidget(self.previousFault)
        self.menu_panel_layout.addWidget(self.toggleHighlight)
        self.menu_panel_layout.addWidget(self.faultNumLabel)
        self.menu_panel_layout.addWidget(self.faultLabel)
        self.menu_panel_layout.addWidget(self.cancelButton)

        # self.openData.setGeometry(QtCore.QRect(0, 0*button_height, side_panel_width, button_height - spacing))
        # self.openVideo.setGeometry(QtCore.QRect(0, 1*button_height, side_panel_width, button_height - spacing))
        # self.saveData.setGeometry(QtCore.QRect(0, 2 * button_height, side_panel_width, button_height - spacing))
        # self.settingsButton.setGeometry(QtCore.QRect(0, 3 * button_height, side_panel_width, button_height - spacing))
        # self.loadChanges.setGeometry(QtCore.QRect(0, 5 * button_height, side_panel_width, button_height - spacing))
        # self.saveChangesToFile.setGeometry(QtCore.QRect(0, 6 * button_height, side_panel_width, button_height - spacing))
        # self.undoChange.setGeometry(QtCore.QRect(0, 8 * button_height, side_panel_width, button_height - spacing))
        # self.redoChange.setGeometry(QtCore.QRect(0, 9 * button_height, side_panel_width, button_height - spacing))
        # self.showHistory.setGeometry(QtCore.QRect(0, 10 * button_height, side_panel_width, button_height - spacing))
        # self.swapAnts.setGeometry(QtCore.QRect(0, 11 * button_height, side_panel_width, button_height - spacing))
        # self.swapTailHead.setGeometry(QtCore.QRect(0, 12 * button_height, side_panel_width, button_height - spacing))
        # self.showFaults.setGeometry(QtCore.QRect(0, 14 * button_height, side_panel_width, button_height - spacing))
        # self.nextFault.setGeometry(QtCore.QRect(0, 16 * button_height, side_panel_width, button_height - spacing))
        # self.previousFault.setGeometry(QtCore.QRect(0, 17 * button_height, side_panel_width, button_height - spacing))
        # self.toggleHighlight.setGeometry(QtCore.QRect(0, 18 * button_height, side_panel_width, button_height - spacing))
        # self.faultNumLabel.setGeometry(QtCore.QRect(0, 19 * button_height, side_panel_width, button_height - spacing))
        # self.faultLabel.setGeometry(QtCore.QRect(0, 20 * button_height, side_panel_width, button_height - spacing))
        # self.cancelButton.setGeometry(QtCore.QRect(0, 21 * button_height, side_panel_width, button_height - spacing))

    def init_speed_slider(self):
        """Initiates components associated with speed of viewing videos"""
        self.speedSlider.setValue(self.frame_rate)
        self.timer.setInterval(1000 / self.frame_rate)
        self.fpsLabel.setText(str(self.frame_rate) + ' fps')
        self.speedSlider.setMinimum(1)
        self.speedSlider.setMaximum(120)

    def init_video_slider(self):
        """Initiates slider used to control position in video"""
        self.videoSlider.setMinimum(0)
        self.videoSlider.setTickInterval(1)

    def resizeEvent(self, QEvent):
        if self.graphics_view_old_w > 0:

            m11 = self.graphics_view.transform().m11()
            m22 = self.graphics_view.transform().m22()
            # scale = min(m11 * (self.video_widget.width() / float(1024)),
            #             m22 * (self.video_widget.height() / float(1024)))

            scale = min((self.video_widget.width() / float(self.graphics_view_old_w)),
                        (self.video_widget.height() / float(self.graphics_view_old_h)))

            scale = self.video_widget.width() / float(self.graphics_view_old_w)
            print scale

            self.graphics_view_old_w = self.video_widget.width()
            self.graphics_view_old_h = self.video_widget.height()

            self.graphics_view.scale(scale, scale)

            # self.graphics_view.zoom(scale, QPointF(512, 512))


        # self.graphics_view.scale(1.01, 1.01)
        # self.graphics_view.centerOn(QPointF(512, 512))
        return
        self.graphics_view_full()

    def graphics_view_full(self):
        """Positions graphics view onto the correct place"""
        settings = QSettings("Ants correction tool")
        w = self.width()
        h = self.height()
        self.graphics_view.setGeometry(settings.value('side_panel_width', default_settings.get_default('side_panel_width'), int), 0, w - settings.value('side_panel_width', default_settings.get_default('side_panel_width'), int), h - settings.value('bottom_panel_height', default_settings.get_default('bottom_panel_height'), int))
        self.position_bottom_panel()
        self.position_side_panel(settings.value('side_panel_width', default_settings.get_default('side_panel_width'), int))

    def init_graphic_view(self):
        """Initiates graphics view"""
        self.graphics_view_full()
        self.graphics_view.setObjectName(_fromUtf8("graphics_view"))
        self.graphics_view.setScene(self.scene)
        val = 2000
        # self.graphics_view.setSceneRect(-val, -val, 2*val, 2*val)

    def load_video_debug(self):
        """Loads fixed video. Used for debug."""
        # self.video = video_manager.VideoManager('/home/flipajs/Dropbox/PycharmProjects/data/NoPlasterNoLid800/NoPlasterNoLid800.m4v')
        # self.video = video_manager.VideoManager('/home/flipajs/my_video-16_c.mkv')
        self.video = video_manager.VideoManager('/home/flipajs/Downloads/c_bigLense_colormarks3.avi')
        image = self.video.next_img()
        self.pixMap = utils.cvimg2qtpixmap(image)
        utils.view_add_bg_image(self.graphics_view, self.pixMap)
        item = self.scene.addPixmap(self.pixMap)
        self.pixMapItem = item
        self.update_frame_number()
        if self.identity_manager is not None:
            self.position_identity_markers()

        try:
            self.frame_rate = int(self.video.fps())
        except:
            self.frame_rate = 30

        self.speedSlider.setValue(self.frame_rate)
        self.videoSlider.setMaximum(self.video.total_frame_count())
        self.videoSlider.setValue(self.video.frame_number())

    def load_next_frame(self):
        """Loads next frame of the video and displays it. If there is no next frame, calls self.out_of_frames"""
        if self.video is not None:
            img = self.video.next_img()
            if not img is None:
                if self.pixMapItem is not None:
                    self.scene.removeItem(self.pixMapItem)
                self.pixMap = utils.cvimg2qtpixmap(img)
                utils.view_add_bg_image(self.graphics_view, self.pixMap)
                item = self.scene.addPixmap(self.pixMap)
                self.pixMapItem = item
                self.update_frame_number()
                self.delete_history_markers()
                self.delete_forward_markers()
                if self.identity_manager is not None:
                    self.position_identity_markers()
            else:
                self.out_of_frames()

    def load_previous_frame(self):
        """Loads previous frame of the video if there is such and displays it"""
        if self.video is not None:
            img = self.video.prev_img()
            if not img is None:
                if self.pixMapItem is not None:
                    self.scene.removeItem(self.pixMapItem)
                self.pixMap = utils.cvimg2qtpixmap(img)
                utils.view_add_bg_image(self.graphics_view, self.pixMap)
                item = self.scene.addPixmap(self.pixMap)
                self.pixMapItem = item
                self.update_frame_number()
                self.delete_history_markers()
                self.delete_forward_markers()
                if self.identity_manager is not None:
                    self.position_identity_markers()

    def play_pause(self):
        """Method of playPause button."""
        settings = QSettings("Ants correction tool")
        if self.video is not None:
            if self.timer.isActive():
                self.timer.stop()
                self.playPause.setIcon(QtGui.QIcon(QtGui.QPixmap('src/play.png')))
                self.playPause.setText("play")
                self.playPause.setShortcut(settings.value(self.playPause.objectName(), default_settings.get_default(str(self.playPause.objectName())), QtGui.QKeySequence))
            else:
                self.timer.start()
                self.playPause.setIcon(QtGui.QIcon(QtGui.QPixmap('src/pause.png')))
                self.playPause.setText("pause")
                self.playPause.setShortcut(settings.value(self.playPause.objectName(), default_settings.get_default(str(self.playPause.objectName())), QtGui.QKeySequence))

    def update_frame_number(self):
        """Updates values of components displaying frame number"""
        self.frameEdit.setText(QString.number(self.video.frame_number() + 1) + QString('/') + QString.number(
            self.video.total_frame_count()))
        self.videoSlider.setValue(self.video.frame_number())

    def update_fault_number(self):
        """Updates value of components displaying number of current fault inspected"""
        if self.current_fault_index < 0:
            self.faultNumLabel.setText("")
        else:
            self.faultNumLabel.setText(QString.number(self.current_fault_index + 1) + QString('/') + QString.number(
                self.identity_manager.get_fault_num()))

    def init_identity_markers(self, identity_num, group_num):
        """Initiates marker that show positions of individual ants. Note that they are all positioned in the topleft
        corner of the window. To position them, use self.position_identity_markers
        """
        settings = QSettings("Ants correction tool")
        for i in range(identity_num):
            ant_markers = []

            if settings.value('view_mode', default_settings.get_default('view_mode'), str) == 'individual':
                r, g, b = visualization_utils.get_color(i, identity_num)
            elif settings.value('view_mode', default_settings.get_default('view_mode'), str) == 'group':
                r, g, b = visualization_utils.get_color(self.identity_manager.get_group(i), group_num)


            item = markers.CenterMarker(0, 0, settings.value('center_marker_size', default_settings.get_default('center_marker_size'), int), QColor(r, g, b), i, self.marker_changed)
            item.setZValue(0.5)
            ant_markers.append(item)
            self.items.append(item)
            self.scene.addItem(item)

            if settings.value('head_detection', default_settings.get_default('head_detection'), bool):
                item = markers.HeadMarker(0, 0, settings.value('head_marker_size', default_settings.get_default('head_marker_size'), int), QColor(r, g, b), i, self.marker_changed)
            else:
                item = markers.TailHeadMarker(0, 0, settings.value('head_marker_size', default_settings.get_default('head_marker_size'), int), QColor(r, g, b), i, self.marker_changed)
            item.setZValue(0.5)
            ant_markers.append(item)
            self.items.append(item)
            self.scene.addItem(item)

            if settings.value('head_detection', default_settings.get_default('head_detection'), bool):
                item = markers.TailMarker(0, 0, settings.value('tail_marker_size', default_settings.get_default('tail_marker_size'), int), QColor(r, g, b), i, self.marker_changed)
            else:
                item = markers.TailHeadMarker(0, 0, settings.value('tail_marker_size', default_settings.get_default('tail_marker_size'), int), QColor(r, g, b), i, self.marker_changed)
            item.setZValue(0.5)
            ant_markers.append(item)
            self.items.append(item)
            self.scene.addItem(item)
            # Connect markers.
            ant_markers[0].add_head_marker(ant_markers[1])
            ant_markers[0].add_tail_marker(ant_markers[2])
            ant_markers[1].add_center_marker(ant_markers[0])
            ant_markers[1].add_other_marker(ant_markers[2])
            ant_markers[2].add_center_marker(ant_markers[0])
            ant_markers[2].add_other_marker(ant_markers[1])
            #Remember markers.
            self.identity_markers.append(ant_markers)

    def delete_identity_markers(self):
        """Removes all markers that are in the window."""
        for i in range(len(self.identity_markers)):
            for j in range(3):
                self.scene.removeItem(self.identity_markers[i][j])
        self.identity_markers = []

    def draw_identity(self, identity, ant_num):
        """Positions markers for individual ant onto given identity. Also sets tooltips displaying the certainty, lost
        and in collision parameters
        """
        try:
            self.identity_markers[ant_num][0].setCenterPos(identity['cx'], identity['cy'])
        except:
            self.identity_markers[ant_num][0].setPos(0, 0)
        try:
            self.identity_markers[ant_num][1].setCenterPos(identity['hx'], identity['hy'])
        except:
            self.identity_markers[ant_num][1].setPos(0, 0)
        try:
            self.identity_markers[ant_num][2].setCenterPos(identity['bx'], identity['by'])
        except:
            self.identity_markers[ant_num][2].setPos(0, 0)
        try:
            self.identity_markers[ant_num][0].setToolTip(
                'Certainty: ' + str(identity['certainty']) + '\nIs lost: ' + str(
                    identity['lost']) + '\nCollides with: ' + str(identity['in_collision_with']))
        except:
            self.identity_markers[ant_num][0].setToolTip("")
        try:
            self.identity_markers[ant_num][1].setToolTip(
                'Certainty: ' + str(identity['certainty']) + '\nIs lost: ' + str(
                    identity['lost']) + '\nCollides with: ' + str(identity['in_collision_with']))
        except:
            self.identity_markers[ant_num][1].setToolTip("")
        try:
            self.identity_markers[ant_num][2].setToolTip(
                'Certainty: ' + str(identity['certainty']) + '\nIs lost: ' + str(
                    identity['lost']) + '\nCollides with: ' + str(identity['in_collision_with']))
        except:
            self.identity_markers[ant_num][2].setToolTip("")

    def speed_slider_changed(self):
        """Method invoked when value of slider controlling speed of video changed it's value"""
        self.frame_rate = self.speedSlider.value()
        self.timer.setInterval(1000 / self.frame_rate)
        self.fpsLabel.setText(str(self.frame_rate) + ' fps')

    def video_slider_changed(self):
        """Method invoked when slider controlling video position changed. To differentiate between
        situations when the slider was changed by user or by part of the program, use videoSlider.usercontrolled
        and videoSlider.recentlyreleased
        """
        if self.videoSlider.usercontrolled:
            self.change_frame(self.videoSlider.value())
        elif self.videoSlider.recentlyreleased:
            self.videoSlider.recentlyreleased = False
            self.change_frame(self.videoSlider.value())

    def show_frame(self):
        """Method of showFrame button. Splits text from frameEdit and calls self.change_frame"""
        if self.video is not None:
            text = self.frameEdit.text()
            match = re.search('[^/]*', text)
            text = match.group()
            if str(text).isdigit():
                self.change_frame(int(text) - 1)
                self.showFrame.setFocus()

    def change_frame(self, position):
        """Changes current frame to position given. If there is no such position, calls self.out_of_frames"""
        if self.video is not None:
            img = self.video.seek_frame_hybrid(position)
            if img is not None:
                if self.pixMapItem is not None:
                    self.scene.removeItem(self.pixMapItem)
                self.pixMap = utils.cvimg2qtpixmap(img)
                utils.view_add_bg_image(self.graphics_view, self.pixMap)
                item = self.scene.addPixmap(self.pixMap)
                self.pixMapItem = item
                self.update_frame_number()
                self.delete_history_markers()
                self.delete_forward_markers()
                if self.identity_manager is not None:
                    self.position_identity_markers()
            else:
                self.out_of_frames()

    def out_of_frames(self):
        """Stops playing of the video if it is playing."""
        if self.timer.isActive():
            self.timer.stop()
            self.playPause.setIcon(QtGui.QIcon(QtGui.QPixmap('src/play.png')))
            self.playPause.setText("play")

    def open_data(self):
        """Method of openData button. Shows a dialog to select a file and than opens ant positions data from that file"""
        filename = unicode(QFileDialog.getOpenFileName(self, "Open Data", "", "Data files (*.arr);;All files (*.*)"))
        if filename != "":
            self.identity_manager = IdentityManager(filename)
            self.delete_identity_markers()
            self.delete_history_markers()
            self.delete_forward_markers()
            self.init_identity_markers(self.identity_manager.ant_num, self.identity_manager.group_num)
            self.stop_showing_faults()
            if self.video is not None:
                self.position_identity_markers()

    def save_data(self):
        """Method of saveData button. Shows a dialog to select a file and saves ant positions into it."""
        filename = unicode(QtGui.QFileDialog.getSaveFileName(self, "Open Data", "", "Data files (*.arr);;All files (*.*)"))
        if filename != "" and self.identity_manager is not None:
            self.identity_manager.save(filename)
            self.autosave_filepath = filename

    def open_video(self):
        """Method of openVideo button. Shows dialog to select file and calls self.load_video"""
        filename = QtGui.QFileDialog.getOpenFileName(self, "Open video", "", "Video files (*.avi *.mp4 *.mkv);;All files (*.*)")
        if filename != "":
            self.load_video(filename)

    def load_video(self, filename):
        """Loads video from given filename. Sets values of all components involved accordingly"""
        self.stop_showing_faults()

        codec = QTextCodec.codecForName('utf8')
        QTextCodec.setCodecForLocale(codec)
        QTextCodec.setCodecForCStrings(codec)
        QTextCodec.setCodecForTr(codec)
        filename = str(filename.toAscii())
        filename = os.path.normpath(filename)
        self.video = video_manager.VideoManager(filename)
        try:
            self.frame_rate = int(self.video.fps())
        except:
            self.frame_rate = 30

        self.speedSlider.setValue(self.frame_rate)
        self.videoSlider.setMaximum(self.video.total_frame_count())
        self.videoSlider.setValue(self.video.frame_number())
        self.load_next_frame()

    # def marker_changed(self, ant_id):
    # 	if self.identity_manager is not None and self.video is not None:
    # 		all_data = dict()
    # 		for antId in range(self.identity_manager.ant_num):
    # 			old_data = self.identity_manager.get_positions(self.video.frame_number(), antId)
    # 			new_data = copy.copy(old_data)
    # 			new_data['cx'] = self.identity_markers[antId][0].centerPos().x()
    # 			new_data['cy'] = self.identity_markers[antId][0].centerPos().y()
    # 			new_data['hx'] = self.identity_markers[antId][1].centerPos().x()
    # 			new_data['hy'] = self.identity_markers[antId][1].centerPos().y()
    # 			new_data['bx'] = self.identity_markers[antId][2].centerPos().x()
    # 			new_data['by'] = self.identity_markers[antId][2].centerPos().y()
    # 			all_data[antId] = new_data
    # 		self.identity_manager.write_change(self.video.frame_number(), 'movement', all_data)
    # 		self.change_count += 1
    # 		if self.change_count == settings.autosave_count:
    # 			self.identity_manager.save(self.autosave_filepath)

    def marker_changed(self, id):
        """A method invoked when a mouse was released from one of the identity markers. Scans for all markers, remembers
        which were selected and writes those into changes.
        """
        settings = QSettings("Ants correction tool")
        if self.identity_manager is not None and self.video is not None:
            changes = dict()
            for ant_id in range(self.identity_manager.ant_num):
                ant_change = {'cx': None, 'cy': None, 'hx': None, 'hy': None, 'bx': None, 'by': None}
                changed = False
                if self.identity_markers[ant_id][0].recently_changed:
                    self.identity_markers[ant_id][0].recently_changed = False
                    ant_change['cx'] = self.identity_markers[ant_id][0].centerPos().x()
                    ant_change['cy'] = self.identity_markers[ant_id][0].centerPos().y()
                    changed = True
                if self.identity_markers[ant_id][1].recently_changed:
                    self.identity_markers[ant_id][1].recently_changed = False
                    ant_change['hx'] = self.identity_markers[ant_id][1].centerPos().x()
                    ant_change['hy'] = self.identity_markers[ant_id][1].centerPos().y()
                    changed = True
                if self.identity_markers[ant_id][2].recently_changed:
                    self.identity_markers[ant_id][2].recently_changed = False
                    ant_change['bx'] = self.identity_markers[ant_id][2].centerPos().x()
                    ant_change['by'] = self.identity_markers[ant_id][2].centerPos().y()
                    changed = True
                if changed:
                    changes[ant_id] = ant_change
            self.identity_manager.write_change(self.video.frame_number(), 'movement', changes)
            self.change_count += 1
            if self.change_count == settings.value('autosave_count', default_settings.get_default('autosave_count'), int):
                self.identity_manager.save(self.autosave_filepath)


    def undo_change(self):
        """Method of undo button. Undoes a change"""
        if self.identity_manager is not None and self.video is not None:
            self.identity_manager.undo_change(self.video.frame_number())
            self.position_identity_markers()

    def redo_change(self):
        """Method of redo button. Redoes a change"""
        if self.identity_manager is not None and self.video is not None:
            self.identity_manager.redo_change(self.video.frame_number())
            self.position_identity_markers()

    def add_forward_markers(self, depth, ant_id):
        """Adds markers indicating future positions of the ant given. Number of markers = depth. If 'markers_shown_history'
        setting equals all, displays all three markers, otherwise just the centre one.
        """
        settings = QSettings("Ants correction tool")
        if self.identity_manager is not None and self.video is not None:
            for i in range(1, min(depth, int(self.video.total_frame_count() - self.video.frame_number() - 1)) + 1):
                pos = self.identity_manager.get_positions(self.video.frame_number() + i, ant_id)

                if settings.value('view_mode', default_settings.get_default('view_mode'), str) == 'individual':
                    r, g, b = visualization_utils.get_color(ant_id, self.identity_manager.ant_num)
                elif settings.value('view_mode', default_settings.get_default('view_mode'), str) == 'group':
                    r, g, b = visualization_utils.get_color(self.identity_manager.get_group(ant_id), self.identity_manager.group_num)

                size = settings.value('center_marker_size', default_settings.get_default('center_marker_size'), int)
                item = utils.add_circle(7, QColor(r, g, b))
                item.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)
                item.setFlag(QtGui.QGraphicsItem.ItemIsMovable, False)
                item.setOpacity(visualization_utils.get_opacity(i, depth))
                item.setPos(pos['cx'] - size / 2, pos['cy'] - size / 2)
                item.setZValue(.5)

                self.scene.addItem(item)
                self.forward_markers[ant_id].append(item)

                if settings.value('markers_shown_history', default_settings.get_default('markers_shown_history'), str) == 'all':
                    size = settings.value('head_marker_size', default_settings.get_default('head_marker_size'), int)
                    item = utils.add_circle(10, QColor(r, g, b))
                    item.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)
                    item.setFlag(QtGui.QGraphicsItem.ItemIsMovable, False)
                    item.setOpacity(visualization_utils.get_opacity(i, depth))
                    item.setPos(pos['hx'] - size / 2, pos['hy'] - size / 2)
                    item.setZValue(.5)

                    self.scene.addItem(item)
                    self.forward_markers[ant_id].append(item)

                    size = settings.value('tail_marker_size', default_settings.get_default('tail_marker_size'), int)
                    item = utils.add_circle(10, QColor(r, g, b))
                    item.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)
                    item.setFlag(QtGui.QGraphicsItem.ItemIsMovable, False)
                    item.setOpacity(visualization_utils.get_opacity(i, depth))
                    item.setPos(pos['bx'] - size / 2, pos['by'] - size / 2)
                    item.setZValue(.5)

                    self.scene.addItem(item)
                    self.forward_markers[ant_id].append(item)

    def add_history_markers(self, depth, ant_id):
        """Adds markers indicating previous positions of the ant given. Number of markers = depth. If 'markers_shown_history'
        setting equals all, displays all three markers, otherwise just the centre one.
        """
        settings = QSettings("Ants correction tool")
        if self.identity_manager is not None and self.video is not None:
            for i in range(1, min(depth, self.video.frame_number()) + 1):
                pos = self.identity_manager.get_positions(self.video.frame_number() - i, ant_id)

                if settings.value('view_mode', default_settings.get_default('view_mode'), str) == 'individual':
                    r, g, b = visualization_utils.get_color(ant_id, self.identity_manager.ant_num)
                elif settings.value('view_mode', default_settings.get_default('view_mode'), str) == 'group':
                    r, g, b = visualization_utils.get_color(self.identity_manager.get_group(ant_id), self.identity_manager.group_num)

                size = settings.value('center_marker_size', default_settings.get_default('center_marker_size'), int)
                item = utils.add_circle(7, QColor(r, g, b))
                item.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)
                item.setFlag(QtGui.QGraphicsItem.ItemIsMovable, False)
                item.setOpacity(visualization_utils.get_opacity(i, depth))
                item.setPos(pos['cx'] - size / 2, pos['cy'] - size / 2)
                item.setZValue(.5)

                self.scene.addItem(item)
                self.history_markers[ant_id].append(item)

                if settings.value('markers_shown_history', default_settings.get_default('markers_shown_history'), str) == 'all':
                    size = settings.value('head_marker_size', default_settings.get_default('head_marker_size'), int)
                    item = utils.add_circle(10, QColor(r, g, b))
                    item.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)
                    item.setFlag(QtGui.QGraphicsItem.ItemIsMovable, False)
                    item.setOpacity(visualization_utils.get_opacity(i, depth))
                    item.setPos(pos['hx'] - size / 2, pos['hy'] - size / 2)
                    item.setZValue(.5)

                    self.scene.addItem(item)
                    self.history_markers[ant_id].append(item)

                    size = settings.value('tail_marker_size', default_settings.get_default('tail_marker_size'), int)
                    item = utils.add_circle(10, QColor(r, g, b))
                    item.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)
                    item.setFlag(QtGui.QGraphicsItem.ItemIsMovable, False)
                    item.setOpacity(visualization_utils.get_opacity(i, depth))
                    item.setPos(pos['bx'] - size / 2, pos['by'] - size / 2)
                    item.setZValue(.5)

                    self.scene.addItem(item)
                    self.history_markers[ant_id].append(item)

    def show_history_and_forward(self):
        """Method of showHistory button"""
        self.show_history()
        self.show_forward_positions()

    def show_history(self):
        """Iterates through all markers and shows history of those selected"""
        settings = QSettings("Ants correction tool")
        for marker_list in self.identity_markers:
            for marker in marker_list:
                if marker.isSelected():
                    if self.history_markers[marker.antId]:
                        self.delete_ant_history_markers(marker.antId)
                    else:
                        self.add_history_markers(settings.value('history_depth', default_settings.get_default('history_depth'), int), marker.antId)
                    break

    def show_forward_positions(self):
        """Iterates through all markers and shows future of those selected"""
        settings = QSettings("Ants correction tool")
        for marker_list in self.identity_markers:
            for marker in marker_list:
                if marker.isSelected():
                    if self.forward_markers[marker.antId]:
                        self.delete_ant_forward_markers(marker.antId)
                    else:
                        self.add_forward_markers(settings.value('forward_depth', default_settings.get_default('forward_depth'), int), marker.antId)
                    break

    def delete_ant_forward_markers(self, ant_id):
        """Removes all markers showing future of ant given"""
        for marker in self.forward_markers[ant_id]:
            self.scene.removeItem(marker)
        self.forward_markers[ant_id] = []

    def delete_forward_markers(self):
        """Removes all markers showing future of all ants."""
        for ant_id in self.forward_markers:
            for marker in self.forward_markers[ant_id]:
                self.scene.removeItem(marker)
        self.forward_markers = dict()
        if self.identity_manager is not None:
            for ant_id in range(self.identity_manager.ant_num):
                self.forward_markers[ant_id] = []

    def delete_ant_history_markers(self, ant_id):
        """Removes all markers showing history of ant given"""
        for marker in self.history_markers[ant_id]:
            self.scene.removeItem(marker)
        self.history_markers[ant_id] = []

    def delete_history_markers(self):
        """Removes all markers showing history of all ants."""
        for ant_id in self.history_markers:
            for marker in self.history_markers[ant_id]:
                self.scene.removeItem(marker)
        self.history_markers = dict()
        if self.identity_manager is not None:
            for ant_id in range(self.identity_manager.ant_num):
                self.history_markers[ant_id] = []

    def swap_ants(self):
        """Method of swapAnts button. If two ants are selected, swaps their positions."""
        settings = QSettings("Ants correction tool")
        selected = []
        for marker_list in self.identity_markers:
            for marker in marker_list:
                if marker.isSelected():
                    selected.append(marker.antId)
                    break
        if len(selected) == 2 and self.identity_manager is not None and self.video is not None:
            self.identity_manager.write_change(self.video.frame_number(), 'swap', selected)
            self.position_identity_markers()
            self.change_count += 1
            if self.change_count == settings.getValue('autosave_count'):
                self.identity_manager.save(self.autosave_filepath)

    def position_identity_markers(self):
        """Iterates through all ants ant positions markers showing their whereabouts according to current video frame."""
        for i in range(self.identity_manager.ant_num):
            self.draw_identity(self.identity_manager.get_positions(self.video.frame_number(), i), i)
            self.identity_markers[i][0].recently_changed = False
            self.identity_markers[i][1].recently_changed = False
            self.identity_markers[i][2].recently_changed = False

    def show_faults(self):
        """Method of showFaults button. Computes possible faults, prepares all necessities for showing them
        and proceeds to call next_fault.
        """
        settings = QSettings("Ants correction tool")
        if self.video is not None and self.identity_manager is not None:

            dialog = QMessageBox(QtGui.QMessageBox.NoIcon, "Computing", "Computing possible detection faults", QtGui.QMessageBox.Ok)
            dialog.show()
            dialog.button(QtGui.QMessageBox.Ok).setVisible(False)
            QtCore.QCoreApplication.processEvents()
            self.identity_manager.compute_faults()
            dialog.close()

            if self.identity_manager.get_fault_num() > 0:
                self.showing_faulty_frames = True
                self.set_fault_utils_visibility(True)

                self.ordered_faults = []
                if settings.value('switches_first', default_settings.get_default('switches_first'), bool):
                    for frame_no in self.identity_manager.get_faulty_switch_frames():
                        for fault in self.identity_manager.get_switch_faults(frame_no):
                            if settings.value('correction_mode', default_settings.get_default('correction_mode'), str) == 'group':
                                same_group = True
                                random_ant = fault['ants'].pop()
                                for ant in fault['ants']:
                                    if self.identity_manager.get_group(ant) != self.identity_manager.get_group(random_ant):
                                        same_group = False
                                        break
                                fault['ants'].add(random_ant)
                                if same_group:
                                    continue
                            self.ordered_faults.append({'frame': frame_no, 'fault': fault})
                    for frame_no in self.identity_manager.get_faulty_non_switch_frames():
                        for fault in self.identity_manager.get_non_switch_faults(frame_no):
                            self.ordered_faults.append({'frame': frame_no, 'fault': fault})
                else:
                    for frame_no in self.identity_manager.get_faulty_frames():
                        for fault in self.identity_manager.get_faults(frame_no):
                            if settings.value('correction_mode', default_settings.get_default('correction_mode'), str) == 'group' and fault['cause'] in self.identity_manager.switch_causes:
                                same_group = True
                                random_ant = fault['ants'].pop()
                                for ant in fault['ants']:
                                    if self.identity_manager.get_group(ant) == self.identity_manager.get_group(random_ant):
                                        same_group = False
                                        break
                                fault['ants'].add(random_ant)
                                if same_group:
                                    continue
                            self.ordered_faults.append({'frame': frame_no, 'fault': fault})

                if self.current_fault is not None:
                    for faulty_ant in self.current_fault['ants']:
                        for marker in self.identity_markers[faulty_ant]:
                            marker.setSelected(False)

                self.current_fault = None
                self.current_fault_index = -1
                self.next_fault()

    def next_fault(self):
        """Method of nextFault button. Shows next fault in the list of faults. The order of faults is determined by
        the 'switches first' setting.
        """
        settings = QSettings("Ants correction tool")
        if self.video is not None and self.identity_manager is not None and self.showing_faulty_frames:
            self.remove_highlighters()
            if settings.value('zoom_on_faults', default_settings.get_default('zoom_on_faults'), bool):
                self.graphics_view.fitInView(QRectF(0, 0, self.graphics_view.width(), self.graphics_view.height()))

            if self.current_fault is not None:
                for faulty_ant in self.current_fault['ants']:
                    for marker in self.identity_markers[faulty_ant]:
                        marker.setSelected(False)

            self.current_fault_index += 1
            if self.current_fault_index >= len(self.ordered_faults):
                self.stop_showing_faults()
                return
            self.current_fault = self.ordered_faults[self.current_fault_index]['fault']
            self.change_frame(self.ordered_faults[self.current_fault_index]['frame'])
            for faulty_ant in self.current_fault['ants']:
                self.highlight_ant(faulty_ant)
                for marker in self.identity_markers[faulty_ant]:
                    marker.setSelected(True)
            if settings.value('zoom_on_faults', default_settings.get_default('zoom_on_faults'), bool):
                self.zoom_on_ants(self.current_fault['ants'])
            self.set_fault_label(self.current_fault['cause'])
            if self.current_fault['cause'] in self.identity_manager.switch_causes:
                self.show_history_and_forward()
            self.update_fault_number()

    def previous_fault(self):
        """Method of previousFault button. Shows previous fault in the list of faults. The order of faults is determined by
        the 'switches first' setting.
        """
        settings = QSettings("Ants correction tool")
        if self.video is not None and self.identity_manager is not None and self.showing_faulty_frames and self.current_fault_index > 0:
            self.remove_highlighters()

            if settings.value('zoom_on_faults', default_settings.get_default('zoom_on_faults'), bool):
                self.graphics_view.fitInView(QRectF(0, 0, self.graphics_view.width(), self.graphics_view.height()))

            if self.current_fault is not None:
                for faulty_ant in self.current_fault['ants']:
                    for marker in self.identity_markers[faulty_ant]:
                        marker.setSelected(False)

            self.current_fault_index -= 1
            self.current_fault = self.ordered_faults[self.current_fault_index]['fault']
            self.change_frame(self.ordered_faults[self.current_fault_index]['frame'])
            for faulty_ant in self.current_fault['ants']:
                self.highlight_ant(faulty_ant)
                for marker in self.identity_markers[faulty_ant]:
                    marker.setSelected(True)
            if settings.value('zoom_on_faults', default_settings.get_default('zoom_on_faults'), bool):
                self.zoom_on_ants(self.current_fault['ants'])
            self.set_fault_label(self.current_fault['cause'])
            self.update_fault_number()

    def set_fault_label(self, cause):
        """Sets text on the label that shows cause of suspicion according to given cause."""
        if cause == "":
            self.faultLabel.setText("")
        elif cause == "len":
            self.faultLabel.setText("Suspicious length")
        elif cause == 'certainty':
            self.faultLabel.setText("Low certainty")
        elif cause == 'collision':
            self.faultLabel.setText("Collision of ants")
        elif cause == 'angle':
            self.faultLabel.setText("Turning too fast")
        elif cause == 'proximity':
            self.faultLabel.setText("Ants too close")
        elif cause == 'overlap':
            self.faultLabel.setText("Ants overlap")
        elif cause == 'lost':
            self.faultLabel.setText("Tracker has lost the ant")

    def highlight_ant(self, ant_id):
        """Draws a dotted circle around the ant given"""
        ant_len = ((self.identity_markers[ant_id][1].pos().x() - self.identity_markers[ant_id][2].pos().x()) ** 2 + (
            self.identity_markers[ant_id][1].pos().y() - self.identity_markers[ant_id][2].pos().y()) ** 2) ** .5

        pen_width = 5

        brush = QtGui.QBrush(QtCore.Qt.NoBrush)
        pen = QtGui.QPen(QtCore.Qt.DotLine)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setWidth(pen_width)
        pen.setColor(QColor('yellow'))

        width = ant_len * (2 ** .5) + pen_width + self.identity_markers[ant_id][2].rect().width()
        height = ant_len * (2 ** .5) + pen_width + self.identity_markers[ant_id][2].rect().height()
        item = QtGui.QGraphicsEllipseItem(self.identity_markers[ant_id][0].centerPos().x() - width / 2,
                                          self.identity_markers[ant_id][0].centerPos().y() - height / 2, width, height)

        item.setBrush(brush)
        item.setPen(pen)
        item.setOpacity(.6)
        item.setZValue(.2)

        if not self.is_highlighting:
            item.setVisible(False)

        self.scene.addItem(item)
        self.ant_highlighters.append(item)

    def remove_highlighters(self):
        """Removes all cirles around ants"""
        for highlighter in self.ant_highlighters:
            self.scene.removeItem(highlighter)
        self.ant_highlighters = []

    def toggle_highlight(self):
        """Turns highlighting of ants on ant off"""
        self.is_highlighting = not self.is_highlighting
        for highlighter in self.ant_highlighters:
            highlighter.setVisible(not highlighter.isVisible())

    def stop_showing_faults(self):
        """Cancels the process of showing faults."""
        self.current_fault = None
        self.showing_faulty_frames = False
        self.set_fault_utils_visibility(False)
        self.current_fault_index = -1
        self.update_fault_number()
        self.remove_highlighters()
        self.set_fault_label("")
        self.showFrame.setFocus()

    def cancel_show_faults(self):
        """Method of cancel button. Calls stop_showing_faults if it is needed"""
        if self.showing_faulty_frames:
            self.stop_showing_faults()

    def save_changes(self):
        """Method of saveChanges button. Displays dialog, reads a filename from it and saves changes done in this session
        along with previously loaded changes into the file.
        """
        if self.identity_manager is not None:
            filename = unicode(QtGui.QFileDialog.getSaveFileName(self, "Save changes", "", "Change files (*.cng)"))
            if os.path.splitext(filename)[1][1:].strip() != "cng":
                filename += '.cng'
            if filename != "":
                self.identity_manager.changes_to_file(filename)

    def load_changes(self):
        """Method of loadChanges button. Loads changes from a file obtained by showing dialog."""
        if self.identity_manager is not None:
            filename = unicode(QtGui.QFileDialog.getOpenFileName(self, "Open change file", "", "Change files (*.cng)"))
            if os.path.splitext(filename)[1][1:].strip() != "cng":
                filename += '.cng'
            if filename != "":
                self.identity_manager.changes_from_file(filename)
                self.position_identity_markers()

    def swap_tail_head(self):
        """Method of swapTailHead button. Swaps tail and head of selected ants. Note that this is only valid when
        'head_detection' setting is on.
        """
        selected = []
        for marker_list in self.identity_markers:
            for marker in marker_list:
                if marker.isSelected():
                    selected.append(marker.antId)
                    break
        if self.identity_manager is not None and self.video is not None:
            for ant in selected:
                self.identity_manager.swap_tail_head_from_frame(self.video.frame_number(), ant)
        self.position_identity_markers()

    def show_settings_dialog(self):
        """Method of showSettings button. Shows settings dialog. If it was accepted, updates all components affected
        by changes"""
        settings = QSettings("Ants correction tool")
        prev_head_detection = settings.value('head_detection', default_settings.get_default('head_detection'), bool)
        dialog = SettingsDialog(self, self.settable_buttons)
        dialog.exec_()
        if dialog.Accepted:
            self.position_bottom_panel()
            settings = QSettings("Ants correction tool")
            self.position_side_panel(settings.value('side_panel_width', default_settings.get_default('side_panel_width'), int))
            self.graphics_view_full()
            if self.identity_manager is not None:
                if not prev_head_detection and settings.value('head_detection', default_settings.get_default('head_detection'), bool):
                    self.identity_manager.pair_points()
                    self.identity_manager.detect_heads()
                self.delete_identity_markers()
                self.init_identity_markers(self.identity_manager.ant_num, self.identity_manager.group_num)
                self.position_identity_markers()
                self.identity_manager.set_new_settings()

    def zoom_on_ants(self, ants):
        """Zooms the view such that all given ants and a bit more will be in the view."""
        max_x = 0
        max_y = 0
        min_x = self.graphics_view.width()
        min_y = self.graphics_view.height()
        max_zoom = 5
        for ant in ants:
            for marker in self.identity_markers[ant]:
                if marker.pos().x() + marker.rect().width() > max_x:
                    max_x = marker.pos().x() + marker.rect().width()
                if marker.pos().y() + marker.rect().height() > max_y:
                    max_y = marker.pos().y() + marker.rect().height()
                if marker.pos().x() < min_x:
                    min_x = marker.pos().x()
                if marker.pos().y() < min_y:
                    min_y = marker.pos().y()
        center = QPointF(float(max_x + min_x) / 2, float(max_y + min_y)/2)
        scale = min(self.graphics_view.width() / float(max_x - min_x), self.graphics_view.height() / float(max_y - min_y))
        self.graphics_view.zoom(min(scale, max_zoom), center)