__author__ = 'flipajs'

import math
import sys

from PyQt6 import QtCore, QtGui, QtWidgets

from core.animal import colors_
from core.project.project import Project
from gui.img_controls import markers
from gui.img_controls.gui_utils import cvimg2qtpixmap
from gui.img_controls.my_view import MyView
from gui.settings import Settings as S_
from utils.video_manager import get_auto_video_manager
import core.config as config


MARKER_SIZE = 15


class VideoSlider(QtWidgets.QSlider):
    """A slider that changes it's value directly to the part where it was clicked instead of slowly sliding there.
    Also, it's nice! """

    def __init__(self, parent=None):
        super(VideoSlider, self).__init__(parent)
        self.usercontrolled = False
        self.recentlyreleased = False
        self.setPageStep(0)
        self.setSingleStep(0)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                background: white;
            }

            QSlider::sub-page:horizontal {
                background: green;
                background: qlineargradient(x1: 1, y1: 0,    x2: 0, y2: 0, stop: 0 #085700, stop: 1 #5DA556);
            }

            QSlider::handle:horizontal {
                background: #5DA556;
                border: 0px;
                width: 0px;
                margin-top: 0px;
                margin-bottom: 0px;
                border-radius: 0px;
            }
        """)

    def mousePressEvent(self, QMouseEvent):
        super(VideoSlider, self).mousePressEvent(QMouseEvent)
        self.usercontrolled = True
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        sr = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            opt,
            QtWidgets.QStyle.SubControl.SC_SliderHandle,
            self)
        if QMouseEvent.button() == QtCore.Qt.MouseButton.LeftButton and not sr.contains(QMouseEvent.pos()):
            if self.orientation() == QtCore.Qt.Orientation.Vertical:
                newVal = self.minimum() + ((self.maximum() - self.minimum()) * (self.height()-QMouseEvent.pos().y()))/self.height()
            else:
                newVal = self.minimum() + (self.maximum() - self.minimum()) * QMouseEvent.pos().x() / self.width()
            if self.invertedAppearance():
                self.setValue(int(self.maximum() - newVal))
            else:
                self.setValue(int(newVal))

    def mouseReleaseEvent(self, QMouseEvent):
        self.usercontrolled = False
        self.recentlyreleased = True
        super(VideoSlider, self).mouseReleaseEvent(QMouseEvent)


class SelectAllLineEdit(QtWidgets.QLineEdit):
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


class CropVideoPage(QtWidgets.QWizardPage):
    def __init__(self):
        super(CropVideoPage, self).__init__()
        self.vbox = QtWidgets.QVBoxLayout()
        self.solver = None
        self.video = None

        self.frame_rate = 25
        self.start_frame = 0
        self.end_frame = 1
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000 // self.frame_rate)
        self.scene = QtWidgets.QGraphicsScene()
        self.pixMap = None
        self.pixMapItem = None

        self.setLayout(self.vbox)

        self.vbox.addWidget(QtWidgets.QLabel('Press <i>Continue</i> to use the whole video or mark start and end of a custom video cut.'))

        graphics_view_widget = QtWidgets.QWidget()
        self.graphics_view = MyView(graphics_view_widget)
        self.graphics_view.setScene(self.scene)

        self.video_widget = QtWidgets.QWidget()
        self.video_layout = QtWidgets.QVBoxLayout()
        self.vbox.addLayout(self.video_layout)
        # self.video_widget.setLayout(self.video_layout)

        self.video_control_widget = QtWidgets.QWidget()
        self.video_control_layout = QtWidgets.QVBoxLayout()
        self.video_control_widget.setLayout(self.video_control_layout)

        self.video_control_buttons_widget = QtWidgets.QWidget()
        self.video_control_buttons_layout = QtWidgets.QHBoxLayout()
        self.video_control_buttons_widget.setLayout(self.video_control_buttons_layout)

        self.video_crop_buttons_widget = QtWidgets.QWidget()
        self.video_crop_buttons_layout = QtWidgets.QHBoxLayout()
        self.video_crop_buttons_widget.setLayout(self.video_crop_buttons_layout)

        self.video_label_widget = QtWidgets.QWidget()
        self.video_label_layout = QtWidgets.QHBoxLayout()
        self.video_label_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.video_label_layout.setSpacing(0)

        self.video_label_widget.setLayout(self.video_label_layout)

        self.video_layout.addWidget(self.graphics_view)
        self.video_layout.addWidget(self.video_crop_buttons_widget)
        self.video_control_buttons_widget.show()
        self.video_layout.addWidget(self.video_label_widget)
        self.video_layout.addWidget(self.video_control_widget)

        self.speedSlider = QtWidgets.QSlider()
        self.speedSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.speedSlider.setMinimum(1)
        self.speedSlider.setMaximum(120)

        self.backward = QtWidgets.QToolButton()
        self.backward.setText('back')
        self.backward.setShortcut(S_.controls.video_prev)
        self.playPause = QtWidgets.QToolButton()
        self.playPause.setText('play')
        self.playPause.setShortcut(S_.controls.video_play_pause)
        self.forward = QtWidgets.QToolButton()
        self.forward.setText('forward')
        self.forward.setShortcut(S_.controls.video_next)

        self.mark_start = QtWidgets.QPushButton('mark start')
        self.mark_stop = QtWidgets.QPushButton('mark stop')
        self.to_start = QtWidgets.QPushButton('go to start')
        self.to_stop = QtWidgets.QPushButton('go to stop')
        self.clear = QtWidgets.QPushButton('reset range')
        self.frameEdit = SelectAllLineEdit()
        self.frameEditValidator = QtGui.QIntValidator(0, 1)
        self.frameEdit.setValidator(self.frameEditValidator)
        self.showFrame = QtWidgets.QPushButton('go to frame')
        self.fpsLabel = QtWidgets.QLabel()
        self.fpsLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.start_frame_sign = QtWidgets.QLabel()
        self.start_frame_sign.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.end_frame_sign = QtWidgets.QLabel()
        self.end_frame_sign.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.num_frames_sign = QtWidgets.QLabel()
        self.update_start_end_num_labels()
        self.videoSlider = VideoSlider()
        self.videoSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.videoSlider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.videoSlider.setMaximumHeight(10)
        self.videoSlider.setMaximum(100)

        self.video_middle_label = QtWidgets.QLabel()
        self.video_middle_label.setStyleSheet("QLabel { background-color: white; }")
        self.video_middle_label.setFixedWidth(0)

        # TODO - set videoSlider width here

        self.video_middle_label.setFixedHeight(5)

        self.video_first_label = QtWidgets.QLabel()
        self.video_first_label.setStyleSheet("QLabel { background-color: yellow; }")
        self.video_first_label.setFixedWidth(0)
        self.video_first_label.setFixedHeight(5)

        self.video_second_label = QtWidgets.QLabel()
        self.video_second_label.setStyleSheet("QLabel { background-color: yellow; }")
        self.video_second_label.setFixedWidth(0)
        self.video_second_label.setFixedHeight(5)

        self.video_label_layout.addWidget(self.video_first_label)
        self.video_label_layout.addWidget(self.video_middle_label)
        self.video_label_layout.addWidget(self.video_second_label)

        self.video_control_layout.addWidget(self.videoSlider)
        self.video_control_layout.addWidget(self.video_control_buttons_widget)

        self.video_control_buttons_layout.addWidget(self.speedSlider)
        self.video_control_buttons_layout.addWidget(self.fpsLabel)
        self.video_control_buttons_layout.addStretch()
        self.video_control_buttons_layout.addWidget(self.backward)
        self.video_control_buttons_layout.addWidget(self.playPause)
        self.video_control_buttons_layout.addWidget(self.forward)
        self.video_control_buttons_layout.addStretch()
        self.video_control_buttons_layout.addWidget(self.frameEdit)
        self.video_control_buttons_layout.addWidget(self.showFrame)
        self.video_crop_buttons_layout.addWidget(self.mark_start)
        self.video_crop_buttons_layout.addWidget(self.mark_stop)
        self.video_crop_buttons_layout.addWidget(self.to_start)
        self.video_crop_buttons_layout.addWidget(self.to_stop)
        self.video_crop_buttons_layout.addWidget(self.clear)
        self.to_start.hide()
        self.to_stop.hide()
        self.clear.hide()
        self.video_crop_buttons_layout.addWidget(self.start_frame_sign)
        self.video_crop_buttons_layout.addWidget(self.end_frame_sign)
        self.video_crop_buttons_layout.addWidget(self.num_frames_sign)
        self.connect_ui()

        self.chunks = []
        self.markers = []
        self.items = []

    def initializePage(self):
        self.set_video(get_auto_video_manager(self.wizard().project))

    def validatePage(self):
        # super(CropVideoPage, self).validatePage()
        self.wizard().project.set_video_trim(self.start_frame, self.end_frame)
        return True

    def set_video(self, video_manager):
        self.video = video_manager
        end_frame_raw = self.video.video_frame_count_without_bounds() - 1
        self.start_frame = self.video.start_t
        self.end_frame = self.video.end_t
        self.videoSlider.setMaximum(end_frame_raw)
        self.update_start_end_num_labels()

        img = self.video.next_frame()
        assert img is not None
        self.pixMap = cvimg2qtpixmap(img)
        item = self.scene.addPixmap(self.pixMap)
        self.pixMapItem = item
        self.update_frame_number()

        self.speedSlider.setValue(int(self.video.fps()))
        self.speed_slider_changed()

        self.frameEditValidator.setRange(0, end_frame_raw)

    def sc_changed(self):
        self.video.crop_model = {'y1': self.sc_y1.value(),
                                 'y2': self.sc_y2.value(),
                                 'x1': self.sc_x1.value(),
                                 'x2': self.sc_x2.value()}
        frame = self.video.frame_number()
        self.video.reset()
        self.change_frame(frame)
        # self.load_next_frame()
        # self.load_previous_frame()

    def marker_changed(self):
        pass

    def speed_slider_changed(self):
        """Method invoked when value of slider controlling speed of video changed it's value"""
        self.frame_rate = self.speedSlider.value()
        self.timer.setInterval(1000 // self.frame_rate)
        self.fpsLabel.setText(str(self.frame_rate) + ' fps')

    def add_data(self, solver):
        self.solver = solver
        t_0_nodes = []

        self.chunks = []
        for n in self.solver.g.nodes():
            if n.frame_ == 0:
                t_0_nodes.append(n)

            for _, n2, d, in self.solver.g.out_edges(n, data=True):
                if 'chunk_ref' in d:
                    self.chunks.append((n, n2, d['chunk_ref']))

        i = 0
        for n, n2, ch in self.chunks:
            r, g, b = colors_[i % len(colors_)]
            item = markers.CenterMarker(0, 0, MARKER_SIZE, QtGui.QColor(r, g, b), i, self.marker_changed)
            item.setZValue(0.5)
            self.items.append(item)
            self.scene.addItem(item)

            if n.frame_ != 0:
                item.setVisible(False)

            i += 1

    def connect_ui(self):
        """Connects GUI elements to appropriate methods"""
        from functools import partial

        self.forward.clicked.connect(self.load_next_frame)
        self.backward.clicked.connect(self.load_previous_frame)
        self.playPause.clicked.connect(self.play_pause)
        self.speedSlider.valueChanged.connect(self.speed_slider_changed)
        self.frameEdit.returnPressed.connect(self.showFrame.click)
        self.showFrame.clicked.connect(partial(self.change_frame, None))
        self.videoSlider.valueChanged.connect(self.video_slider_changed)
        self.timer.timeout.connect(self.load_next_frame)

        self.mark_start.clicked.connect(self.get_start_frame_number)
        self.mark_stop.clicked.connect(self.get_end_frame_number)
        self.clear.clicked.connect(self.all_clear)
        self.to_start.clicked.connect(self.go_to_start)
        self.to_stop.clicked.connect(self.go_to_stop)

        # self.sc_y1.valueChanged.connect(self.sc_changed)
        # self.sc_x1.valueChanged.connect(self.sc_changed)
        # self.sc_y2.valueChanged.connect(self.sc_changed)
        # self.sc_x2.valueChanged.connect(self.sc_changed)

    def load_next_frame(self):
        """Loads next frame of the video and displays it. If there is no next frame, calls self.out_of_frames"""
        if self.video is not None:
            img = self.video.next_frame()
            if img is not None:
                if self.pixMapItem is not None:
                    self.scene.removeItem(self.pixMapItem)
                self.pixMap = cvimg2qtpixmap(img)
                view_add_bg_image(self.graphics_view, self.pixMap)
                item = self.scene.addPixmap(self.pixMap)
                self.pixMapItem = item
                self.update_frame_number()
            else:
                self.out_of_frames()

    def load_previous_frame(self):
        """Loads previous frame of the video if there is such and displays it"""
        if self.video is not None:
            img = self.video.previous_frame()
            if img is not None:
                if self.pixMapItem is not None:
                    self.scene.removeItem(self.pixMapItem)
                self.pixMap = cvimg2qtpixmap(img)
                view_add_bg_image(self.graphics_view, self.pixMap)
                item = self.scene.addPixmap(self.pixMap)
                self.pixMapItem = item
                self.update_frame_number()

    def play_pause(self):
        """Method of playPause button."""
        # settings = QSettings("Ants correction tool")
        if self.video is not None:
            if self.timer.isActive():
                self.timer.stop()
                self.playPause.setText("play")
                self.playPause.setShortcut(S_.controls.video_play_pause)
            else:
                self.timer.start()
                self.playPause.setText("pause")
                self.playPause.setShortcut(S_.controls.video_play_pause)

    def update_frame_number(self):
        """Updates values of components displaying frame number"""
        self.frameEdit.setText(str(self.video.frame_number()))
        self.videoSlider.setValue(self.video.frame_number())

    def out_of_frames(self):
        """Stops playing of the video if it is playing."""
        if self.timer.isActive():
            self.timer.stop()
            self.playPause.setText("play")

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

    def change_frame(self, position=None):
        """Changes current frame to position given. If there is no such position, calls self.out_of_frames"""
        if position is None:
            position = int(self.frameEdit.text())

        if self.video is not None:
            img = self.video.seek_frame(position)
            if img is not None:
                if self.pixMapItem is not None:
                    self.scene.removeItem(self.pixMapItem)
                self.pixMap = cvimg2qtpixmap(img)
                view_add_bg_image(self.graphics_view, self.pixMap)
                item = self.scene.addPixmap(self.pixMap)
                self.pixMapItem = item
                self.update_frame_number()
            else:
                self.out_of_frames()

    def get_start_frame_number(self):
        self.start_frame = self.video.frame_number()
        self.to_start.show()
        self.to_stop.show()
        self.clear.show()
        self.video_labels_repaint()

    def get_end_frame_number(self):
        self.end_frame = self.video.frame_number()
        self.to_start.show()
        self.to_stop.show()
        self.clear.show()
        self.video_labels_repaint()

    def video_labels_repaint(self):
        self.width = self.videoSlider.width()
        self.ratio = self.width / float(self.video.total_frame_count())

        self.first = int(self.ratio * self.start_frame)
        self.second = int(self.ratio * (self.video.total_frame_count() - self.end_frame))
        self.middle = int(self.width - self.first - self.second)

        if self.start_frame > self.end_frame:
            self.end_frame = self.video.total_frame_count() - 1
            self.update_start_end_num_labels()
            self.video_labels_repaint()
        else:
            self.video_first_label.setFixedWidth(self.first)
            self.video_middle_label.setFixedWidth(self.middle)
            self.video_second_label.setFixedWidth(self.second)

        self.update_start_end_num_labels()

    def update_start_end_num_labels(self):
        self.start_frame_sign.setText("start frame: {}".format(self.start_frame))
        self.end_frame_sign.setText("end frame: {}".format(self.end_frame))
        self.num_frames_sign.setText("frames in range: {}".format(self.end_frame - self.start_frame + 1))

    def go_to_start(self):
        self.change_frame(self.start_frame)

    def go_to_stop(self):
        self.change_frame(self.end_frame)

    def all_clear(self):
        self.start_frame = 0
        self.end_frame = self.video.total_frame_count() - 1
        self.to_start.hide()
        self.to_stop.hide()
        self.clear.hide()
        self.video_labels_repaint()


def view_add_bg_image(g_view, pix_map):
    gv_w = g_view.geometry().width()
    gv_h = g_view.geometry().height()
    im_w = pix_map.width()
    im_h = pix_map.height()

    m11 = g_view.transform().m11()
    m22 = g_view.transform().m22()

    if m11 and m22 == 1:
        if gv_w / float(im_w) <= gv_h / float(im_h):
            val = math.floor((gv_w / float(im_w))*100) / 100
            g_view.scale(val, val)
        else:
            val = math.floor((gv_h / float(im_h))*100) / 100
            g_view.scale(val, val)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    project = Project()
    project.load('/home/matej/prace/ferda/projects/1_initial_projects_180808_diverse/Cam1_clip (copy)')

    widget = CropVideoPage()
    widget.set_video(get_auto_video_manager(project))
    widget.showMaximized()

    app.exec()
    app.deleteLater()
    sys.exit()
