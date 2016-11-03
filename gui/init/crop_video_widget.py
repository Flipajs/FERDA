__author__ = 'flipajs'

import sys
import math

from PyQt4 import QtGui, QtCore

from gui.img_controls.my_view import MyView
from utils.video_manager import get_auto_video_manager
from gui.img_controls.gui_utils import cvimg2qtpixmap
from gui.img_controls import markers
from core.animal import colors_
from core.settings import Settings as S_
from core.project.project import Project

MARKER_SIZE = 15

class VideoSlider(QtGui.QSlider):
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
        opt = QtGui.QStyleOptionSlider()
        self.initStyleOption(opt)
        sr = self.style().subControlRect(QtGui.QStyle.CC_Slider, opt, QtGui.QStyle.SC_SliderHandle, self)

        if QMouseEvent.button() == QtCore.Qt.LeftButton and not sr.contains(QMouseEvent.pos()):
            if self.orientation() == QtCore.Qt.Vertical:
                newVal = self.minimum() + ((self.maximum() - self.minimum()) * (self.height()-QMouseEvent.y()))/self.height()
            else:
                newVal = self.minimum() + (self.maximum() - self.minimum()) * QMouseEvent.x() / self.width()
            if self.invertedAppearance():
                self.setValue(self.maximum() - newVal)
            else:
                self.setValue(newVal)

    def mouseReleaseEvent(self, QMouseEvent):
        self.usercontrolled = False
        self.recentlyreleased = True
        super(VideoSlider, self).mouseReleaseEvent(QMouseEvent)


class SelectAllLineEdit(QtGui.QLineEdit):
    def __init__(self, parent=None):
        super(SelectAllLineEdit, self).__init__(parent)
        self.readyToEdit = True
        self.setFixedHeight(15)

    def mousePressEvent(self, e, Parent=None):
        super(SelectAllLineEdit, self).mousePressEvent(e) #required to deselect on 2e click
        if self.readyToEdit:
            self.selectAll()
            self.readyToEdit = False

    def focusOutEvent(self, e):
        super(SelectAllLineEdit, self).focusOutEvent(e) #required to remove cursor on focusOut
        self.deselect()
        self.readyToEdit = True


class CropVideoWidget(QtGui.QWidget):
    def __init__(self, project):
        super(CropVideoWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        self.solver = None
        self.project = project
        self.video = get_auto_video_manager(project)

        self.frame_rate = 30
        self.start_frame = 1
        self.end_frame = self.video.total_frame_count()
        self.toggle_borders_bool = False
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000 / self.frame_rate)
        self.scene = QtGui.QGraphicsScene()
        self.pixMap = None
        self.pixMapItem = None

        self.setLayout(self.vbox)

        graphics_view_widget = QtGui.QWidget()
        self.graphics_view = MyView(graphics_view_widget)
        self.graphics_view.setScene(self.scene)

        self.video_widget = QtGui.QWidget()
        self.video_layout = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.video_layout)
        self.video_widget.setLayout(self.video_layout)

        self.video_control_widget = QtGui.QWidget()
        self.video_control_layout = QtGui.QVBoxLayout()
        self.video_control_widget.setLayout(self.video_control_layout)
        self.video_control_widget.setMaximumHeight(70)

        self.video_control_buttons_widget = QtGui.QWidget()
        self.video_control_buttons_layout = QtGui.QHBoxLayout()
        self.video_control_buttons_widget.setLayout(self.video_control_buttons_layout)

        self.video_crop_buttons_widget = QtGui.QWidget()
        self.video_crop_buttons_layout = QtGui.QHBoxLayout()
        self.video_crop_buttons_widget.setLayout(self.video_crop_buttons_layout)

        self.video_label_widget = QtGui.QWidget()
        self.video_label_layout = QtGui.QHBoxLayout()
        self.video_label_layout.setAlignment(QtCore.Qt.AlignLeft)
        self.video_label_layout.setSpacing(0)

        self.video_label_widget.setLayout(self.video_label_layout)

        self.video_layout.addWidget(self.graphics_view)
        self.video_layout.addWidget(self.video_crop_buttons_widget)
        self.video_control_buttons_widget.show()
        self.video_layout.addWidget(self.video_label_widget)
        self.video_layout.addWidget(self.video_control_widget)

        self.speedSlider = QtGui.QSlider()
        self.speedSlider.setOrientation(QtCore.Qt.Horizontal)
        self.speedSlider.setMinimum(0)
        self.speedSlider.setMaximum(99)

        self.backward = QtGui.QPushButton('Back')
        self.backward.setShortcut(S_.controls.video_prev)
        self.playPause = QtGui.QPushButton('Play')
        self.playPause.setShortcut(S_.controls.video_play_pause)
        self.forward = QtGui.QPushButton('Forward')
        self.forward.setShortcut(S_.controls.video_next)
        self.mark_start = QtGui.QPushButton('Mark start')
        # self.mark_start.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_S))
        self.mark_stop = QtGui.QPushButton('Mark stop')
        # self.mark_stop.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_D))
        self.to_start = QtGui.QPushButton('Go to start')
        # self.to_start.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_0))
        self.to_stop = QtGui.QPushButton('Go to stop')
        # self.to_start.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_1))
        self.clear = QtGui.QPushButton('Clear')
        # self.to_start.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_1))
        self.toggle_borders = QtGui.QPushButton('Toggle borders')
        # self.to_start.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_1))
        self.frameEdit = SelectAllLineEdit()
        self.showFrame = QtGui.QPushButton('Show')
        self.fpsLabel = QtGui.QLabel()
        self.fpsLabel.setAlignment(QtCore.Qt.AlignRight)
        self.start_frame_sign = QtGui.QLabel()
        self.start_frame_sign.setAlignment(QtCore.Qt.AlignCenter)
        self.start_frame_sign.setText("Start Frame: " + str(self.start_frame))
        self.end_frame_sign = QtGui.QLabel()
        self.end_frame_sign.setAlignment(QtCore.Qt.AlignCenter)
        self.end_frame_sign.setText("End Frame: " + str(self.end_frame))
        self.videoSlider = VideoSlider()
        self.videoSlider.setOrientation(QtCore.Qt.Horizontal)
        self.videoSlider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.videoSlider.setMaximumHeight(10)
        self.videoSlider.setMaximum(self.video.total_frame_count())

        self.video_middle_label = QtGui.QLabel()
        self.video_middle_label.setStyleSheet("QLabel { background-color: white; }")
        self.video_middle_label.setFixedWidth(0)

        # TODO - set videoSlider width here

        self.video_middle_label.setFixedHeight(5)

        self.video_first_label = QtGui.QLabel()
        self.video_first_label.setStyleSheet("QLabel { background-color: yellow; }")
        self.video_first_label.setFixedWidth(0)
        self.video_first_label.setFixedHeight(5)

        self.video_second_label = QtGui.QLabel()
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
        self.video_control_buttons_layout.addWidget(self.backward)
        self.video_control_buttons_layout.addWidget(self.playPause)
        self.video_control_buttons_layout.addWidget(self.forward)
        self.video_control_buttons_layout.addWidget(self.showFrame)
        self.video_control_buttons_layout.addWidget(self.frameEdit)
        self.video_crop_buttons_layout.addWidget(self.mark_start)
        self.video_crop_buttons_layout.addWidget(self.mark_stop)
        self.video_crop_buttons_layout.addWidget(self.to_start)
        self.video_crop_buttons_layout.addWidget(self.to_stop)
        self.video_crop_buttons_layout.addWidget(self.clear)
        self.to_start.hide()
        self.to_stop.hide()
        self.toggle_borders.hide()
        self.clear.hide()
        self.video_crop_buttons_layout.addWidget(self.toggle_borders)
        self.video_crop_buttons_layout.addWidget(self.start_frame_sign)
        self.video_crop_buttons_layout.addWidget(self.end_frame_sign)

        self.connect_GUI()

        img = self.video.next_frame()

        if img is not None:
            self.pixMap = cvimg2qtpixmap(img)
            item = self.scene.addPixmap(self.pixMap)
            self.pixMapItem = item
            self.update_frame_number()

        self.chunks = []
        self.markers = []
        self.items = []

    def marker_changed(self):
        pass

    def init_speed_slider(self):
        """Initiates components associated with speed of viewing videos"""
        self.speedSlider.setValue(self.frame_rate)
        self.timer.setInterval(1000 / self.frame_rate)
        self.fpsLabel.setText(str(self.frame_rate) + ' fps')
        self.speedSlider.setMinimum(1)
        self.speedSlider.setMaximum(120)

    def speed_slider_changed(self):
        """Method invoked when value of slider controlling speed of video changed it's value"""
        self.frame_rate = self.speedSlider.value()
        self.timer.setInterval(1000 / self.frame_rate)
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

    def connect_GUI(self):
        """Connects GUI elements to appropriate methods"""
        self.forward.clicked.connect(self.load_next_frame)
        self.backward.clicked.connect(self.load_previous_frame)
        self.playPause.clicked.connect(self.play_pause)
        self.speedSlider.valueChanged.connect(self.speed_slider_changed)
        # self.showFrame.clicked.connect(self.show_frame)
        self.videoSlider.valueChanged.connect(self.video_slider_changed)
        self.timer.timeout.connect(self.load_next_frame)

        self.mark_start.clicked.connect(self.get_start_frame_number)
        self.mark_stop.clicked.connect(self.get_end_frame_number)
        self.clear.clicked.connect(self.all_clear)
        self.toggle_borders.clicked.connect(self.set_borders_action)
        self.to_start.clicked.connect(self.go_to_start)
        self.to_stop.clicked.connect(self.go_to_stop)


    def load_next_frame(self):
        """Loads next frame of the video and displays it. If there is no next frame, calls self.out_of_frames"""
        if self.toggle_borders_bool:
            if(self.video.frame_number() + 2 == self.end_frame or \
               self.video.frame_number() + 2 == self.start_frame):
                self.play_pause()
            # elif(self.video.frame_number() + 1  > self.end_frame) or\
            #     (self.video.frame_number() + 1  < self.start_frame):
            #     self.change_frame(self.start_frame)

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
        self.frameEdit.setText(str(int(self.video.frame_number() + 1)) + '/' + str(self.video.total_frame_count()))
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

    def change_frame(self, position):
        """Changes current frame to position given. If there is no such position, calls self.out_of_frames"""
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
        self.start_frame = self.video.frame_number() + 1
        self.to_start.show()
        self.to_stop.show()
        self.toggle_borders.show()
        self.clear.show()
        self.video_labels_repaint()

    def get_end_frame_number(self):
        self.end_frame = self.video.frame_number() + 1
        self.to_start.show()
        self.to_stop.show()
        self.toggle_borders.show()
        self.clear.show()
        self.video_labels_repaint()

    def video_labels_repaint(self):

        self.width = self.videoSlider.width()
        self.ratio = self.width / float(self.video.total_frame_count())

        self.first = self.ratio * self.start_frame
        self.second = self.ratio * (self.video.total_frame_count() - self.end_frame)
        self.middle = self.width - self.first - self.second

        if self.start_frame > self.end_frame:
            self.end_frame = self.video.total_frame_count()
            self.video_signs_repaint()
            self.video_labels_repaint()
        else:
            self.video_first_label.setFixedWidth(self.first)
            self.video_middle_label.setFixedWidth(self.middle)
            self.video_second_label.setFixedWidth(self.second)

        self.video_signs_repaint()

    def video_signs_repaint(self):
        self.start_frame_sign.setText("Start Frame: " + str(self.start_frame))
        self.end_frame_sign.setText("End Frame: " + str(self.end_frame))

    def go_to_start(self):
        self.change_frame(self.start_frame)

    def go_to_stop(self):
        self.change_frame(self.end_frame)

    def all_clear(self):
        self.start_frame = 0
        self.end_frame = self.video.total_frame_count()
        self.to_start.hide()
        self.to_stop.hide()
        self.toggle_borders.hide()
        self.clear.hide()
        self.video_labels_repaint()

    def set_borders_action(self):
        self.toggle_borders_bool = False if self.toggle_borders_bool else True
        self.toggle_borders.setText("Untoggle borders" if self.toggle_borders_bool else "Toggle Borders")

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
    app = QtGui.QApplication(sys.argv)
    project = Project()
    project.load('/home/simon/Documents/res/c3_0h30/c3_0h30.fproj')

    ex = CropVideoWidget(project)
    ex.showMaximized()
    print "test"

    app.exec_()
    app.deleteLater()
    sys.exit()