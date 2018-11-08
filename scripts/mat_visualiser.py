from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from builtins import str
from builtins import range
from past.utils import old_div
import math

from PyQt4 import QtGui, QtCore

from core.animal import colors_
from gui.img_controls import markers
from gui.img_controls.gui_utils import cvimg2qtpixmap
from gui.img_controls.my_view import MyView
from gui.settings import Settings as S_
from utils.video_manager import get_auto_video_manager

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
                newVal = self.minimum() + old_div(((self.maximum() - self.minimum()) * (self.height()-QMouseEvent.y())),self.height())
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


class ResultsWidget(QtGui.QWidget):
    def __init__(self, project, start_on_frame=-1):
        super(ResultsWidget, self).__init__()

        self.vbox = QtGui.QVBoxLayout()
        self.solver = None
        self.project = project
        self.video = get_auto_video_manager(project)

        self.frame_rate = 30
        self.timer = QtCore.QTimer()
        self.timer.setInterval(old_div(1000, self.frame_rate))
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
        # self.video_widget.setLayout(self.video_layout)

        self.video_control_widget = QtGui.QWidget()
        self.video_control_layout = QtGui.QVBoxLayout()
        self.video_control_widget.setLayout(self.video_control_layout)
        self.video_control_widget.setMaximumHeight(70)
        self.video_control_widget.setContentsMargins(0, 0, 0, 0)

        self.video_control_buttons_widget = QtGui.QWidget()
        self.video_control_buttons_layout = QtGui.QHBoxLayout()
        self.video_control_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.video_control_buttons_widget.setLayout(self.video_control_buttons_layout)

        self.video_layout.addWidget(self.graphics_view)
        self.video_layout.addWidget(self.video_control_widget)

        self.speedSlider = QtGui.QSlider()
        self.speedSlider.setOrientation(QtCore.Qt.Horizontal)
        self.speedSlider.setMinimum(0)
        self.speedSlider.setMaximum(99)

        self.backward = QtGui.QPushButton('back')
        self.backward.setShortcut(S_.controls.video_prev)
        self.playPause = QtGui.QPushButton('play')
        self.playPause.setShortcut(S_.controls.video_play_pause)
        self.forward = QtGui.QPushButton('forward')
        self.forward.setShortcut(S_.controls.video_next)
        self.frameEdit = SelectAllLineEdit()
        self.showFrame = QtGui.QPushButton('show')
        self.fpsLabel = QtGui.QLabel()
        self.fpsLabel.setAlignment(QtCore.Qt.AlignRight)
        self.videoSlider = VideoSlider()
        self.videoSlider.setOrientation(QtCore.Qt.Horizontal)
        self.videoSlider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.videoSlider.setMaximumHeight(15)
        self.videoSlider.setMaximum(self.video.total_frame_count())

        self.video_control_layout.addWidget(self.videoSlider)
        self.video_control_layout.addWidget(self.video_control_buttons_widget)

        self.frame_jump_button = QtGui.QPushButton('jump')
        self.frame_jump_button.clicked.connect(self.frame_jump)

        self.video_control_buttons_layout.addWidget(self.speedSlider)
        self.video_control_buttons_layout.addWidget(self.fpsLabel)
        self.video_control_buttons_layout.addWidget(self.backward)
        self.video_control_buttons_layout.addWidget(self.playPause)
        self.video_control_buttons_layout.addWidget(self.forward)
        self.video_control_buttons_layout.addWidget(self.showFrame)
        self.video_control_buttons_layout.addWidget(self.frameEdit)
        self.video_control_buttons_layout.addWidget(self.frame_jump_button)

        self.setTabOrder(self.frameEdit, self.frame_jump_button)

        self.connect_GUI()

        img = self.video.next_frame()

        if img is not None:
            self.pixMap = cvimg2qtpixmap(img)
            item = self.scene.addPixmap(self.pixMap)
            self.pixMapItem = item
            self.update_frame_number()

        self.chunks = []
        self.starting_frames = {}
        self.markers = []
        self.items = []

        self.active_markers = []

        self.highlight_marker = None
        self.highlight_timer = QtCore.QTimer()
        self.highlight_timer.timeout.connect(self.decrease_highlight_marker_opacity)

        self.highlight_marker2nd = None
        self.highlight_marker2nd_frame = -1
        self.highlight_timer2nd = QtCore.QTimer()
        from functools import partial
        self.highlight_timer2nd.timeout.connect(partial(self.decrease_highlight_marker_opacity, True))


    def frame_jump(self):
        f = int(self.frameEdit.text())
        self.change_frame(f)

    def decrease_highlight_marker_opacity(self, second=False):
        if second:
            op = self.highlight_marker2nd.opacity()

            dec_fact = 0.02
            if op > 0:
                if op < 0.2:
                    dec_fact = 0.04

                self.highlight_marker2nd.setOpacity(op - dec_fact)

                # self.highlight_marker.setVisible(False)
                # self.highlight_marker = markers.CenterMarker(0, 0, radius, QtGui.QColor(255, 255, 0), 0, self.marker_changed)
                # self.highlight_marker.setOpacity(0.40)
                # self.highlight_marker.setPos(centroid[1]-radius/2, centroid[0]-radius/2)
                # self.scene.addItem(self.highlight_marker)
            else:
                self.highlight_timer2nd.stop()
        else:
            op = self.highlight_marker.opacity()

            dec_fact = 0.02
            if op > 0:
                if op < 0.2:
                    dec_fact = 0.04

                self.highlight_marker.setOpacity(op - dec_fact)

                # self.highlight_marker.setVisible(False)
                # self.highlight_marker = markers.CenterMarker(0, 0, radius, QtGui.QColor(255, 255, 0), 0, self.marker_changed)
                # self.highlight_marker.setOpacity(0.40)
                # self.highlight_marker.setPos(centroid[1]-radius/2, centroid[0]-radius/2)
                # self.scene.addItem(self.highlight_marker)
            else:
                self.highlight_timer.stop()

    @staticmethod
    def marker_changed(id_):
        print(id_)

    def update_marker_position(self, marker, c):
        sf = self.project.other_parameters.img_subsample_factor
        if sf > 1.0:
            c[0] *= sf
            c[1] *= sf

        marker.setVisible(True)
        marker.setPos(c[1] - old_div(MARKER_SIZE, 2), c[0] - old_div(MARKER_SIZE,2))

    def highlight_area(self, data, radius=50):
        centroid = data['n1'].centroid()
        self.highlight_marker = markers.CenterMarker(0, 0, radius, QtGui.QColor(167, 255, 36), 0, print_id)
        self.highlight_marker.setOpacity(0.40)
        self.highlight_marker.setPos(centroid[1]-old_div(radius,2), centroid[0]-old_div(radius,2))
        self.scene.addItem(self.highlight_marker)
        self.highlight_timer.start(50)

        if data['n2']:
            self.highlight_marker2nd = markers.CenterMarker(0, 0, radius, QtGui.QColor(36, 255, 167), 0, print_id)
            self.highlight_marker2nd.setOpacity(0.40)
            centroid = data['n2'].centroid()
            self.highlight_marker2nd.setPos(centroid[1]-old_div(radius,2), centroid[0]-old_div(radius,2))
            self.highlight_marker2nd_frame = data['n2'].frame_

    def update_positions_optimized(self, frame):
        new_active_markers = []
        for m_id, ch in self.active_markers:
            if frame == ch.end_n.frame_ + 1:
                self.items[m_id].setVisible(False)
            else:
                new_active_markers.append((m_id, ch))
                c = ch.get_centroid_in_time(frame).copy()
                self.update_marker_position(self.items[m_id], c)

        self.active_markers = new_active_markers

        if frame in self.starting_frames:
            for ch, m_id in self.starting_frames[frame]:
                c = ch.get_centroid_in_time(frame).copy()
                self.update_marker_position(self.items[m_id], c)
                self.active_markers.append((m_id, ch))

    def update_positions(self, frame, optimized=True):
        if optimized:
            self.update_positions_optimized(frame)
            return

        self.active_markers = []

        i = 0
        for ch in self.chunks:
            c = ch.get_centroid_in_time(frame)

            if c is None:
                self.items[i].setVisible(False)
            else:
                c = c.copy()
                self.update_marker_position(self.items[i], c)
                self.active_markers.append((i, ch))

            i += 1

    def init_speed_slider(self):
        """Initiates components associated with speed of viewing videos"""
        self.speedSlider.setValue(self.frame_rate)
        self.timer.setInterval(old_div(1000, self.frame_rate))
        self.fpsLabel.setText(str(self.frame_rate) + ' fps')
        self.speedSlider.setMinimum(1)
        self.speedSlider.setMaximum(120)

    def speed_slider_changed(self):
        """Method invoked when value of slider controlling speed of video changed it's value"""
        self.frame_rate = self.speedSlider.value()
        self.timer.setInterval(old_div(1000, self.frame_rate))
        self.fpsLabel.setText(str(self.frame_rate) + ' fps')

    def add_data(self, solver, just_around_frame=-1, margin=1000):
        self.solver = solver
        self.chunks = self.solver.chunk_list()
        i = 0

        t1 = just_around_frame - margin
        t2 = just_around_frame + margin

        if just_around_frame > -1:
            chs = []
            for ch in self.chunks:
                if t1 < ch.start_t() < t2 or t1 < ch.end_t() < t2:
                    r, g, b = colors_[i % len(colors_)]
                    item = markers.CenterMarker(0, 0, MARKER_SIZE, QtGui.QColor(r, g, b), ch.id_, print_id)
                    item.setZValue(0.5)
                    self.items.append(item)
                    self.scene.addItem(item)

                    self.starting_frames.setdefault(ch.start_n.frame_, []).append((ch, i))
                    # if ch.start_n.frame_ != 0:

                    item.setVisible(False)

                    chs.append(ch)
                    i += 1

            self.chunks = chs
        else:
            for ch in self.chunks:
                r, g, b = colors_[i % len(colors_)]
                item = markers.CenterMarker(0, 0, MARKER_SIZE, QtGui.QColor(r, g, b), ch.id_, print_id)
                item.setZValue(0.5)
                self.items.append(item)
                self.scene.addItem(item)

                self.starting_frames.setdefault(ch.start_n.frame_, []).append((ch, i))
                # if ch.start_n.frame_ != 0:

                item.setVisible(False)

                i += 1

        self.update_positions(0, optimized=False)

    def add_data_mat(self, chunks):
        i = 0
        self.chunks = chunks
        for ch in self.chunks:
            r, g, b = colors_[i % len(colors_)]
            item = markers.CenterMarker(0, 0, MARKER_SIZE, QtGui.QColor(r, g, b), ch.id_, print_id)
            item.setZValue(0.5)
            self.items.append(item)
            self.scene.addItem(item)

            self.starting_frames.setdefault(ch.start_n.frame_, []).append((ch, i))

            item.setVisible(False)

            i += 1

        self.update_positions(0, optimized=False)

    def connect_GUI(self):
        """Connects GUI elements to appropriate methods"""
        self.forward.clicked.connect(self.load_next_frame)
        self.backward.clicked.connect(self.load_previous_frame)
        self.playPause.clicked.connect(self.play_pause)
        self.speedSlider.valueChanged.connect(self.speed_slider_changed)
        # self.showFrame.clicked.connect(self.show_frame)
        self.videoSlider.valueChanged.connect(self.video_slider_changed)
        self.timer.timeout.connect(self.load_next_frame)

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
                self.update_positions(self.video.frame_number())
            else:
                self.out_of_frames()

        if self.video.frame_number() == self.highlight_marker2nd_frame:
            print("SHOW")
            self.scene.addItem(self.highlight_marker2nd)
            self.highlight_timer2nd.start(50)
            self.highlight_marker2nd_frame = -1

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
                self.update_positions(self.video.frame_number(), optimized=False)

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
                self.update_positions(self.video.frame_number(), optimized=False)
            else:
                self.out_of_frames()


def view_add_bg_image(g_view, pix_map):
    gv_w = g_view.geometry().width()
    gv_h = g_view.geometry().height()
    im_w = pix_map.width()
    im_h = pix_map.height()

    m11 = g_view.transform().m11()
    m22 = g_view.transform().m22()

    if m11 and m22 == 1:
        if old_div(gv_w, float(im_w)) <= old_div(gv_h, float(im_h)):
            val = old_div(math.floor((old_div(gv_w, float(im_w)))*100), 100)
            g_view.scale(val, val)
        else:
            val = old_div(math.floor((old_div(gv_h, float(im_h)))*100), 100)
            g_view.scale(val, val)

def get_chunks_from_mat(mat):
    from core.graph.chunk import Chunk
    from core.region.region import Region
    import numpy as np
    from core.graph.reduced import Reduced

    chunks = []
    for ch in mat['FERDA'][0]:
        new_ch = Chunk()

        if len(ch['x']) > 0:
            start_n = Region()
            y = ch['y'][0][0][0]
            x = ch['x'][0][0][0]
            frame = ch['frame'][0][0][0]

            start_n.centroid_ = np.array([y[0], x[0]])
            start_n.frame_ = frame[0]
            new_ch.start_n = start_n

            for i in range(1, len(x)):
                r = Reduced()
                r.centroid_ = np.array([y[i], x[i]])
                r.frame_ = frame[i]
                new_ch.reduced.append(r)

            end_n = Region()
            end_n.centroid_ = np.array([y[-1], x[-1]])
            end_n.frame_ = frame [-1]
            new_ch.end_n = end_n

        chunks.append(new_ch)

    return chunks

def print_id(id_):
    print(id_)

if __name__ == "__main__":
    import sys
    from core.project.project import Project
    import scipy.io as sio

    # CHANGE THIS:
    project_path = '/Users/flipajs/Documents/wd/eight_22/eight22.fproj'
    mat_path = '/Users/flipajs/Documents/wd/eight_22/out.mat'

    app = QtGui.QApplication(sys.argv)

    p = Project()
    p.load(project_path)

    ex = ResultsWidget(p)

    mat = sio.loadmat(mat_path)

    chunks = get_chunks_from_mat(mat)
    ex.add_data_mat(chunks)
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()