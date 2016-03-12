__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore
from gui.img_controls.my_view import MyView
from utils.video_manager import get_auto_video_manager
from gui.img_controls.utils import cvimg2qtpixmap
import math
import cv2
from viewer.gui.img_controls import markers
from core.animal import colors_
from core.settings import Settings as S_
from core.graph.region_chunk import RegionChunk
import numpy as np
from video_slider import VideoSlider
from select_all_line_edit import SelectAllLineEdit


MARKER_SIZE = 15

class ResultsWidget(QtGui.QWidget):
    def __init__(self, project, start_on_frame=-1):
        super(ResultsWidget, self).__init__()

        self.show_identities = True

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.setContentsMargins(0, 0, 0, 0)
        self.right_vbox = QtGui.QVBoxLayout()
        self.right_vbox.setContentsMargins(0, 0, 0, 0)
        self.solver = None
        self.project = project
        self.video = get_auto_video_manager(project)

        self.frame_rate = 30
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000 / self.frame_rate)
        self.scene = QtGui.QGraphicsScene()
        self.pixMap = None
        self.pixMapItem = None

        self.setLayout(self.hbox)
        self.splitter = QtGui.QSplitter()

        if self.show_identities:
            self.scroll_ = QtGui.QScrollArea()
            self.scroll_.setWidgetResizable(True)
            from gui.correction.identities_widget import IdentitiesWidget
            self.identities_widget = IdentitiesWidget(self.project)
            self.identities_widget.setMinimumWidth(200)
            self.scroll_.setWidget(self.identities_widget)

            self.splitter.addWidget(self.scroll_)

        self.right_w = QtGui.QWidget()
        self.right_w.setLayout(self.right_vbox)
        self.splitter.addWidget(self.right_w)

        self.hbox.addWidget(self.splitter)

        graphics_view_widget = QtGui.QWidget()
        self.graphics_view = MyView(graphics_view_widget)
        self.graphics_view.setScene(self.scene)

        self.video_widget = QtGui.QWidget()
        self.video_layout = QtGui.QVBoxLayout()
        self.right_vbox.addLayout(self.video_layout)

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
        self.init_speed_slider()

        self.backward = QtGui.QPushButton('back')
        self.backward.setShortcut(S_.controls.video_prev)
        self.playPause = QtGui.QPushButton('play')
        self.playPause.setShortcut(S_.controls.video_play_pause)
        self.forward = QtGui.QPushButton('forward')
        self.forward.setShortcut(S_.controls.video_next)
        self.frameEdit = SelectAllLineEdit()
        self.frameEdit.returnPressed.connect(self.frame_jump)
        self.frameEdit.setFixedHeight(30)
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

        self.frame_jump_button.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.visu_controls_layout = QtGui.QHBoxLayout()
        self.video_control_buttons_layout.addLayout(self.visu_controls_layout)
        self.video_control_buttons_layout.addWidget(self.speedSlider)

        self.video_control_buttons_layout.addWidget(self.speedSlider)
        self.video_control_buttons_layout.addWidget(self.fpsLabel)
        self.video_control_buttons_layout.addWidget(self.backward)
        self.video_control_buttons_layout.addWidget(self.playPause)
        self.video_control_buttons_layout.addWidget(self.forward)
        self.video_control_buttons_layout.addWidget(self.showFrame)
        self.video_control_buttons_layout.addWidget(self.frameEdit)
        self.video_control_buttons_layout.addWidget(self.frame_jump_button)

        self.reset_colors_b = QtGui.QPushButton('reset colors')
        self.reset_colors_b.clicked.connect(self.reset_colors)
        self.video_control_buttons_layout.addWidget(self.reset_colors_b)

        self.reset_colors_action = QtGui.QAction('reset_colors', self)
        self.reset_colors_action.triggered.connect(self.reset_colors)
        self.reset_colors_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Control + QtCore.Qt.Key_0))
        self.addAction(self.reset_colors_action)

        self.setTabOrder(self.frameEdit, self.frame_jump_button)

        self.show_filled_ch = QtGui.QCheckBox('show filled')
        self.show_filled_ch.setChecked(False)
        # lambda is used because if only self.update_position is given, it will give it parameters...
        self.show_filled_ch.stateChanged.connect(lambda x: self.update_positions())
        self.visu_controls_layout.addWidget(self.show_filled_ch)

        self.show_contour_ch = QtGui.QCheckBox('contours')
        self.show_contour_ch.setChecked(True)
        # lambda is used because if only self.update_position is given, it will give it parameters...
        self.show_contour_ch.stateChanged.connect(lambda x: self.update_positions())
        self.visu_controls_layout.addWidget(self.show_contour_ch)

        self.show_gt_markers = QtGui.QCheckBox('gt markers')
        self.show_gt_markers.setChecked(True)
        self.show_gt_markers.stateChanged.connect(lambda x: self.update_positions())
        self.visu_controls_layout.addWidget(self.show_gt_markers)

        self.show_staurated_ch = QtGui.QCheckBox('img saturated')
        self.show_staurated_ch.setChecked(False)
        self.show_staurated_ch.stateChanged.connect(lambda x: self.update_positions())
        self.visu_controls_layout.addWidget(self.show_staurated_ch)
        # self.visu_controls_layout.addWidget(QtGui.QLabel('markers:'))
        # self.show_markers_ch = QtGui.QCheckBox()
        # self.show_markers_ch.setChecked(False)
        # # lambda is used because if only self.update_position is given, it will give it parameters...
        # self.show_markers_ch.stateChanged.connect(lambda x: self.update_positions())
        # self.visu_controls_layout.addWidget(self.show_markers_ch)

        self.connect_GUI()

        self.video.next_frame()
        #
        # if img is not None:
        #     self.pixMap = cvimg2qtpixmap(img)
        #     item = self.scene.addPixmap(self.pixMap)
        #     self.pixMapItem = item
        #     self.update_frame_number()

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

        self.colormarks_items = []

        self.one_frame_items = []

        self.alpha_contour = 240
        self.alpha_filled = 120

        self.gitems = {'gt_markers': []}

        self.colors_ = [
                QtGui.QColor().fromRgbF(0, 0, 1), #
                QtGui.QColor().fromRgbF(1, 0, 0),
                QtGui.QColor().fromRgbF(1, 1, 0),
                QtGui.QColor().fromRgbF(0, 1, 0), #
                QtGui.QColor().fromRgbF(0, 1, 1),
                QtGui.QColor().fromRgbF(1, 1, 1)
            ]

        self.update_positions()
        # self.graphics_view.set

    def test_print_(self):
        print "TEST"

    def draw_region(self, r, animal_id, use_ch_color=None, alpha=120):
        from utils.img import get_cropped_pts

        pts_, roi = get_cropped_pts(r, return_roi=True, only_contour=False if self.show_filled_ch.isChecked() else True)
        offset = roi.top_left_corner()

        qim_ = QtGui.QImage(roi.width(), roi.height(), QtGui.QImage.Format_ARGB32)
        qim_.fill(QtGui.qRgba(0, 0, 0, 0))

        c = QtGui.qRgba(180, 180, 180, alpha)
        if use_ch_color:
            c = QtGui.qRgba(use_ch_color.red(), use_ch_color.green(), use_ch_color.blue(), alpha)
        else:
            if animal_id > -1:
                c_ = self.colors_[animal_id]
                c = QtGui.qRgba(c_.red(), c_.green(), c_.blue(), alpha)

        for i in range(pts_.shape[0]):
            qim_.setPixel(pts_[i, 1], pts_[i, 0], c)

        self.one_frame_items.append(self.scene.addPixmap(QtGui.QPixmap.fromImage(qim_)))
        self.one_frame_items[-1].setPos(offset[1], offset[0])
        self.one_frame_items[-1].setZValue(0.6)

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
            else:
                self.highlight_timer2nd.stop()
        else:
            op = self.highlight_marker.opacity()

            dec_fact = 0.02
            if op > 0:
                if op < 0.2:
                    dec_fact = 0.04

                self.highlight_marker.setOpacity(op - dec_fact)
            else:
                self.highlight_timer.stop()

    def marker_changed(self, id_):
        from core.graph.region_chunk import RegionChunk

        ch = self.project.chm[id_]
        rch = RegionChunk(ch, self.project.gm, self.project.rm)
        f = self.video.frame_number()

        print id_, rch.region_in_t(f)

    def update_marker_position(self, marker, c):
        sf = self.project.other_parameters.img_subsample_factor
        if sf > 1.0:
            c[0] *= sf
            c[1] *= sf

        if self.show_markers_ch.isChecked():
            marker.setVisible(True)

        marker.setPos(c[1] - MARKER_SIZE / 2, c[0] - MARKER_SIZE/2)

    def highlight_area(self, data, radius=50):
        centroid = data['n1'].centroid()
        self.highlight_marker = markers.CenterMarker(0, 0, radius, QtGui.QColor(167, 255, 36), 0, self.marker_changed)
        self.highlight_marker.setOpacity(0.40)
        self.highlight_marker.setPos(centroid[1]-radius/2, centroid[0]-radius/2)
        self.scene.addItem(self.highlight_marker)
        self.highlight_timer.start(50)

        if data['n2']:
            self.highlight_marker2nd = markers.CenterMarker(0, 0, radius, QtGui.QColor(36, 255, 167), 0, self.marker_changed)
            self.highlight_marker2nd.setOpacity(0.40)
            centroid = data['n2'].centroid()
            self.highlight_marker2nd.setPos(centroid[1]-radius/2, centroid[0]-radius/2)
            self.highlight_marker2nd_frame = data['n2'].frame_

    def _show_gt_markers(self, animal_ids2centroids):
        for a in self.project.animals:
            c_ = QtGui.QColor(a.color_[2], a.color_[1], a.color_[0])
            gt_m = markers.CenterMarker(0, 0, 10, c_, id=0, changeHandler=self._gt_marker_clicked)

            x = 10*a.id
            y = -1

            if a.id in animal_ids2centroids:
                y = animal_ids2centroids[a.id][0] - 5
                x = animal_ids2centroids[a.id][1] - 5

            gt_m.setPos(x, y)

            self.gitems['gt_markers'].append(gt_m)
            self.scene.addItem(gt_m)

    def _clear_items(self):
        for it in self.gitems['gt_markers']:
            self.scene.removeItem(it)

        self.gitems['gt_markers'] = []

        for i in range(len(self.one_frame_items)):
            self.scene.removeItem(self.one_frame_items[0])
            self.one_frame_items.pop(0)

        for c in self.colormarks_items:
            c.setVisible(False)

        self.colormarks_items = []

    def _update_bg_img(self, frame):
        img = self.video.get_frame(frame)
        if img is not None:
            if self.pixMapItem is not None:
                self.scene.removeItem(self.pixMapItem)

            if self.show_staurated_ch.isChecked():
                from utils.img import img_saturation
                img = img_saturation(img, saturation_coef=2.0, intensity_coef=1.05)


            self.pixMap = cvimg2qtpixmap(img)
            item = self.scene.addPixmap(self.pixMap)
            self.pixMapItem = item
            self.update_frame_number()
        else:
            self.out_of_frames()

    def update_positions(self, frame=-1):
        if frame == -1:
            frame = self.video.frame_number()

        self._clear_items()
        self._update_bg_img(frame)

        animal_ids2centroids = {}
        for ch in self.project.chm.chunks_in_frame(frame):
            rch = RegionChunk(ch, self.project.gm, self.project.rm)
            r = rch.region_in_t(frame)
            c = r.centroid().copy()


            if ch.animal_id_ > -1:
                animal_ids2centroids[ch.animal_id_] = c

            if self.show_contour_ch.isChecked() or self.show_filled_ch.isChecked():
                alpha = self.alpha_filled if self.show_filled_ch.isChecked() else self.alpha_contour
                self.draw_region(r, ch.animal_id_, use_ch_color=ch.color, alpha=alpha)

        if self.show_gt_markers.isChecked():
            self._show_gt_markers(animal_ids2centroids)


    def _gt_marker_clicked(self, id):
        print id

    def init_speed_slider(self):
        """Initiates components associated with speed of viewing videos"""
        self.speedSlider.setValue(self.frame_rate)
        self.timer.setInterval(1000 / self.frame_rate)
        # self.fpsLabel.setText(str(self.frame_rate) + ' fps')
        self.speedSlider.setMinimum(1)
        self.speedSlider.setMaximum(120)

    def speed_slider_changed(self):
        """Method invoked when value of slider controlling speed of video changed it's value"""
        self.frame_rate = self.speedSlider.value()
        self.timer.setInterval(1000 / self.frame_rate)
        self.fpsLabel.setText(str(self.frame_rate) + ' fps')

    def add_data(self, solver, just_around_frame=-1, margin=1000):
        self.solver = solver

        # self.chunks = self.solver.chm.chunk_list()
        self.chunks = []

        for v_id in self.project.gm.get_all_relevant_vertices():
            ch_id = self.project.gm.g.vp['chunk_start_id'][self.project.gm.g.vertex(v_id)]
            if ch_id > 0:
                self.chunks.append(self.project.chm[ch_id])

        i = 0

        t1 = just_around_frame - margin
        t2 = just_around_frame + margin

        if just_around_frame > -1:
            chs = []
            for ch in self.chunks:
                rch = RegionChunk(ch, self.project.gm, self.project.rm)
                if t1 < rch.start_frame() < t2 or t1 < rch.end_frame() < t2:
                    item = markers.CenterMarker(0, 0, MARKER_SIZE, ch.color, ch.id_, self.marker_changed)
                    item.setZValue(0.5)
                    self.items.append(item)
                    self.scene.addItem(item)

                    self.starting_frames.setdefault(rch.start_frame(), []).append((ch, i))

                    item.setVisible(False)

                    chs.append(ch)
                    i += 1

            self.chunks = chs
        else:
            import cPickle as pickle
            animal_id_mapping = None
            try:
                with open(self.project.working_directory+'/temp/animal_id_mapping.pkl', 'rb') as f_:
                    animal_id_mapping = pickle.load(f_)
            except:
                pass

            for ch in self.chunks:
                rch = RegionChunk(ch, self.project.gm, self.project.rm)

                if ch.id_ == 22:
                    print "22"

                col_ = ch.color
                if animal_id_mapping is not None:
                    if ch.id_ in animal_id_mapping:
                        animal_id = animal_id_mapping[ch.id_]
                        col_ = self.colors_[animal_id]
                    else:
                        col_ = QtGui.QColor().fromRgbF(0.3, 0.3, 0.3)

                item = markers.CenterMarker(0, 0, MARKER_SIZE, col_, ch.id_, self.marker_changed)
                item.setZValue(0.5)
                self.items.append(item)
                self.scene.addItem(item)

                self.starting_frames.setdefault(rch.start_frame(), []).append((ch, i))

                item.setVisible(False)

                i += 1

        self.update_positions(0)

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
            self.video.next_frame()
            self.update_positions(self.video.frame_number())
            # if img is not None:
            #     if self.pixMapItem is not None:
            #         self.scene.removeItem(self.pixMapItem)
            #
            #     self.pixMap = cvimg2qtpixmap(img)
            #     # view_add_bg_image(self.graphics_view, self.pixMap)
            #     item = self.scene.addPixmap(self.pixMap)
            #     self.pixMapItem = item
            #     self.update_frame_number()
            #     self.update_positions(self.video.frame_number())
            # else:
            #     self.out_of_frames()

        if self.video.frame_number() == self.highlight_marker2nd_frame:
            print "SHOW"
            self.scene.addItem(self.highlight_marker2nd)
            self.highlight_timer2nd.start(50)
            self.highlight_marker2nd_frame = -1

    def load_previous_frame(self):
        """Loads previous frame of the video if there is such and displays it"""
        if self.video is not None:
            self.video.previous_frame()
            # if img is not None:
            #     if self.pixMapItem is not None:
            #         self.scene.removeItem(self.pixMapItem)
            #     self.pixMap = cvimg2qtpixmap(img)
            #     # view_add_bg_image(self.graphics_view, self.pixMap)
            #     item = self.scene.addPixmap(self.pixMap)
            #     self.pixMapItem = item
            #     self.update_frame_number()
            self.update_positions(self.video.frame_number())

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
            self.video.seek_frame(position)
            # if img is not None:
            #     if self.pixMapItem is not None:
            #         self.scene.removeItem(self.pixMapItem)
            #     self.pixMap = cvimg2qtpixmap(img)
            #     # view_add_bg_image(self.graphics_view, self.pixMap)
            #     item = self.scene.addPixmap(self.pixMap)
            #     self.pixMapItem = item
            #     self.update_frame_number()
            self.update_positions(self.video.frame_number())
            # else:
            #     self.out_of_frames()

    def reset_colors(self):
        print "COLORIZING "
        from utils.color_manager import colorize_project
        colorize_project(self.project)

        print "COLORIZING DONE..."

# def view_add_bg_image(g_view, pix_map):
#     gv_w = g_view.geometry().width()
#     gv_h = g_view.geometry().height()
#     im_w = pix_map.width()
#     im_h = pix_map.height()
#
#     m11 = g_view.transform().m11()
#     m22 = g_view.transform().m22()
#
#     if m11 and m22 == 1:
#         if gv_w / float(im_w) <= gv_h / float(im_h):
#             val = math.floor((gv_w / float(im_w))*100) / 100
#             g_view.scale(val, val)
#         else:
#             val = math.floor((gv_h / float(im_h))*100) / 100
#             g_view.scale(val, val)