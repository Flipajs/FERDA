__author__ = 'fnaiser'

import cPickle as pickle

import numpy as np
from PyQt4 import QtGui, QtCore

from core.graph.region_chunk import RegionChunk
from core.settings import Settings as S_
from gui.img_controls.my_view import MyView
from gui.img_controls.utils import cvimg2qtpixmap
from select_all_line_edit import SelectAllLineEdit
from utils.misc import is_flipajs_pc
from utils.video_manager import get_auto_video_manager
from video_slider import VideoSlider
from viewer.gui.img_controls import markers

MARKER_SIZE = 15

class ResultsWidget(QtGui.QWidget):
    def __init__(self, project, start_on_frame=-1, decide_tracklet_callback=None):
        super(ResultsWidget, self).__init__()

        self.decide_tracklet_callback = decide_tracklet_callback

        self.show_identities = False
        self.loop_highlight_tracklets = []
        self.loop_end = -1

        self.active_tracklet_id = -1

        self.hide_visualisation_ = False

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
        self.help_timer = QtCore.QTimer()
        self.scene = QtGui.QGraphicsScene()
        self.pixMap = None
        self.pixMapItem = None

        self.setLayout(self.hbox)
        self.splitter = QtGui.QSplitter()

        self.left_w = QtGui.QWidget()
        self.left_vbox = QtGui.QVBoxLayout()
        self.left_vbox.setContentsMargins(0, 0, 0, 0)
        self.left_w.setLayout(self.left_vbox)
        self.save_gt_b = QtGui.QPushButton('save gt')
        self.save_gt_b.clicked.connect(self.__save_gt)
        self.save_gt_a = QtGui.QAction('save gt', self)
        self.save_gt_a.triggered.connect(self.__save_gt)
        self.save_gt_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_G))
        self.addAction(self.save_gt_a)

        self.left_vbox.addWidget(self.save_gt_b)

        if self.show_identities:
            self.scroll_ = QtGui.QScrollArea()
            self.scroll_.setWidgetResizable(True)
            from gui.correction.identities_widget import IdentitiesWidget
            self.identities_widget = IdentitiesWidget(self.project)
            self.identities_widget.setMinimumWidth(200)
            self.scroll_.setWidget(self.identities_widget)

            self.left_vbox.addWidget(self.scroll_)

        self.splitter.addWidget(self.left_w)

        self.info_l = QtGui.QLabel('info')

        # TODO: show list of tracklets instead of QLine edit...
        # TODO: show range on frame time line
        # TODO: checkbox - stop at the end...
        self.highlight_tracklet_input = QtGui.QLineEdit('tracklet id')
        self.highlight_tracklet_button = QtGui.QPushButton('show tracklet')
        self.highlight_tracklet_button.clicked.connect(self.highlight_tracklet_button_clicked)
        self.stop_highlight_tracklet = QtGui.QPushButton('stop highlight tracklet')
        self.stop_highlight_tracklet.clicked.connect(self.stop_highlight_tracklet_clicked)
        self.left_vbox.addWidget(self.highlight_tracklet_input)
        self.left_vbox.addWidget(self.highlight_tracklet_button)
        self.left_vbox.addWidget(self.stop_highlight_tracklet)

        self.left_vbox.addWidget(self.info_l)

        self.decide_tracklet_button = QtGui.QPushButton('decide tracklet')
        self.decide_tracklet_button.clicked.connect(self.decide_tracklet)
        self.decide_tracklet_button.setDisabled(True)
        self.left_vbox.addWidget(self.decide_tracklet_button)

        self.decide_tracklet_action = QtGui.QAction('decide tracklet', self)
        self.decide_tracklet_action.triggered.connect(self.decide_tracklet)
        self.decide_tracklet_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_D))
        self.addAction(self.decide_tracklet_action)

        self.evolve_gt_b = QtGui.QPushButton('evolve GT')
        self.evolve_gt_b.clicked.connect(self._evolve_gt)
        self.left_vbox.addWidget(self.evolve_gt_b)


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

        self.backward = QtGui.QPushButton('<')
        self.backward.setShortcut(S_.controls.video_prev)
        self.playPause = QtGui.QPushButton('play')
        self.playPause.setShortcut(S_.controls.video_play_pause)
        self.forward = QtGui.QPushButton('>')
        self.forward.setShortcut(S_.controls.video_next)
        self.frameEdit = SelectAllLineEdit()
        self.frameEdit.returnPressed.connect(self.frame_jump)
        self.frameEdit.setFixedHeight(30)
        # self.showFrame = QtGui.QPushButton('show')
        self.fpsLabel = QtGui.QLabel()
        self.fpsLabel.setAlignment(QtCore.Qt.AlignRight)
        self.videoSlider = VideoSlider()
        self.videoSlider.setOrientation(QtCore.Qt.Horizontal)
        self.videoSlider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.videoSlider.setMaximumHeight(15)
        self.videoSlider.setMaximum(self.video.total_frame_count())



        self.video_control_layout.addWidget(self.videoSlider)
        self.video_control_layout.addWidget(self.video_control_buttons_widget)

        self.frame_jump_button = QtGui.QPushButton('go')
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
        # self.video_control_buttons_layout.addWidget(self.showFrame)
        self.video_control_buttons_layout.addWidget(self.frameEdit)
        self.video_control_buttons_layout.addWidget(self.frame_jump_button)

        self.init_speed_slider()

        self.hide_visualisation_action = QtGui.QAction('hide visualisation', self)
        self.hide_visualisation_action.triggered.connect(self.hide_visualisation)
        self.hide_visualisation_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_H))
        self.addAction(self.hide_visualisation_action)

        # self.reset_colors_b = QtGui.QPushButton('reset colors')
        # self.reset_colors_b.clicked.connect(self.reset_colors)
        # self.video_control_buttons_layout.addWidget(self.reset_colors_b)

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

        self.contour_without_colors = QtGui.QCheckBox('w\'out colors')
        self.contour_without_colors.setChecked(False)
        self.contour_without_colors.stateChanged.connect(lambda x: self.update_positions())
        self.visu_controls_layout.addWidget(self.contour_without_colors)

        self.show_gt_markers = QtGui.QCheckBox('gt markers')
        self.show_gt_markers.setChecked(True)
        self.show_gt_markers.stateChanged.connect(lambda x: self.update_positions())
        self.visu_controls_layout.addWidget(self.show_gt_markers)

        self.show_saturated_ch = QtGui.QCheckBox('img saturated')
        self.show_saturated_ch.setChecked(False)
        self.show_saturated_ch.stateChanged.connect(lambda x: self.update_positions())
        self.visu_controls_layout.addWidget(self.show_saturated_ch)

        self.show_saturated_action = QtGui.QAction('show saturated', self)
        self.show_saturated_action.triggered.connect(lambda x: self.show_saturated_ch.setChecked(not self.show_saturated_ch.isChecked()))
        self.show_saturated_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_S))
        self.addAction(self.show_saturated_action)

        self.show_id_bar = QtGui.QCheckBox('id bar')
        self.show_id_bar.setChecked(True)
        self.show_id_bar.stateChanged.connect(lambda x: self.update_positions())
        self.visu_controls_layout.addWidget(self.show_id_bar)

        self.big_video_forward_a = QtGui.QAction('big next', self)
        self.big_video_forward_a.triggered.connect(lambda x: self.change_frame(self.video.frame_number() + 50))
        self.big_video_forward_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_0))
        self.addAction(self.big_video_forward_a)

        self.big_video_backward_a = QtGui.QAction('big next', self)
        self.big_video_backward_a.triggered.connect(lambda x: self.change_frame(self.video.frame_number() - 50))
        self.big_video_backward_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_9))
        self.addAction(self.big_video_backward_a)

        self.connect_GUI()

        self.video.next_frame()
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

        self._gt = None
        self._gt_file = self.project.working_directory + '/GT_sparse.pkl'
        self._gt_corr_step = 50
        if is_flipajs_pc():
            try:
                with open(self._gt_file, 'rb') as f:
                    self._gt = pickle.load(f)
            except:
                self._gt = {}

        # self.update_positions()

    def decide_tracklet(self):
        if self.active_tracklet_id > -1:
            self.decide_tracklet_callback(self.project.chm[self.active_tracklet_id])
            self.active_tracklet_id = -1
            self.decide_tracklet_button.setDisabled(True)

            self.update_positions()

    def hide_visualisation(self):
        self.hide_visualisation_ = not self.hide_visualisation_
        self.update_positions()

    def stop_highlight_tracklet_clicked(self):
        self.loop_highlight_tracklets = []
        self.help_timer.stop()
        self.loop_end = -1
        self.update_positions()

    def highlight_tracklet_button_clicked(self):
        try:
            id_ = int(self.highlight_tracklet_input.text())
            tracklet = self.project.chm[id_]
        except:
            return

        # TODO: global parameter - margin
        self.play_and_highlight_tracklet(tracklet, margin=5)

    def play_and_highlight_tracklet(self, tracklet, frame=-1, margin=0):
        # frame=-1 ... start from beginning

        self.loop_begin = max(0, tracklet.start_frame(self.project.gm) - margin)
        self.loop_end = min(tracklet.end_frame(self.project.gm) + margin, self.video.total_frame_count()-1)
        self.loop_highlight_tracklets = [tracklet.id()]

        if frame < 0:
            frame = self.loop_begin

        self.change_frame(frame)
        self.timer.stop()
        self.play_pause()

        # TODO: frame loop
        # TODO: visualize loop

    def _evolve_gt(self):
        my_data = {}

        # TODO: clearmetrics bug workaround...
        max_frame = 0
        for frame in self._gt:
            # if frame > 4000:
            #     continue

            my_data[frame] = np.array(([None] * len(self.project.animals)))
            for ch in self.project.chm.chunks_in_frame(frame):
                rch = RegionChunk(ch, self.project.gm, self.project.rm)

                if ch.is_only_one_id_assigned(len(self.project.animals)):
                    id_ = list(ch.P)[0]
                    my_data[frame][id_] = rch.centroid_in_t(frame)

            max_frame = max(max_frame, frame)

        for f in xrange(max_frame+100):
            my_data.setdefault(f, np.array([None] * len(self.project.animals)))

        from utils.clearmetrics.clearmetrics import ClearMetrics
        threshold = 10

        gt_ = self.__prepare_gt()

        clear = ClearMetrics(gt_, my_data, threshold)
        clear.match_sequence()
        evaluation = [clear.get_mota(),
                      clear.get_motp(),
                      clear.get_fn_count(),
                      clear.get_fp_count(),
                      clear.get_mismatches_count(),
                      clear.get_object_count(),
                      clear.get_matches_count()]

        print ("MOTA: %.3f\nMOTP: %.3f\n#FN: %d\n#FP:%d\n#mismatches: %d\n#objects: %d\n#matches: %d") % (
            tuple(evaluation)
        )

        return evaluation

    def __prepare_gt(self):
        new_ = {}

        for frame in self._gt:
            new_[frame] = [None] * len(self._gt[frame])

            for i, data in enumerate(self._gt[frame]):
                y = data[0]
                x = data[1]

                if y < 50 and x < 100:
                    continue

                else:
                    new_[frame][i] = (y, x)

        return new_

    def __save_gt(self):
        if self._gt is None:
            print "No GT file opened"
            return

        frame = self.video.frame_number()
        self._gt.setdefault(frame, [None]*len(self.project.animals))

        for it in self.gitems['gt_markers']:
            self._gt[frame][it.id] = (it.centerPos().y(), it.centerPos().x())

        with open(self._gt_file, 'wb') as f:
            pickle.dump(self._gt, f)

        self.change_frame(frame + self._gt_corr_step)
        print self._gt[frame]

    def draw_region(self, r, tracklet, use_ch_color=None, alpha=120):
        from utils.img import get_cropped_pts

        pts_, roi = get_cropped_pts(r, return_roi=True, only_contour=False if self.show_filled_ch.isChecked() else True)
        offset = roi.top_left_corner()

        qim_ = QtGui.QImage(roi.width(), roi.height(), QtGui.QImage.Format_ARGB32)
        qim_.fill(QtGui.qRgba(0, 0, 0, 0))

        c = QtGui.qRgba(100, 100, 100, 200)
        if self.contour_without_colors.isChecked():
            if tracklet.is_only_one_id_assigned(len(self.project.animals)):
                id_ = list(tracklet.P)[0]
                c_ = self.project.animals[id_].color_
                c = QtGui.qRgba(c_[2], c_[1], c_[0], 255)
        elif use_ch_color:
            c = QtGui.qRgba(use_ch_color.red(), use_ch_color.green(), use_ch_color.blue(), alpha)

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

    def __find_nearest_free_marker_pos(self, y, x):
        import itertools
        x_ = round(x / float(self.marker_helper_step))
        y_ = round(y / float(self.marker_helper_step))

        if self.marker_pos_helper[int(y_), int(x_)]:
            for a, b in itertools.product([-1, 0, 1], [-1, 0, 1]):
                if not self.marker_pos_helper[int(y_+a), int(x_+b)]:
                    y_ += a
                    x_ += b

                    y = y_ * self.marker_helper_step
                    x = x_ * self.marker_helper_step
                    break

        self.marker_pos_helper[int(y_), int(x_)] = True
        return y, x

    def update_positions_optimized(self, frame):
        new_active_markers = []

        # TODO: BGR, offset 1
        # R B G Y dark B
        colors = [
            [0, 0, 0],
            [0, 0, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 255, 255],
            [150, 0, 0]
        ]

        for m_id, ch in self.active_markers:
            rch = RegionChunk(ch,  self.project.gm, self.project.rm)
            if frame == rch.end_frame() + 1:
                self.items[m_id].setVisible(False)
            else:
                new_active_markers.append((m_id, ch))
                r = rch.region_in_t(frame)

                if r is None:
                    print "None region, frame: {}, ch.id_: {}".format(frame, ch.id_)
                    continue

                c = r.centroid().copy()
                self.update_marker_position(self.items[m_id], c)

    def __add_marker(self, x, y, c_, id_, z_value, type_):
        radius = 13

        if type_ == 'GT':
            radius = 20
        elif type_ == 'multiple':
            radius = 9


        if type_ != 'GT':
            y, x = self.__find_nearest_free_marker_pos(y, x)

        gt_m = markers.CenterMarker(0, 0, radius, c_, id=id_, changeHandler=self._gt_marker_clicked)

        gt_m.setPos(x - radius/2, y-radius/2)
        gt_m.setZValue(z_value)

        self.gitems['gt_markers'].append(gt_m)
        self.scene.addItem(gt_m)

    def _show_gt_markers(self, animal_ids2centroids):
        if self._gt is None:
            return

        for a in self.project.animals:
            c_ = QtGui.QColor(a.color_[2], a.color_[1], a.color_[0])

            frame = self.video.frame_number()

            if frame in self._gt:
                y = self._gt[frame][a.id][0]
                x = self._gt[frame][a.id][1]
                self.__add_marker(x, y, c_, a.id, 0.7, type_='GT')

            if a.id in animal_ids2centroids:
                for i, data in enumerate(animal_ids2centroids[a.id]):
                    centroid = data[0]
                    decided = data[1]
                    ch = data[2]
                    y = centroid[0]
                    x = centroid[1]
                    type_ = 'normal' if decided else 'multiple'
                    self.__add_marker(x, y, c_, ch.id_, 0.75, type_=type_)
            else:
                x = 10*a.id
                y = -1

                self.__add_marker(x, y, c_, a.id, 0.75, type_='undef')

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

            if self.show_saturated_ch.isChecked():
                from utils.img import img_saturation
                img = img_saturation(img, saturation_coef=2.0, intensity_coef=1.05)


            self.pixMap = cvimg2qtpixmap(img)
            item = self.scene.addPixmap(self.pixMap)
            self.pixMapItem = item
            self.update_frame_number()
        else:
            self.out_of_frames()

    def update_positions(self, frame=-1):
        self.marker_helper_step = 7
        from math import ceil
        self.marker_pos_helper = np.zeros((int(ceil(self.video.img().shape[0] / self.marker_helper_step)),
                                            int(ceil(self.video.img().shape[1] / self.marker_helper_step))),
                                          dtype=np.bool)


        if frame == -1:
            frame = self.video.frame_number()

        self._clear_items()
        self._update_bg_img(frame)

        if self.hide_visualisation_:
            return

        animal_ids2centroids = {}
        for ch in self.project.chm.chunks_in_frame(frame):
            rch = RegionChunk(ch, self.project.gm, self.project.rm)
            r = rch.region_in_t(frame)
            c = r.centroid().copy()

            # TODO: solve somewhere else:
            if not hasattr(ch, 'P'):
                ch.P = set()
                ch.N = set()

            if self.show_id_bar.isChecked():
                try:
                    self.show_pn_ids_visualisation(ch, frame)
                except AttributeError:
                    pass

            # TODO: fix for option when only P set is displayed using circles
            try:
                for id_ in ch.P:
                    animal_ids2centroids.setdefault(id_, [])
                    animal_ids2centroids[id_].append((c, ch.is_only_one_id_assigned(len(self.project.animals)), ch))
            except:
                pass

            if self.show_contour_ch.isChecked() or self.show_filled_ch.isChecked():
                alpha = self.alpha_filled if self.show_filled_ch.isChecked() else self.alpha_contour

                c = ch.color
                self.draw_region(r, ch, use_ch_color=c, alpha=alpha)

        if self.show_gt_markers.isChecked():
            self._show_gt_markers(animal_ids2centroids)

        self.__highlight_tracklets()

    def __highlight_tracklets(self):
        for id_ in self.loop_highlight_tracklets:
            tracklet = self.project.chm[id_]
            frame = self.video.frame_number()

            r = RegionChunk(tracklet, self.project.gm, self.project.rm).region_in_t(frame)

            # out of frame range
            if r is None:
                return

            centroid = r.centroid()
            # TODO: global parameters
            radius = 130

            m = markers.CenterMarker(0, 0, radius, QtGui.QColor(167, 255, 36), 0, self.marker_changed)
            m.setOpacity(0.20)
            m.setPos(centroid[1]-radius/2, centroid[0]-radius/2)
            self.scene.addItem(m)
            self.one_frame_items.append(m)

    def show_pn_ids_visualisation(self, tracklet, frame):
        rch = RegionChunk(tracklet, self.project.gm, self.project.rm)

        from gui.view import pn_ids_visualisation
        from gui.view.pn_ids_visualisation import default_params

        params = default_params
        params['colors'] = []

        ids_ = range(len(self.project.animals))
        for i, a in enumerate(self.project.animals):
            params['colors'].append([a.color_[0], a.color_[1], a.color_[2]])

        item = pn_ids_visualisation.get_pixmap_item(ids_, tracklet.P, tracklet.N,
                                                     tracklet_id=tracklet.id(),
                                                     callback=self.pn_pixmap_clicked,
                                                    # TODO: probs on Demand
                                                     probs=None,
                                                     # probs=tracklet.animal_id_['probabilities'],
                                                     params=params
                                                     )

        reg = rch.region_in_t(frame)

        self.scene.addItem(item)

        self.one_frame_items.append(item)
        self.one_frame_items[-1].setPos(reg.centroid()[1], reg.centroid()[0])
        self.one_frame_items[-1].setZValue(0.9)
        # self.one_frame_items[-1].setOpacity(0.9)
        self.one_frame_items[-1].setFlags(QtGui.QGraphicsItem.ItemIsMovable)

    def pn_pixmap_clicked(self, id_):
        self._gt_marker_clicked(id_)

    def __compute_radius(self, r):
        from scipy.spatial.distance import cdist
        c = r.centroid()
        c.shape = (1, 2)
        D = cdist(c, r.contour())
        radius = np.max(D)

        return radius

    def _gt_marker_clicked(self, id_):
        self.decide_tracklet_button.setDisabled(False)
        self.active_tracklet_id = id_
        s = 'id: '+str(id_)

        ch = self.project.chm[id_]
        f = self.video.frame_number()
        rch = RegionChunk(ch, self.project.gm, self.project.rm)
        r = rch.region_in_t(f)
        import textwrap
        s += "\n" + str(r)
        s = textwrap.fill(s, 50)
        s += " radius: {:.3}".format(self.__compute_radius(r))

        if ch.start_frame(self.project.gm) == f:
            s += "\nin_degree: " + str(ch.start_vertex(self.project.gm).in_degree())

        if ch.end_frame(self.project.gm) == f:
            s += "\nout degree: " + str(ch.end_vertex(self.project.gm).out_degree())

        s += "\nlength: " + str(ch.length()) + " s: " + str(ch.start_frame(self.project.gm)) + " e: " + str(ch.end_frame(self.project.gm))


        avg_area = 0
        for r in rch.regions_gen():
            avg_area += r.area()

        avg_area /= rch.chunk_.length()

        s += "\navg area: "+str(avg_area)
        # from core.learning.learning_process import get_features_var1, features2str_var1
        # s += "\nFeature vector: "+ features2str_var1(get_features_var1(r, self.project))

        self.info_l.setText(s)

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
            chunk_available_ids = None
            try:
                with open(self.project.working_directory+'/temp/chunk_available_ids.pkl', 'rb') as f_:
                    chunk_available_ids = pickle.load(f_)
            except:
                pass

            for ch in self.chunks:
                rch = RegionChunk(ch, self.project.gm, self.project.rm)

                if ch.id_ == 22:
                    print "22"

                col_ = ch.color
                if chunk_available_ids is not None:
                    if ch.id_ in chunk_available_ids:
                        animal_id = chunk_available_ids[ch.id_]
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

    def __continue_loop(self):
        self.help_timer.stop()
        # TODO: if loop only once... check something and remove flags...
        self.change_frame(self.loop_begin)
        self.timer.start()

    def load_next_frame(self):
        """Loads next frame of the video and displays it. If there is no next frame, calls self.out_of_frames"""
        if self.video is not None:
            self.video.next_frame()
            self.update_positions(self.video.frame_number())
        else:
            self.play_pause()

        if self.video.frame_number() == self.loop_end:
            # TODO: global parameter
            wait_in_the_end = 600

            self.timer.stop()
            self.help_timer.setInterval(wait_in_the_end)
            self.help_timer.start()
            self.help_timer.timeout.connect(self.__continue_loop)
            return

            print len(self.project.gm.vertices_in_t[self.video.frame_number()])

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
            self.update_positions(self.video.frame_number())

    def reset_colors(self):
        print "COLORIZING "
        from utils.color_manager import colorize_project
        colorize_project(self.project)

        print "COLORIZING DONE..."
<<<<<<< HEAD
=======



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
>>>>>>> cluster_preparations
