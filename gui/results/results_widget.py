__author__ = 'fnaiser'

import cPickle as pickle

import numpy as np
from PyQt4 import QtGui, QtCore

from core.graph.region_chunk import RegionChunk
from gui.video_player.video_player import VideoPlayer
from utils.misc import is_flipajs_pc
from gui.img_controls import markers
from utils.img import img_saturation_coef
from functools import partial
import warnings
from gui.gui_utils import SelectAllLineEdit, ClickableQGraphicsPixmapItem

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

        self.help_timer = QtCore.QTimer()
        self.scene = QtGui.QGraphicsScene()
        self.pixMap = None
        self.pixMapItem = None

        # used when save GT is called
        self._gt_markers = {}

        self._highlight_regions = set()
        self._highlight_tracklets = set()
        self.highlight_color = QtGui.qRgba(175, 255, 56, 200)  # green

        self.setLayout(self.hbox)
        self.splitter = QtGui.QSplitter()

        self.left_w = QtGui.QWidget()
        self.left_w.setMaximumWidth(250)
        self.left_vbox = QtGui.QVBoxLayout()
        self.left_vbox.setContentsMargins(0, 0, 0, 0)
        self.left_w.setLayout(self.left_vbox)

        if self.show_identities:
            self.scroll_ = QtGui.QScrollArea()
            self.scroll_.setWidgetResizable(True)
            from gui.results.identities_widget import IdentitiesWidget
            self.identities_widget = IdentitiesWidget(self.project)
            self.identities_widget.setMinimumWidth(200)
            self.scroll_.setWidget(self.identities_widget)

            self.left_vbox.addWidget(self.scroll_)

        self.splitter.addWidget(self.left_w)

        # GT Box
        self.gt_box = QtGui.QGroupBox('Ground Truth')
        self.gt_box.setLayout(QtGui.QVBoxLayout())
        self.evolve_gt_b = QtGui.QPushButton('evolve GT')
        self.evolve_gt_b.clicked.connect(self._evolve_gt)
        self.gt_box.layout().addWidget(self.evolve_gt_b)

        self.save_gt_b = QtGui.QPushButton('save gt')
        self.save_gt_b.clicked.connect(self.__save_gt)
        self.gt_box.layout().addWidget(self.save_gt_b)
        self.save_gt_a = QtGui.QAction('save gt', self)
        self.save_gt_a.triggered.connect(self.__save_gt)
        self.save_gt_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_G))
        self.addAction(self.save_gt_a)

        self.auto_gt_assignment_b = QtGui.QPushButton('auto GT')
        self.auto_gt_assignment_b.clicked.connect(self.__auto_gt_assignment)
        self.gt_box.layout().addWidget(self.auto_gt_assignment_b)

        self.auto_gt_assignment_action = QtGui.QAction('save gt', self)
        self.auto_gt_assignment_action.triggered.connect(self.__auto_gt_assignment)
        self.auto_gt_assignment_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_A))
        self.addAction(self.auto_gt_assignment_action)

        self.add_gt_markers_b = QtGui.QPushButton('add GT markers')
        self.add_gt_markers_b.clicked.connect(self.__add_gt_markers)
        self.gt_box.layout().addWidget(self.add_gt_markers_b)

        self.add_gt_markers_action = QtGui.QAction('add gt markers', self)
        self.add_gt_markers_action.triggered.connect(self.__add_gt_markers)
        self.add_gt_markers_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_D))
        self.addAction(self.add_gt_markers_action)

        self.left_vbox.addWidget(self.gt_box)

        # TRACKLET BOX
        self.tracklet_box = QtGui.QGroupBox('Tracklet controls')
        self.tracklet_box.setLayout(QtGui.QVBoxLayout())

        self.highlight_tracklet_input = SelectAllLineEdit('tracklet id')
        self.highlight_tracklet_input.returnPressed.connect(self.highlight_tracklet_button_clicked)

        self.tracklet_p_label = QtGui.QLabel('P: ')
        self.tracklet_n_label = QtGui.QLabel('N: ')

        self.highlight_tracklet_button = QtGui.QPushButton('show tracklet')
        self.highlight_tracklet_button.clicked.connect(self.highlight_tracklet_button_clicked)

        self.stop_highlight_tracklet = QtGui.QPushButton('stop highlight tracklet')
        self.stop_highlight_tracklet.clicked.connect(self.stop_highlight_tracklet_clicked)

        self.decide_tracklet_button = QtGui.QPushButton('decide tracklet')
        self.decide_tracklet_button.clicked.connect(self.decide_tracklet)
        self.decide_tracklet_button.setDisabled(True)

        self.decide_tracklet_action = QtGui.QAction('decide tracklet', self)
        self.decide_tracklet_action.triggered.connect(self.decide_tracklet)
        self.decide_tracklet_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_D))
        self.addAction(self.decide_tracklet_action)

        self.tracklet_end_button = QtGui.QPushButton('go to end')
        self.tracklet_end_button.clicked.connect(self.tracklet_end)

        self.tracklet_end_action = QtGui.QAction('tracklet end', self)
        self.tracklet_end_action.triggered.connect(self.tracklet_end)
        self.tracklet_end_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_E))
        self.addAction(self.tracklet_end_action)

        self.tracklet_begin_button = QtGui.QPushButton('go to beginning')
        self.tracklet_begin_button.clicked.connect(self.tracklet_begin)

        self.tracklet_begin_action = QtGui.QAction('tracklet begin', self)
        self.tracklet_begin_action.triggered.connect(self.tracklet_begin)
        self.tracklet_begin_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_S))
        self.addAction(self.tracklet_begin_action)

        self.tracklet_box.layout().addWidget(self.highlight_tracklet_input)
        self.tracklet_box.layout().addWidget(self.tracklet_p_label)
        self.tracklet_box.layout().addWidget(self.tracklet_n_label)
        self.tracklet_box.layout().addWidget(self.highlight_tracklet_button)
        self.tracklet_box.layout().addWidget(self.stop_highlight_tracklet)
        self.tracklet_box.layout().addWidget(self.decide_tracklet_button)
        self.tracklet_box.layout().addWidget(self.tracklet_end_button)
        self.tracklet_box.layout().addWidget(self.tracklet_begin_button)

        self.left_vbox.addWidget(self.tracklet_box)

        self.debug_box = QtGui.QGroupBox('debug box')
        self.debug_box.setLayout(QtGui.QVBoxLayout())
        self.reset_chunk_ids_b = QtGui.QPushButton('Reset chunk IDs')
        self.reset_chunk_ids_b.clicked.connect(self.reset_chunk_ids)
        self.debug_box.layout().addWidget(self.reset_chunk_ids_b)

        self.left_vbox.addWidget(self.debug_box)

        self.info_l = QtGui.QLabel('info')

        # TODO: show list of tracklets instead of QLine edit...
        # TODO: show range on frame time line
        # TODO: checkbox - stop at the end...

        self.left_vbox.addWidget(self.info_l)

        self.right_w = QtGui.QWidget()
        self.right_w.setLayout(self.right_vbox)
        self.splitter.addWidget(self.right_w)

        self.hbox.addWidget(self.splitter)

        self.video_layout = QtGui.QVBoxLayout()
        self.right_vbox.addLayout(self.video_layout)

        self.video_player = VideoPlayer(self.project)
        self.video_player.set_frame_change_callback(self.update_visualisations)
        self.video_layout.addWidget(self.video_player)

        self.hide_visualisation_action = QtGui.QAction('hide visualisation', self)
        self.hide_visualisation_action.triggered.connect(self.hide_visualisation)
        self.hide_visualisation_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_H))
        self.addAction(self.hide_visualisation_action)

        self.visu_controls_layout = QtGui.QHBoxLayout()
        self.video_layout.addLayout(self.visu_controls_layout)

        self.show_filled_ch = QtGui.QCheckBox('show filled')
        self.show_filled_ch.setChecked(True)
        # lambda is used because if only self.update_position is given, it will give it parameters...
        self.show_filled_ch.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.show_filled_ch)

        self.show_contour_ch = QtGui.QCheckBox('contours')
        self.show_contour_ch.setChecked(False)
        # lambda is used because if only self.update_position is given, it will give it parameters...
        self.show_contour_ch.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.show_contour_ch)

        self.contour_without_colors = QtGui.QCheckBox('id clrs')
        self.contour_without_colors.setChecked(True)
        self.contour_without_colors.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.contour_without_colors)

        self.show_markers = QtGui.QCheckBox('markers')
        self.show_markers.setChecked(True)
        self.show_markers.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.show_markers)

        self.show_saturated_ch = QtGui.QCheckBox('img sat')
        self.show_saturated_ch.setChecked(False)
        self.show_saturated_ch.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.show_saturated_ch)

        self.show_saturated_action = QtGui.QAction('show saturated', self)
        self.show_saturated_action.triggered.connect(lambda x: self.show_saturated_ch.setChecked(not self.show_saturated_ch.isChecked()))
        self.show_saturated_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_S))
        self.addAction(self.show_saturated_action)

        self.show_id_bar = QtGui.QCheckBox('id bar')
        self.show_id_bar.setChecked(False)
        self.show_id_bar.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.show_id_bar)

        self.toggle_id_bar_action = QtGui.QAction('toggle id bar', self)
        self.toggle_id_bar_action.triggered.connect(lambda x: self.show_id_bar.setChecked(not self.show_id_bar.isChecked()))
        self.toggle_id_bar_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_I))
        self.addAction(self.toggle_id_bar_action)

        self.goto_frame_action = QtGui.QAction('go to frame', self)
        self.goto_frame_action.triggered.connect(lambda x: self.video_player.frame_edit.setFocus())
        self.goto_frame_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_F))
        self.addAction(self.goto_frame_action)

        self.down_region_action = QtGui.QAction('down region', self)
        self.down_region_action.triggered.connect(partial(self.__set_active_relative_tracklet, -1, True))
        self.down_region_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_Up))
        self.addAction(self.down_region_action)

        self.up_region_action = QtGui.QAction('up region', self)
        self.up_region_action.triggered.connect(partial(self.__set_active_relative_tracklet, 1, True))
        self.up_region_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_Down))
        self.addAction(self.up_region_action)

        self.left_region_action = QtGui.QAction('left region', self)
        self.left_region_action.triggered.connect(partial(self.__set_active_relative_tracklet, -1, False))
        self.left_region_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_Left))
        self.addAction(self.left_region_action)

        self.right_region_action = QtGui.QAction('right region', self)
        self.right_region_action.triggered.connect(partial(self.__set_active_relative_tracklet, 1, False))
        self.right_region_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_Right))
        self.addAction(self.right_region_action)

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

        self.highlight_timer2nd.timeout.connect(partial(self.decrease_highlight_marker_opacity, True))

        self.colormarks_items = []

        self.one_frame_items = []

        self.alpha_contour = 240
        self.alpha_filled = 120

        self.colors_ = [
                QtGui.QColor().fromRgbF(0, 0, 1), #
                QtGui.QColor().fromRgbF(1, 0, 0),
                QtGui.QColor().fromRgbF(1, 1, 0),
                QtGui.QColor().fromRgbF(0, 1, 0), #
                QtGui.QColor().fromRgbF(0, 1, 1),
                QtGui.QColor().fromRgbF(1, 1, 1)
            ]

        # TODO: add develop option to load from project file...

        self._gt = {}
        if is_flipajs_pc():
            self._gt_corr_step = 50
            try:
                with open(self.project.GT_file, 'rb') as f:
                    self._gt = pickle.load(f)
            except:
                print "GT was not loaded"

        # self.redraw_video_player_visualisations()

    def reset_chunk_ids(self):
        msg = "Do you really want to delete all assigned ids to chunks?"
        reply = QtGui.QMessageBox.question(self, 'Message',
                                           msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

        if reply == QtGui.QMessageBox.Yes:
            for ch in self.project.chm.chunk_gen():
                ch.P = set()
                ch.N = set()

            if self.clear_user_decisions_callback:
                self.clear_user_decisions_callback()

            self.video_player.redraw_visualisations()

    def decide_tracklet(self):
        if self.active_tracklet_id > -1:
            self.decide_tracklet_callback(self.project.chm[self.active_tracklet_id])
            # self._set_active_tracklet_id(-1)
            self.decide_tracklet_button.setDisabled(True)

            self.redraw_video_player_visualisations()

    def tracklet_begin(self):
        if self.active_tracklet_id > -1:
            tracklet = self.project.chm[self.active_tracklet_id]
            frame = tracklet.start_frame(self.project.gm)

            self.video_player.goto(frame)

    def tracklet_end(self):
        if self.active_tracklet_id > -1:
            tracklet = self.project.chm[self.active_tracklet_id]
            frame = tracklet.end_frame(self.project.gm)

            self.video_player.goto(frame)

    def hide_visualisation(self):
        self.hide_visualisation_ = not self.hide_visualisation_
        self.redraw_video_player_visualisations()

    def stop_highlight_tracklet_clicked(self):
        self.loop_highlight_tracklets = []
        self.help_timer.stop()
        self.loop_end = -1
        self._set_active_tracklet_id(-1)
        self._highlight_tracklets = set()
        self.redraw_video_player_visualisations()

    def highlight_tracklet_button_clicked(self):
        self._highlight_tracklets = set()
        try:
            id_ = int(self.highlight_tracklet_input.text())
            tracklet = self.project.chm[id_]
        except:
            return

        # TODO: global parameter - margin
        self.play_and_highlight_tracklet(tracklet, margin=5)

    def play_and_highlight_tracklet(self, tracklet, frame=-1, margin=0):
        self._set_active_tracklet_id(tracklet.id())
        self._highlight_tracklets.add(tracklet)

        import warnings
        warnings.warn('not fully reimplemented in video_player', UserWarning)

        # return

        # frame=-1 ... start from beginning

        self.loop_begin = max(0, tracklet.start_frame(self.project.gm) - margin)
        self.loop_end = min(tracklet.end_frame(self.project.gm) + margin, self.video_player.total_frame_count()-1)
        self.loop_highlight_tracklets = [tracklet.id()]

        if frame < 0:
            frame = self.loop_begin

        self.video_player.goto(frame)

        # self.timer.stop()
        self.video_player.play()
        self.video_player.setFocus()

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

                # TODO: fix old GT files based on new specification....
                if y < 50 and x < 100:
                    continue

                else:
                    new_[frame][i] = (y, x)

        return new_

    def __save_gt(self):
        if self._gt is None:
            print "No GT file opened"
            return

        frame = self.video_player.current_frame()
        self._gt.setdefault(frame, [None]*len(self.project.animals))

        for it in self._gt_markers.itervalues():
            self._gt[frame][it.id] = (it.centerPos().y(), it.centerPos().x())

        with open(self.project.GT_file, 'wb') as f:
            pickle.dump(self._gt, f)

        print self._gt[frame]

    def __auto_gt_assignment(self):
        frame = self.video_player.current_frame()
        for ch in self.project.chm.chunks_in_frame(frame):
            rch = RegionChunk(ch, self.project.gm, self.project.rm)
            r = rch.region_in_t(frame)
            centroid = r.centroid().copy()

            try:
                if len(ch.P) == 1:
                    id_ = list(ch.P)[0]
                    print id_, self._gt_markers[id_].centerPos().y()
                    if self._gt[frame][id_][0] < 0:
                        print id_
                        self._gt[frame][id_] = (centroid[0], centroid[1])
            except:
                pass

        self.video_player.redraw_visualisations()

    def __add_gt_markers(self):
        if self._gt is None:
            print "No GT file opened"
            return

        frame = self.video_player.current_frame()
        self._gt.setdefault(frame, [None]*len(self.project.animals))
        for i in range(len(self.project.animals)):
            if self._gt[frame][i] is None:
                self._gt[frame][i] = (-10, i*10)

        self.video_player.redraw_visualisations()

    def __dilate(self, pts, roi):
        iterations = 7

        im = np.zeros((roi.height() + 2*iterations, roi.width() + 2*iterations), dtype=np.bool)
        im[pts[:, 0] + iterations, pts[:, 1] + iterations] = True

        from scipy.ndimage.morphology import binary_dilation

        im1 = binary_dilation(im, iterations=1)
        im = binary_dilation(im, iterations=iterations)

        # remove previous pts with 1 px border
        im ^= im1
        # im[pts[:, 0] + iterations, pts[:, 1] + iterations] = False

        pts = np.where(im)
        pts = np.vstack((pts[0], pts[1]))
        pts = pts.T

        from utils.roi import ROI
        roi = ROI(roi.y()-iterations,
                  roi.x()-iterations,
                  roi.height()+2*iterations,
                  roi.width()+2*iterations)

        return pts, roi

    def draw_region(self, r, tracklet, use_ch_color=None, alpha=120, highlight_contour=False, force_color=None):
        from utils.img import get_cropped_pts

        only_contour = False if self.show_filled_ch.isChecked() else True
        # we need region pts to dilate and substract previous pts to get result
        only_contour = only_contour and not highlight_contour

        pts_, roi = get_cropped_pts(r, return_roi=True, only_contour=only_contour)
        if highlight_contour:
            pts_, roi = self.__dilate(pts_, roi)

        offset = roi.top_left_corner()

        qim_ = QtGui.QImage(roi.width(), roi.height(), QtGui.QImage.Format_ARGB32)
        qim_.fill(QtGui.qRgba(0, 0, 0, 0))

        if force_color is None:
            c = QtGui.qRgba(150, 150, 150, 200)
            if self.contour_without_colors.isChecked():
                if tracklet.is_only_one_id_assigned(len(self.project.animals)):
                    id_ = list(tracklet.P)[0]
                    c_ = self.project.animals[id_].color_
                    c = QtGui.qRgba(c_[2], c_[1], c_[0], 255)
            elif use_ch_color:
                c = QtGui.qRgba(use_ch_color.red(), use_ch_color.green(), use_ch_color.blue(), alpha)
        else:
            c = force_color

        for i in range(pts_.shape[0]):
            qim_.setPixel(pts_[i, 1], pts_[i, 0], c)

        pixmap = QtGui.QPixmap.fromImage(qim_)

        item = ClickableQGraphicsPixmapItem(pixmap, tracklet.id(), self._gt_marker_clicked)
        item.setPos(offset[1], offset[0])
        item.setZValue(0.6)

        return item

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
        f = self.video_player.current_frame()

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

    def __add_marker(self, x, y, c_, id_, z_value, type_):
        radius = 13

        if type_ == 'GT':
            radius = 20
        elif type_ == 'multiple':
            radius = 9

        m = markers.CenterMarker(0, 0, radius, c_, id=id_, changeHandler=self._gt_marker_clicked)

        m.setPos(x - radius/2, y-radius/2)
        m.setZValue(z_value)

        if type_ == 'GT':
            self._gt_markers[id_] = m

        self.video_player.visualise_temp(m, category=type_)

    def _show_gt_markers(self):
        if self._gt is None:
            return

        for a in self.project.animals:
            c_ = QtGui.QColor(a.color_[2], a.color_[1], a.color_[0])

            frame = self.video_player.current_frame()

            if frame in self._gt:
                y = self._gt[frame][a.id][0]
                x = self._gt[frame][a.id][1]
                self.__add_marker(x, y, c_, a.id, 0.7, type_='GT')

    def _show_id_markers(self, animal_ids2centroids):
        for a in self.project.animals:
            c_ = QtGui.QColor(a.color_[2], a.color_[1], a.color_[0])

            if a.id in animal_ids2centroids:
                for i, data in enumerate(animal_ids2centroids[a.id]):
                    centroid = data[0]
                    y = centroid[0]
                    x = centroid[1]
                    decided = data[1]

                    ch = data[2]
                    type_ = 'normal' if decided else 'multiple'
                    self.__add_marker(x, y, c_, ch.id(), 0.75, type_=type_)
            else:
                x = 10*a.id
                y = -1

                self.__add_marker(x, y, c_, None, 0.75, type_='undef')

    def update_visualisations(self):
        if self.hide_visualisation_:
            return

        # reset...
        self._gt_markers = {}
        frame = self.video_player.current_frame()

        animal_ids2centroids = {}

        for ch in self.project.chm.chunks_in_frame(frame):
            rch = RegionChunk(ch, self.project.gm, self.project.rm)
            r = rch.region_in_t(frame)
            centroid = r.centroid().copy()

            if self.show_id_bar.isChecked():
                try:
                    item = self.show_pn_ids_visualisation(ch, frame)

                    self.video_player.visualise_temp(item, category='id_bar')
                except AttributeError:
                    pass

            if self.show_contour_ch.isChecked() or self.show_filled_ch.isChecked():
                alpha = self.alpha_filled if self.show_filled_ch.isChecked() else self.alpha_contour

                c_ = ch.color
                item = self.draw_region(r, ch, use_ch_color=c_, alpha=alpha)

                self.video_player.visualise_temp(item, category='region')

            # # TODO: fix for option when only P set is displayed using circles
            try:
                if len(ch.P) == 1:
                    id_ = list(ch.P)[0]
                    animal_ids2centroids.setdefault(id_, [])
                    animal_ids2centroids[id_].append((centroid, ch.is_only_one_id_assigned(len(self.project.animals)), ch))
            except:
                pass

            if ch.id() == self.active_tracklet_id: # in self._highlight_regions or ch in self._highlight_tracklets:
                item = self.draw_region(r, ch, highlight_contour=True, force_color=self.highlight_color)
                self.video_player.visualise_temp(item, category='region_highlight')

        if self.show_markers.isChecked():
            self._show_gt_markers()
            self._show_id_markers(animal_ids2centroids)


    def _img_saturation(self, img):
        return img_saturation_coef(img, 2.0, 1.05)

    def redraw_video_player_visualisations(self):
        callback = None
        if self.show_saturated_ch.isChecked():
            callback = self._img_saturation

        self.video_player.set_image_processor_callback(callback)
        self.video_player.redraw_visualisations()
        # TODO: looper
        # self.__highlight_tracklets()

    def __highlight_tracklets(self):
        for id_ in self.loop_highlight_tracklets:
            tracklet = self.project.chm[id_]
            frame = self.video_player.current_frame()

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
        # self.scene.addItem(item)
        item.setPos(reg.centroid()[1], reg.centroid()[0])
        item.setZValue(0.9)
        item.setFlags(QtGui.QGraphicsItem.ItemIsMovable)
        # self.one_frame_items.append(item)

        return item

    def pn_pixmap_clicked(self, id_):
        self._gt_marker_clicked(id_)

    def __compute_radius(self, r):
        from scipy.spatial.distance import cdist
        c = r.centroid()
        c.shape = (1, 2)
        D = cdist(c, r.contour())
        radius = np.max(D)

        return radius

    def _update_tracklet_info(self):
        id_ = self.active_tracklet_id
        if id_ < 0:
            self.info_l.setText(' ')
            self.tracklet_p_label.setText(' ')
            self.tracklet_n_label.setText(' ')

            return

        s = 'id: ' + str(id_)

        ch = self.project.chm[id_]
        f = self.video_player.current_frame()
        rch = RegionChunk(ch, self.project.gm, self.project.rm)
        r = rch.region_in_t(f)
        import textwrap
        s += "\n" + str(r)
        s = textwrap.fill(s, 40)
        s += " radius: {:.3}".format(self.__compute_radius(r))

        if ch.start_frame(self.project.gm) == f:
            s += "\nin_degree: " + str(ch.start_vertex(self.project.gm).in_degree())

        if ch.end_frame(self.project.gm) == f:
            s += "\nout degree: " + str(ch.end_vertex(self.project.gm).out_degree())

        s += "\nlength: " + str(ch.length()) + " s: " + str(ch.start_frame(self.project.gm)) + " e: " + str(
            ch.end_frame(self.project.gm))

        avg_area = 0
        for r_ in rch.regions_gen():
            avg_area += r_.area()

        avg_area /= rch.chunk_.length()

        s += "\navg area: " + str(avg_area)
        # from core.learning.learning_process import get_features_var1, features2str_var1
        # s += "\nFeature vector: "+ features2str_var1(get_features_var1(r, self.project))

        self.info_l.setText(s)

        self.tracklet_p_label.setText(str(ch.P))
        self.tracklet_n_label.setText(str(ch.N))

        # TODO: solve better... something like set
        # self._highlight_tracklets = set()
        # self._highlight_tracklets.add(id_)
        self.video_player.redraw_visualisations()

    def _gt_marker_clicked(self, id_):
        self.decide_tracklet_button.setDisabled(False)
        self._set_active_tracklet_id(id_)
        self.highlight_tracklet_input.setText(str(id_))

        self._update_tracklet_info()

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

        self.redraw_video_player_visualisations(0)

    def __continue_loop(self):
        self.help_timer.stop()
        # TODO: if loop only once... check something and remove flags...
        self.change_frame(self.loop_begin)
        self.timer.start()

    def _set_active_tracklet_id(self, id_=None):
        if id_ == None:
            try:
                id_ = int(self.highlight_tracklet_input.text())
            except:
                return

        self.active_tracklet_id = id_
        self._update_tracklet_info()

    def __set_active_relative_tracklet(self, offset, ud=True):
        frame = self.video_player.current_frame()

        data = []
        for t in self.project.chm.chunks_in_frame(frame):
            rch = RegionChunk(t, self.project.gm, self.project.rm)
            c = rch.centroid_in_t(frame)
            data.append((t.id(), c[0], c[1]))

        if ud:
            data.sort(key=lambda tup: tup[1])
        else:
            data.sort(key=lambda tup: tup[2])

        ids_ = []
        for tup in data:
            ids_.append(tup[0])

        try:
            self._set_active_tracklet_id(ids_[ids_.index(self.active_tracklet_id) + offset])
        except:
            if offset < 0:
                self._set_active_tracklet_id(ids_[-1])
            else:
                self._set_active_tracklet_id(ids_[0])
