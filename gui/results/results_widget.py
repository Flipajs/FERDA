__author__ = 'fnaiser'

import cPickle as pickle
import Queue
from functools import partial

import numpy as np
from PyQt4 import QtGui, QtCore

from core.graph.region_chunk import RegionChunk
from core.settings import Settings as S_
from gui.gui_utils import SelectAllLineEdit, ClickableQGraphicsPixmapItem
from gui.img_controls import markers
from gui.video_player.video_player import VideoPlayer
from utils.img import img_saturation_coef
from utils.misc import is_flipajs_pc
from core.region.region import get_region_endpoints
from utils.idtracker import load_idtracker_data
import cv2
import time
from gui.qt_flow_layout import FlowLayout
from tqdm import tqdm
from pyqtgraph import makeQImage

MARKER_SIZE = 15

class ResultsWidget(QtGui.QWidget):
    def __init__(self, project, start_on_frame=-1, callbacks=None):
        super(ResultsWidget, self).__init__()

        self.decide_tracklet_callback = None
        if 'decide_tracklet' in callbacks:
            self.decide_tracklet_callback = callbacks['decide_tracklet']

        self.edit_tracklet_callback = None
        if 'edit_tracklet' in callbacks:
            self.edit_tracklet_callback = callbacks['edit_tracklet']

        self.get_separated_frame_callback = None
        if 'get_separated_frame' in callbacks:
            self.get_separated_frame_callback = callbacks['get_separated_frame']

        if 'update_N_sets' in callbacks:
            self.update_N_sets_callback = callbacks['update_N_sets']

        if 'tracklet_measurements' in callbacks:
            self.tracklet_measurements = callbacks['tracklet_measurements']

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

        self.idtracker_data = None
        self.idtracker_data_permutation = {}

        self._highlight_regions = set()
        self._highlight_tracklets = set()
        self.highlight_color = (175, 255, 56, 200)  # green

        self.setLayout(self.hbox)
        self.splitter = QtGui.QSplitter()

        self.left_w = QtGui.QWidget()
        self.left_w.setMaximumWidth(250)
        self.left_vbox = QtGui.QVBoxLayout()
        self.left_vbox.setContentsMargins(0, 0, 0, 0)
        self.left_w.setLayout(self.left_vbox)

        self.save_button = QtGui.QPushButton('save project')
        self.save_button.clicked.connect(lambda x: self.project.save())
        self.left_vbox.addWidget(self.save_button)

        self.set_colors_b = QtGui.QPushButton('set ID colors')
        self.set_colors_b.clicked.connect(self.show_color_settings)
        self.left_vbox.addWidget(self.set_colors_b)

        if self.show_identities:
            self.scroll_ = QtGui.QScrollArea()
            self.scroll_.setWidgetResizable(True)
            from gui.results.identities_widget import IdentitiesWidget
            self.identities_widget = IdentitiesWidget(self.project)
            self.identities_widget.setMinimumWidth(200)
            self.scroll_.setWidget(self.identities_widget)

            self.left_vbox.addWidget(self.scroll_)

        self.l_scroll_ = QtGui.QScrollArea()
        self.l_scroll_.setWidgetResizable(True)
        self.l_scroll_.setWidget(self.left_w)

        self.splitter.addWidget(self.l_scroll_)

        # GT Box
        self.gt_box = QtGui.QGroupBox('Ground Truth')
        self.gt_box.setLayout(QtGui.QVBoxLayout())

        self.gt_file_label = QtGui.QLabel('None')
        self.gt_box.layout().addWidget(self.gt_file_label)
        if hasattr(self.project, 'GT_file'):
            self.gt_file_label.setText(self.project.GT_file)

        self.load_gt_b = QtGui.QPushButton('load GT')
        self.load_gt_b.clicked.connect(self.load_gt_file_dialog)
        self.gt_box.layout().addWidget(self.load_gt_b)

        self.eval_gt_draw_ch = QtGui.QCheckBox('draw coverage image')
        self.gt_box.layout().addWidget(self.eval_gt_draw_ch)

        self.evolve_gt_b = QtGui.QPushButton('evaluate GT')
        self.evolve_gt_b.clicked.connect(self._evaluate_gt)
        self.gt_box.layout().addWidget(self.evolve_gt_b)

        self.evolve_gt_id_b = QtGui.QPushButton('ID assignment STATS')
        self.evolve_gt_id_b.clicked.connect(self._evolve_gt_id)
        self.gt_box.layout().addWidget(self.evolve_gt_id_b)

        self.save_gt_b = QtGui.QPushButton('save gt')
        self.save_gt_b.clicked.connect(self.__save_gt)
        self.gt_box.layout().addWidget(self.save_gt_b)
        self.save_gt_a = QtGui.QAction('save gt', self)
        self.save_gt_a.triggered.connect(self.__save_gt)
        self.save_gt_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_G))
        self.addAction(self.save_gt_a)

        self.gt_find_permutation_b = QtGui.QPushButton('find permutation')
        self.gt_find_permutation_b.clicked.connect(self._gt_find_permutation)
        self.gt_box.layout().addWidget(self.gt_find_permutation_b)

        self.gt_find_permutation_here_b = QtGui.QPushButton('find permutation here')
        self.gt_find_permutation_here_b.clicked.connect(partial(self._gt_find_permutation, True))
        self.gt_box.layout().addWidget(self.gt_find_permutation_here_b)

        self.auto_gt_assignment_b = QtGui.QPushButton('auto GT')
        self.auto_gt_assignment_b.clicked.connect(self.__auto_gt_assignment)
        self.gt_box.layout().addWidget(self.auto_gt_assignment_b)

        self.auto_gt_assignment_action = QtGui.QAction('save gt', self)
        self.auto_gt_assignment_action.triggered.connect(self.__auto_gt_assignment)
        self.auto_gt_assignment_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_A))
        self.addAction(self.auto_gt_assignment_action)

        self.add_gt_markers_b = QtGui.QPushButton('add GT markers')
        self.add_gt_markers_b.clicked.connect(self.__add_gt_markers)

        self.gt_duplicate_from_prev_frame = QtGui.QAction('duplicat gt from previous frame', self)
        self.gt_duplicate_from_prev_frame.triggered.connect(self.__duplicate_gt)
        self.gt_duplicate_from_prev_frame.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_D))
        self.addAction(self.gt_duplicate_from_prev_frame)

        self.big_gt_duplicate_from_prev_frame = QtGui.QAction('duplicat gt from previous frame', self)
        self.big_gt_duplicate_from_prev_frame.triggered.connect(partial(self.__duplicate_gt, 60))
        self.big_gt_duplicate_from_prev_frame.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_D))
        self.addAction(self.big_gt_duplicate_from_prev_frame)

        self.gt_box.layout().addWidget(self.add_gt_markers_b)

        # self.add_gt_markers_action = QtGui.QAction('add gt markers', self)
        # self.add_gt_markers_action.triggered.connect(self.__add_gt_markers)
        # self.add_gt_markers_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_D))
        # self.addAction(self.add_gt_markers_action)

        self.create_gt_from_results_b = QtGui.QPushButton('create gt from results')
        self.create_gt_from_results_b.clicked.connect(self.__create_gt_from_results)
        self.gt_box.layout().addWidget(self.create_gt_from_results_b)

        # TRACKLET BOX
        self.tracklet_box = QtGui.QGroupBox('Tracklet controls')
        self.tracklet_box.setLayout(QtGui.QVBoxLayout())

        self.highlight_tracklet_input = SelectAllLineEdit('tracklet id')
        self.highlight_tracklet_input.returnPressed.connect(self.highlight_tracklet_button_clicked)

        self.tracklet_p_label = QtGui.QLabel('P: ')
        self.tracklet_n_label = QtGui.QLabel('N: ')

        self.highlight_tracklet_button = QtGui.QPushButton('show tracklet')
        self.highlight_tracklet_button.clicked.connect(self.highlight_tracklet_button_clicked)

        self.highlight_region_button = QtGui.QPushButton('show region')
        self.highlight_region_button.clicked.connect(self.highlight_region_button_clicked)

        self.stop_highlight_tracklet = QtGui.QPushButton('stop highlight tracklet')
        self.stop_highlight_tracklet.clicked.connect(self.stop_highlight_tracklet_clicked)

        self.decide_tracklet_button = QtGui.QPushButton('decide tracklet')
        self.decide_tracklet_button.clicked.connect(self.decide_tracklet)
        self.decide_tracklet_button.setDisabled(True)

        self.decide_tracklet_action = QtGui.QAction('decide tracklet', self)
        self.decide_tracklet_action.triggered.connect(self.decide_tracklet)
        self.decide_tracklet_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_D))
        self.addAction(self.decide_tracklet_action)

        self.edit_tracklet_action = QtGui.QAction('edit tracklet', self)
        self.edit_tracklet_action.triggered.connect(self.edit_tracklet)
        self.edit_tracklet_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_D))
        self.addAction(self.edit_tracklet_action)


        self.tracklet_end_button = QtGui.QPushButton('go to tracklet end')
        self.tracklet_end_button.clicked.connect(self.tracklet_end)

        self.id_end_button = QtGui.QPushButton('go to id end')
        self.id_end_button.clicked.connect(lambda x: self.id_end())

        self.tracklet_end_action = QtGui.QAction('tracklet end', self)
        self.tracklet_end_action.triggered.connect(self.tracklet_end)
        self.tracklet_end_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_E))
        self.addAction(self.tracklet_end_action)

        self.id_end_action = QtGui.QAction('id end', self)
        self.id_end_action.triggered.connect(lambda x: self.id_end())
        self.id_end_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_E))
        self.addAction(self.id_end_action)

        self.tracklet_begin_button = QtGui.QPushButton('go to beginning')
        self.tracklet_begin_button.clicked.connect(self.tracklet_begin)

        self.tracklet_begin_action = QtGui.QAction('tracklet begin', self)
        self.tracklet_begin_action.triggered.connect(self.tracklet_begin)
        self.tracklet_begin_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_S))
        self.addAction(self.tracklet_begin_action)

        self.id_begin_button = QtGui.QPushButton('go to id beginning')
        self.id_begin_button.clicked.connect(lambda x: self.id_end(False))

        self.id_begin_action = QtGui.QAction('id begin', self)
        self.id_begin_action.triggered.connect(lambda x: self.id_end(False))
        self.id_begin_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_S))
        self.addAction(self.id_begin_action)


        # goto next / prev node (something like go to next interesting point)
        self.next_graph_node_action = QtGui.QAction('next graph node', self)
        self.next_graph_node_action.triggered.connect(lambda x: self.goto_next_graph_node(frame=None))
        self.next_graph_node_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_N))
        self.addAction(self.next_graph_node_action)

        self.prev_graph_node_action = QtGui.QAction('prev graph node', self)
        self.prev_graph_node_action.triggered.connect(lambda x: self.goto_prev_graph_node(frame=None))
        self.prev_graph_node_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_B))
        self.addAction(self.prev_graph_node_action)

        self.split_tracklet_b = QtGui.QPushButton('split tracklet')
        self.split_tracklet_b.clicked.connect(self.split_tracklet)

        self.tracklet_box.layout().addWidget(self.highlight_tracklet_input)
        self.tracklet_box.layout().addWidget(self.tracklet_p_label)
        self.tracklet_box.layout().addWidget(self.tracklet_n_label)
        self.tracklet_box.layout().addWidget(self.highlight_tracklet_button)
        self.tracklet_box.layout().addWidget(self.highlight_region_button)
        self.tracklet_box.layout().addWidget(self.stop_highlight_tracklet)
        self.tracklet_box.layout().addWidget(self.decide_tracklet_button)
        self.tracklet_box.layout().addWidget(self.tracklet_begin_button)
        self.tracklet_box.layout().addWidget(self.tracklet_end_button)
        self.tracklet_box.layout().addWidget(self.id_begin_button)
        self.tracklet_box.layout().addWidget(self.id_end_button)
        self.tracklet_box.layout().addWidget(self.split_tracklet_b)

        self.left_vbox.addWidget(self.tracklet_box)
        self.info_l = QtGui.QLabel('info')
        self.tracklet_box.layout().addWidget(self.info_l)

        self.debug_box = QtGui.QGroupBox('debug box')
        self.debug_box.setLayout(QtGui.QVBoxLayout())

        self.video_player = VideoPlayer(self.project)
        self.video_player.set_frame_change_callback(self.update_visualisations)

        self.show_results_flayout = QtGui.QFormLayout()

        self.show_results_summary_start_i = QtGui.QLineEdit()
        self.show_results_summary_start_i.setText(str(0))
        self.show_results_flayout.addRow('start: ', self.show_results_summary_start_i)

        self.show_results_summary_steps_i = QtGui.QLineEdit()
        self.show_results_summary_steps_i.setText(str(max(1, int(self.video_player.total_frame_count() / 200))))
        self.show_results_flayout.addRow('step: ', self.show_results_summary_steps_i)

        self.show_results_summary_end_i = QtGui.QLineEdit()
        self.show_results_summary_end_i.setText(str(int(self.video_player.total_frame_count())))
        self.show_results_flayout.addRow('end: ', self.show_results_summary_end_i)

        self.show_summary_b = QtGui.QPushButton('show results summary')
        self.show_summary_b.clicked.connect(self.show_results_summary)
        self.show_results_flayout.addWidget(self.show_summary_b)

        self.debug_box.layout().addLayout(self.show_results_flayout)

        self.show_cs_analysis_b = QtGui.QPushButton('CS analysis')
        self.show_cs_analysis_b.clicked.connect(self.cs_analysis)
        self.debug_box.layout().addWidget(self.show_cs_analysis_b)

        self.reset_chunk_ids_b = QtGui.QPushButton('Reset chunk IDs')
        self.reset_chunk_ids_b.clicked.connect(self.reset_chunk_ids)
        self.debug_box.layout().addWidget(self.reset_chunk_ids_b)

        self.assign_ids_from_gt_b = QtGui.QPushButton('assign ID from GT')
        self.assign_ids_from_gt_b.clicked.connect(self.assign_ids_from_gt)

        self.print_conflic_tracklets_b = QtGui.QPushButton('print conflicts')
        self.print_conflic_tracklets_b.clicked.connect(self.print_conflicts)

        self.print_undecided_tracklets_b = QtGui.QPushButton('print undecided')
        self.print_undecided_tracklets_b.clicked.connect(self.print_undecided)
        self.show_idtracker_i = QtGui.QLineEdit('/Users/flipajs/Documents/wd/idTracker/'+self.project.name+'/trajectories.mat')
        self.show_idtracker_b = QtGui.QPushButton('show idtracker')
        self.show_idtracker_b.clicked.connect(self.show_idtracker_data)

        self.head_fix_b = QtGui.QPushButton('head fix')
        self.head_fix_b.clicked.connect(self.head_fix)

        self.show_movement_model_b = QtGui.QPushButton('show movement model')
        self.show_movement_model_b.clicked.connect(self.show_movement_model)

        self.export_video_b = QtGui.QPushButton('export visualisation')
        self.export_video_b.clicked.connect(self.export_visualisation)

        self.update_N_sets_b = QtGui.QPushButton('update N sets')
        self.update_N_sets_b.clicked.connect(self.update_N_sets)

        self.debug_box.layout().addWidget(self.print_conflic_tracklets_b)
        self.debug_box.layout().addWidget(self.print_undecided_tracklets_b)
        self.debug_box.layout().addWidget(self.assign_ids_from_gt_b)
        self.debug_box.layout().addWidget(self.show_idtracker_i)
        self.debug_box.layout().addWidget(self.show_idtracker_b)
        self.debug_box.layout().addWidget(self.head_fix_b)
        self.debug_box.layout().addWidget(self.show_movement_model_b)
        self.debug_box.layout().addWidget(self.export_video_b)
        self.debug_box.layout().addWidget(self.update_N_sets_b)

        self.left_vbox.addWidget(self.debug_box)
        self.left_vbox.addWidget(self.gt_box)

        # TODO: show list of tracklets instead of QLine edit...
        # TODO: show range on frame time line
        # TODO: checkbox - stop at the end...

        self.right_w = QtGui.QWidget()
        self.right_w.setLayout(self.right_vbox)
        self.splitter.addWidget(self.right_w)

        self.hbox.addWidget(self.splitter)

        self.video_layout = QtGui.QVBoxLayout()
        self.right_vbox.addLayout(self.video_layout)

        self.video_layout.addWidget(self.video_player)

        self.hide_visualisation_action = QtGui.QAction('hide visualisation', self)
        self.hide_visualisation_action.triggered.connect(self.hide_visualisation)
        self.hide_visualisation_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_H))
        self.addAction(self.hide_visualisation_action)

        self.visu_controls_layout = FlowLayout()
        self.video_layout.addLayout(self.visu_controls_layout)

        self.show_filled_ch = QtGui.QCheckBox('filled')
        self.show_filled_ch.setChecked(True)
        # lambda is used because if only self.update_position is given, it will give it parameters...
        self.show_filled_ch.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.show_filled_ch)

        self.show_tracklets_without_id = QtGui.QCheckBox('tracklet w/o id')
        self.show_tracklets_without_id.setChecked(True)
        # lambda is used because if only self.update_position is given, it will give it parameters...
        self.show_tracklets_without_id.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.show_tracklets_without_id)

        self.show_contour_ch = QtGui.QCheckBox('contours')
        self.show_contour_ch.setChecked(False)
        # lambda is used because if only self.update_position is given, it will give it parameters...
        self.show_contour_ch.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.show_contour_ch)

        self.contour_without_colors = QtGui.QCheckBox('id clrs')
        self.contour_without_colors.setChecked(True)
        self.contour_without_colors.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.contour_without_colors)

        self.show_markers = QtGui.QCheckBox('GT markers')
        self.show_markers.setChecked(True)
        self.show_markers.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.show_markers)

        self.show_saturated_ch = QtGui.QCheckBox('saturate img')
        self.show_saturated_ch.setChecked(False)
        self.show_saturated_ch.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.show_saturated_ch)

        self.show_saturated_action = QtGui.QAction('show saturated', self)
        self.show_saturated_action.triggered.connect(lambda x: self.show_saturated_ch.setChecked(not self.show_saturated_ch.isChecked()))
        self.show_saturated_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_S))
        self.addAction(self.show_saturated_action)

        self.show_id_bar = QtGui.QCheckBox('id bar')
        self.show_id_bar.setChecked(True)
        self.show_id_bar.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.show_id_bar)

        self.show_tracklet_class = QtGui.QCheckBox('show t-cardinality')
        self.show_tracklet_class.setToolTip('Show region cardinality classification.')
        self.show_tracklet_class.setChecked(False)
        self.show_tracklet_class.stateChanged.connect(lambda x: self.redraw_video_player_visualisations())
        self.visu_controls_layout.addWidget(self.show_tracklet_class)

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

        self.splitter.setSizes([270, 1500])

        self.old_crops = [None] * len(self.project.animals)

        self.colors_ = [
            QtGui.QColor().fromRgbF(0, 0, 1), #
            QtGui.QColor().fromRgbF(1, 0, 0),
            QtGui.QColor().fromRgbF(1, 1, 0),
            QtGui.QColor().fromRgbF(0, 1, 0), #
            QtGui.QColor().fromRgbF(0, 1, 1),
            QtGui.QColor().fromRgbF(1, 1, 1)
        ]

        # TODO: add develop option to load from project file...


        self._load_gt()

        try:
            if len(self.project.animals) == len(self.project.chm.chunks_in_frame(self.video_player.current_frame())):
                self._gt_(True)
        except:
            pass

        # self.redraw_video_player_visualisations()

    def reset_chunk_ids(self):
        msg = "Do you really want to delete all assigned ids to chunks?"
        reply = QtGui.QMessageBox.question(self, 'Message',
                                           msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

        if reply == QtGui.QMessageBox.Yes:
            for ch in self.project.chm.chunk_gen():
                ch.P = set()
                ch.N = set()

            self.video_player.redraw_visualisations()

    def decide_tracklet(self):
        if self.active_tracklet_id > -1:
            self.decide_tracklet_callback(self.project.chm[self.active_tracklet_id])
            # self._set_active_tracklet_id(-1)
            self.decide_tracklet_button.setDisabled(True)

            self.redraw_video_player_visualisations()

            self.setFocus()

    def edit_tracklet(self):
        if self.active_tracklet_id > -1:
            if self.edit_tracklet_callback:
                self.edit_tracklet_callback(self.project.chm[self.active_tracklet_id])

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

    def id_end(self, forward=True):
        frame_step = 1 if forward else -1

        if self.active_tracklet_id > -1:
            tid = self.active_tracklet_id

            t = self.project.chm[tid]
            if t.is_only_one_id_assigned(len(self.project.animals)):
                id_ = list(t.P)[0]
            else:
                return

            while True:
                if forward:
                    frame = t.end_frame(self.project.gm)
                else:
                    frame = t.start_frame(self.project.gm)

                new_t_found = False
                for new_t in self.project.chm.chunks_in_frame(frame+frame_step):
                    if new_t.is_only_one_id_assigned(len(self.project.animals)):
                        new_id = list(new_t.P)[0]

                        if id_ == new_id:
                            self.active_tracklet_id = new_t.id()
                            t = new_t
                            if forward:
                                frame = t.end_frame(self.project.gm)
                            else:
                                frame = t.start_frame(self.project.gm)
                            new_t_found = True

                if not new_t_found:
                    break

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

    def highlight_region_button_clicked(self):
        try:
            id_ = int(self.highlight_tracklet_input.text())
            r = self.project.rm[id_]
        except:
            return

        frame = r.frame()

        t_id = None
        for t in self.project.chm.chunks_in_frame(frame):
            if id_ in t.rid_gen(self.project.gm):
                t_id = t.id()
                break


        self.video_player.goto(frame)
        print id_

        if t_id:
            self._set_active_tracklet_id(t_id)

    def highlight_tracklet_button_clicked(self):
        self._highlight_tracklets = set()
        try:
            id_ = int(self.highlight_tracklet_input.text())
            tracklet = self.project.chm[id_]
        except Exception as e:
            print e
            return

        if tracklet is None:
            return

        frame = tracklet.start_frame(self.project.gm)
        self.video_player.goto(frame)
        print id_
        self._set_active_tracklet_id(id_)

    def play_and_highlight_tracklet(self, tracklet, frame=-1, margin=0):
        # return

        # frame=-1 ... start from beginning

        self.loop_begin = max(0, tracklet.start_frame(self.project.gm) - margin)
        self.loop_end = min(tracklet.end_frame(self.project.gm) + margin, self.video_player.total_frame_count()-1)
        self.loop_highlight_tracklets = [tracklet.id()]

        if frame < 0:
            frame = self.loop_begin

        self.video_player.goto(frame)

        self._set_active_tracklet_id(tracklet.id())
        self._highlight_tracklets.add(tracklet)

        import warnings
        warnings.warn('not fully reimplemented in video_player', UserWarning)

        # self.timer.stop()
        self.video_player.play()
        self.video_player.setFocus()

    def _evaluate_gt(self):
        from utils.gt.evaluator import compare_trackers
        compare_trackers(self.project, skip_idtracker=True, gt_ferda_perm=self._gt.get_permutation_reversed(),
                         gt=self._gt, draw=self.eval_gt_draw_ch.isChecked())

    def _evolve_gt_id(self):
        from utils.gt.evaluator import Evaluator
        if self._gt is not None:
            ev = Evaluator(None, self._gt)
            ev.eval_ids(self.project)

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

    def update_N_sets(self):
        if self.update_N_sets_callback:
            self.update_N_sets_callback()
        else:
            print "update N sets callback not present in results_widget"


    def __save_gt(self):
        self._gt.save(self.project.GT_file)
        print "GT saved..."

        # if self._gt is None:
        #     print "No GT file opened"
        #     return
        #
        # frame = self.video_player.current_frame()
        # self._gt.setdefault(frame, [None]*len(self.project.animals))
        #
        # for it in self._gt_markers.itervalues():
        #     self._gt[frame][it.id] = (it.centerPos().y(), it.centerPos().x())
        #
        # with open(self.project.GT_file, 'wb') as f:
        #     pickle.dump(self._gt, f)
        #
        # print self._gt[frame]

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
        if r.is_origin_interaction():
            y, x = r.centroid()
            # TODO:
            a = 50
            b = 17

            item = QtGui.QGraphicsEllipseItem(-a/2, -b/2, a, b)
            brush = QtGui.QBrush(QtCore.Qt.SolidPattern)

            opacity = 0.5
            if len(tracklet.P):
                id_ = list(tracklet.P)[0]
                c_ = self.project.animals[id_].color_
                c_ = QtGui.QColor(c_[2], c_[1], c_[0], alpha)
            else:
                c_ = QtGui.QColor(use_ch_color.red(), use_ch_color.green(), use_ch_color.blue())
                opacity = 1.0

            brush.setColor(c_)
            item.setBrush(brush)
            item.setOpacity(opacity)
            item.setZValue(0.65)
            item.rotate(np.rad2deg(r.theta_))
            item.setPos(x, y)
            return item

        # hnus... prepsat
        from utils.img import get_cropped_pts

        alpha = S_.visualization.segmentation_alpha
        is_id_tracklet = tracklet.is_only_one_id_assigned(len(self.project.animals))
        only_contour = False if self.show_filled_ch.isChecked() else True
        # we need region pts to dilate and substract previous pts to get result
        only_contour = only_contour and not highlight_contour

        if not is_id_tracklet and self.show_contour_ch.isChecked() and not self.show_tracklets_without_id.isChecked():
            only_contour = True

        if not S_.visualization.no_single_id_filled and not tracklet.is_only_one_id_assigned(len(self.project.animals)):
            only_contour = True

        pts_, roi = get_cropped_pts(r, return_roi=True, only_contour=only_contour)
        if highlight_contour:
            pts_, roi = self.__dilate(pts_, roi)

        offset = roi.top_left_corner()

        # qim_ = QtGui.QImage(roi.width(), roi.height(), QtGui.QImage.Format_ARGB32)
        # qim_.fill(QtGui.qRgba(0, 0, 0, 0))

        step = 1

        if force_color is None:
            c = S_.visualization.default_region_color

            if tracklet.length() == 1:
                c = c.blue(), c.green(), c.red(), c.alpha()
            else:
                c = c.red(), c.green(), c.blue(), c.alpha()

            if self.contour_without_colors.isChecked():
                if is_id_tracklet:
                    id_ = list(tracklet.P)[0]
                    c_ = self.project.animals[id_].color_
                    c = c_[2], c_[1], c_[0], alpha
                else:
                    if not self.show_tracklets_without_id.isChecked() and not self.show_contour_ch.isChecked():
                        return

                    step = 3
                    if use_ch_color:
                        c = use_ch_color.red(), use_ch_color.green(), use_ch_color.blue(), alpha
            elif use_ch_color:
                c = use_ch_color.red(), use_ch_color.green(), use_ch_color.blue(), alpha
        else:
            c = force_color

        if r.is_origin_interaction():
            step = 1

        if self.show_tracklet_class.isChecked():
            step = 1
            c = self.get_tracklet_class_color(tracklet)
            c = (c[0], c[1], c[2], 255)

        rgba = np.zeros((roi.width(), roi.height(), 4), dtype=np.uint8)
        # 2 1 0 BGR vs RGB...
        # this if might help a little bit with performance...
        if step > 1:
            rgba[pts_[::step, 1], pts_[::step, 0], :] = c[2], c[1], c[0], c[3]
        else:
            rgba[pts_[:, 1], pts_[:, 0], :] = c[2], c[1], c[0], c[3]

        # TODO: do other way if possible. This is the only place where pyqtgraph is needed.
        qim_ = makeQImage(rgba)

        pixmap = QtGui.QPixmap.fromImage(qim_)

        item = ClickableQGraphicsPixmapItem(pixmap, tracklet.id(), self._highlight_tracklet)
        item.setPos(offset[1], offset[0])
        item.setZValue(0.6)

        return item

    def get_tracklet_class_color(self, tracklet):
        """
        returns r, g, b color representation for tracklet class
        Args:
            tracklet: 

        Returns:

        """

        if tracklet.is_single():
            return (4, 204, 14)
        elif tracklet.is_multi():
            return (28, 190, 255)
        elif tracklet.is_noise():
            return (255, 0, 0)
        elif tracklet.is_part():
            return (0, 190, 0)
        else:
            return (128, 128, 128)

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

    def __add_marker(self, x, y, c_, id_, z_value, type_, radius=13, alpha=None):
        if type_ == 'GT':
            radius = 20
        elif type_ == 'multiple':
            radius = 9

        if type_ == 'tracker1':
            radius = 7

        m = markers.CenterMarker(0, 0, radius, c_, id=id_, changeHandler=self._gt_marker_clicked)

        if alpha:
            m.setOpacity(alpha)

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

            positions = self._gt.get_clear_positions(frame)
            if a.id < len(positions):
                data = positions[a.id]
                if data is None:
                    y = 10
                    x = 10 * a.id
                else:
                    y = data[0]
                    x = data[1]
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

    def draw_history(self):
        # TODO: settings
        radius = 13

        num_animals = len(self.project.animals)
        current_frame = self.video_player.current_frame()
        h_depth = S_.visualization.history_depth
        for frame in range(max(0, current_frame - h_depth), current_frame, S_.visualization.history_depth_step):
            h = h_depth - (current_frame - frame)
            h_ratio = (h / float(h_depth))
            a = S_.visualization.basic_marker_opacity
            exponent = S_.visualization.history_alpha
            alpha = a * (h_ratio) ** exponent

            r_ = max(1, int(round(radius * h_ratio)))

            if alpha < 0.01:
                continue

            for ch in self.project.chm.chunks_in_frame(frame):
                if ch.is_only_one_id_assigned(num_animals):
                    rch = RegionChunk(ch, self.project.gm, self.project.rm)
                    r = rch.region_in_t(frame)
                    centroid = r.centroid()

                    id_ = list(ch.P)[0]
                    c_ = self.project.animals[id_].color_
                    c_ = QtGui.QColor(c_[2], c_[1], c_[0])

                    y = centroid[0]
                    x = centroid[1]

                    self.__add_marker(x, y, c_, None, 0.5, type_='default', radius=r_, alpha=alpha)

    def draw_id_profiles(self):
        from utils.img import get_safe_selection
        from gui.img_controls.gui_utils import cvimg2qtpixmap
        from core.animal import colors_

        frame = self.video_player.current_frame()
        img = self.video_player._vm.img().copy()

        A = 50
        it_x = img.shape[1] + 1

        stripe_w = 10
        for i in range(len(self.project.animals)):
            it_y = (2*A) * i
            s = np.zeros((2*A, stripe_w, 3), dtype=np.uint8)
            for j in range(3):
                s[:, :, j] = self.project.animals[i].color_[j]

            pixmap = cvimg2qtpixmap(s)
            pixmap = QtGui.QGraphicsPixmapItem(pixmap)
            pixmap.setPos(it_x, it_y)

            self.video_player.visualise_temp(pixmap)

        it_x += stripe_w + 1

        num_animals = len(self.project.animals)
        idset = set(range(num_animals))
        for ch in self.project.chm.chunks_in_frame(frame):
            if len(ch.P) == 1:
                id_ = list(ch.P)[0]
                it_y = (2 * A) * id_
            else:
                continue

            idset.discard(id_)

            rch = RegionChunk(ch, self.project.gm, self.project.rm)
            r = rch.region_in_t(frame)

            if r is None:
                continue

            cenY, cenX = r.centroid()

            crop = get_safe_selection(img, cenY-A, cenX-A, 2*A, 2*A, fill_color=(0, 0, 0))

            if self.show_saturated_ch.isChecked():
                crop = self._img_saturation(crop)

            self.old_crops[id_] = crop

            pixmap = cvimg2qtpixmap(crop)
            pixmap = QtGui.QGraphicsPixmapItem(pixmap)
            pixmap.setPos(it_x, it_y)

            self.video_player.visualise_temp(pixmap)

        DARKEN = 25
        DARKEN_COLOR = 35

        # TODO: if frame seek, reset self.old_crops

        # display last positions of unknown...
        for id_ in idset:
            crop = self.old_crops[id_]

            if crop is None:
                continue

            it_y = (2 * A) * id_

            crop = np.asarray(np.maximum(0, np.asarray(self.old_crops[id_], dtype=np.int) - DARKEN), dtype=np.uint8)
            crop[A, :, :] = 20
            pixmap = cvimg2qtpixmap(crop)
            pixmap = QtGui.QGraphicsPixmapItem(pixmap)
            pixmap.setPos(it_x, it_y)

            self.video_player.visualise_temp(pixmap)

            s = np.zeros((2 * A, stripe_w, 3), dtype=np.uint8)
            for j in range(3):
                s[:, :, j] = self.project.animals[id_].color_[j]

            val_ = int((2*A) / 9)
            s[4*val_:5*val_, :, :] = 20

            pixmap = cvimg2qtpixmap(s)
            pixmap = QtGui.QGraphicsPixmapItem(pixmap)
            pixmap.setPos(it_x - stripe_w - 1, it_y)

            self.video_player.visualise_temp(pixmap)


    def update_visualisations(self):
        try:
            self.draw_id_profiles()
        except:
            pass

        self._update_tracklet_info(from_update_visu=True)

        # with open('/Users/flipajs/Desktop/temp/pairs/' + 'exp1' + '/head_rfc.pkl', 'rb') as f:
        #     rfc = pickle.load(f)

        if self.hide_visualisation_:
            return

        # reset...
        self._gt_markers = {}
        frame = self.video_player.current_frame()

        animal_ids2centroids = {}

        for ch in self.project.chm.chunks_in_frame(frame):
            # try:
                rch = RegionChunk(ch, self.project.gm, self.project.rm)
                r = rch.region_in_t(frame)

                # print r.id_
                # if r.id_ > 22612:
                #     print r.id_

                if r is None:
                    print ch
                    continue

                centroid = r.centroid().copy()

                if self.show_id_bar.isChecked() and len(ch.N) < len(self.project.animals)-1:
                    try:
                        item = self.show_pn_ids_visualisation(ch, frame)

                        self.video_player.visualise_temp(item, category='id_bar')
                    except AttributeError:
                        pass

                if self.show_contour_ch.isChecked() or self.show_filled_ch.isChecked():
                    alpha = self.alpha_filled if self.show_filled_ch.isChecked() else self.alpha_contour

                    c_ = ch.color
                    if c_ is None:
                        import random
                        c = QtGui.QColor.fromRgb(random.randint(0, 255),
                                                 random.randint(0, 255),
                                                 random.randint(0, 255))
                        ch.color = c
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

            # except:
            #     pass

                # from scripts.regions_stats import fix_head
                # fix_head(self.project, r, rfc)

                # head, _ = get_region_endpoints(r)
                # head_item = markers.CenterMarker(head[1], head[0], 5, QtGui.QColor(0, 0, 0), ch.id(),
                #                                  self._gt_marker_clicked)
                #
                # head_item.setZValue(0.95)
                #
                # self.video_player.visualise_temp(head_item, category='head')

        if S_.visualization.history_depth > 0:
            self.draw_history()

        if self.show_markers.isChecked():
            self._show_gt_markers()
            # self._show_id_markers(animal_ids2centroids)

        if self.idtracker_data is not None:
            for id_, it in enumerate(self.idtracker_data[frame]):
                id_ = self.idtracker_data_permutation[id_]
                a = self.project.animals[id_]
                c_ = QtGui.QColor(a.color_[2], a.color_[1], a.color_[0])
                x, y = it[0], it[1]

                if np.isnan(x):
                    x, y = 10*(id_+1), 10

                type_ = 'tracker1'

                w = 10
                item = QtGui.QGraphicsRectItem(0, 0, w, w)
                brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
                brush.setColor(c_)
                item.setBrush(brush)
                item.setOpacity(0.8)
                item.setZValue(1.0)
                item.setPos(x-w/2, y-w/2)

                self.video_player.visualise_temp(item, category=type_)

                # self.__add_marker(x, y, c_, None, 0.75, type_=type_)

    def _img_saturation(self, img):
        return img_saturation_coef(img, 1.5, 0.95)

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

        tlen = 0
        ptr = 0
        p_ = S_.visualization.tracklet_len_per_px
        if p_:
            tlen = int(len(tracklet)/float(p_))
            ptr = int((frame - tracklet.start_frame(self.project.gm)) / float(p_))

        c = self.get_tracklet_class_color(tracklet)

        id_ = list(tracklet.P)
        virtual_id = False
        if len(tracklet.P) and id_ >= len(self.project.animals):
            virtual_id = True
        item = pn_ids_visualisation.get_pixmap_item(ids_, tracklet.P, tracklet.N,
                                                     tracklet_id=tracklet.id(),
                                                     callback=self.pn_pixmap_clicked,
                                                    # TODO: probs on Demand
                                                     probs=None,
                                                     # probs=tracklet.animal_id_['probabilities'],
                                                     params=params, tracklet_len=tlen, tracklet_ptr=ptr,
                                                     tracklet_class_color=c,
                                                     virtual_id=virtual_id
                                                     )

        reg = rch.region_in_t(frame)
        item.setPos(reg.centroid()[1], reg.centroid()[0])
        item.setZValue(0.9)
        item.setFlags(QtGui.QGraphicsItem.ItemIsMovable)

        return item

    def pn_pixmap_clicked(self, id_):
        self._highlight_tracklet(id_)

    def __compute_radius(self, r):
        from scipy.spatial.distance import cdist
        c = r.centroid()
        c.shape = (1, 2)
        D = cdist(c, r.contour())
        radius = np.max(D)

        return radius

    def _update_tracklet_info(self, from_update_visu=False):
        import textwrap
        id_ = self.active_tracklet_id
        if id_ < 0:
            self.info_l.setText(' ')
            self.tracklet_p_label.setText(' ')
            self.tracklet_n_label.setText(' ')

            return

        ch = self.project.chm[id_]
        f = self.video_player.current_frame()
        rch = RegionChunk(ch, self.project.gm, self.project.rm)
        r = rch.region_in_t(f)

        s = "tracklet (id: " + str(id_) + ")"
        if ch.start_frame(self.project.gm) == f:
            s += "\n in_degree: " + str(ch.start_vertex(self.project.gm).in_degree())

        if ch.end_frame(self.project.gm) == f:
            s += "\n out degree: " + str(ch.end_vertex(self.project.gm).out_degree())

        s += "\n length: " + str(ch.length()) + " s: " + str(ch.start_frame(self.project.gm)) + " e: " + str(
            ch.end_frame(self.project.gm))

        s += "\n tracklet class: "+ch.segmentation_class_str()

        if hasattr(ch, 'id_decision_info'):
            s += "\n idDecision: {}".format(ch.id_decision_info)

        # avg_area = 0
        # for r_ in rch.regions_gen():
        #     avg_area += r_.area()
        #
        # avg_area /= rch.chunk_.length()

        # s += "\n avg area: " + str(avg_area)
        # s += "\n avg area: " + str(avg_area)

        if r:
            s += "\n\nregion (id: "+str(r.id())+")"
            s += "\n " + textwrap.fill(str(r), 40)
            s += " radius: {:.3}\n".format(self.__compute_radius(r))

        # if self.tracklet_measurements is not None:
        #     s += "\nTracklet ID probs: \n"
        #     try:
        #         vals = np.mean(self.tracklet_measurements(ch.id()))
        #     except TypeError:
        #         vals = self.tracklet_measurements(ch.id())
        #
        #     if vals is not None:
        #         for i, a in enumerate(vals.flatten()):
        #             s += "\t{}: {:.2%}\n".format(i, a)

        self.info_l.setText(s)

        self.tracklet_p_label.setText('P: '+str(ch.P))
        self.tracklet_n_label.setText('N: '+str(ch.N))

        # TODO: solve better... something like set
        # self._highlight_tracklets = set()
        # self._highlight_tracklets.add(id_)
        if not from_update_visu:
            self.video_player.redraw_visualisations()

    def _gt_marker_clicked(self, id_):
        # try:
        frame = self.video_player.current_frame()
        y, x = self._gt_markers[id_].centerPos().y(), self._gt_markers[id_].centerPos().x()
        self._gt.set_position(frame, id_, y, x)
        # except:
        #     pass

        # self._highlight_tracklet(id_)

        self.video_player.redraw_visualisations()

    def _highlight_tracklet(self, id_):
        try:
            self.decide_tracklet_button.setDisabled(False)
            self._set_active_tracklet_id(id_)
            self.highlight_tracklet_input.setText(str(id_))

            self._update_tracklet_info()
        except:
            pass


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
                        col_ = self.project.animals[animal_id].color_
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
            # TODO: remove this... temp skip decided.
            if len(t.P) == 1:
                continue

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

    def goto_next_graph_node(self, frame=None):
        if frame is None:
            frame = self.video_player.current_frame()

        min_frame = self.video_player.total_frame_count() - 1
        t_id = -1

        for t in self.project.chm.chunks_in_frame(frame):
            if min_frame > t.end_frame(self.project.gm):
                min_frame = t.end_frame(self.project.gm)
                t_id = t.id()

        if min_frame == frame and frame < self.video_player.total_frame_count() - 1:
            self.goto_next_graph_node(frame=frame+1)
            return

        self.video_player.goto(min_frame)
        if t_id > -1:
            self._set_active_tracklet_id(t_id)
            self.video_player.redraw_visualisations()

    def goto_prev_graph_node(self, frame=None):
        if frame is None:
            frame = self.video_player.current_frame()

        max_frame = 0
        t_id = -1
        for t in self.project.chm.chunks_in_frame(frame):
            if max_frame < t.start_frame(self.project.gm):
                max_frame = t.start_frame(self.project.gm)
                t_id = t.id()

        if max_frame == frame and frame > 0:
            self.goto_prev_graph_node(frame-1)
            return

        self.video_player.goto(max_frame)
        if t_id > -1:
            self._set_active_tracklet_id(t_id)
            self.video_player.redraw_visualisations()

    def __get_gt_stats(self, gt):
        # TODO move to utils.gt.gt
        frames = sorted(map(int, gt.iterkeys()))

        num_ids = len(gt[frames[0]])
        id_coverage = np.zeros((num_ids, ))

        for f in frames:
            for id_, data in enumerate(gt[f]):
                if data is not None:
                    id_coverage[id_] += 1

        print "--- GT info ---"
        print "#frames: ", len(frames)
        print "ID coverage: "
        for i in range(num_ids):
            print " {}:{:.2%}".format(i, id_coverage[i] / float(len(frames)))

    def __create_gt_from_results(self):
        from utils.gt.gt import GT

        if not hasattr(self.project, 'GT_file'):
            path = self.project.working_directory + '/' + 'GT.pkl'
        else:
            path = self.project.GT_file

        self._gt = GT()
        self._gt.set_offset(y=self.project.video_crop_model['y1'],
                            x=self.project.video_crop_model['x1'],
                            frames=self.project.video_start_t)

        self._gt.build_from_PN(self.project)

        self._gt.save(path)

    def load_gt_file_dialog(self):
        import os
        import gui.gui_utils

        path = ''
        if os.path.isdir(S_.temp.last_gt_path):
            path = S_.temp.last_gt_path

        self.project.GT_file = gui.gui_utils.file_name_dialog(self, 'Select GT file', filter_="Pickle (*.pkl)", path=path)

        if self.project.GT_file:
            S_.temp.last_gt_path = os.path.dirname(self.project.GT_file)

        self._load_gt()
        self.video_player.redraw_visualisations()

    def _load_gt(self):
        from utils.gt.gt import GT
        self._gt = GT()
        # self._gt.set_offset(y=self.project.video_crop_model['y1'],
        #                        x=self.project.video_crop_model['x1'],
        #                        frames=self.project.video_start_t)

        try:
            path = self.project.GT_file
            self._gt.load(path)
            self._gt.set_offset(y=self.project.video_crop_model['y1'],
                                x=self.project.video_crop_model['x1'],
                                frames=self.project.video_start_t)
        except:
            self._gt = None

    def assign_ids_from_gt(self):
        import warnings
        warnings.warn("this method is outdated and assigns IDs also to multi-regions...")

        # for frame, data in self._gt.iteritems():
        for frame, data in self._gt.get_clear_positions_dict().iteritems():
            print frame
            matches = [list() for _ in range(len(self.project.animals))]
            for t in self.project.chm.chunks_in_frame(frame):
                rch = RegionChunk(t, self.project.gm, self.project.rm)
                c = rch.centroid_in_t(frame)

                best_d = 50
                best_id = -1
                for id_, c2 in enumerate(data):
                    if c2 is not None:
                        d = ((c[0]-c2[0])**2 + (c[1]-c2[1])**2)**0.5
                        if best_d > d:
                            best_d = d
                            best_id = id_

                if best_id > -1:
                    matches[best_id].append(t)

            for id_, arr in enumerate(matches):
                if len(arr) == 1:
                    tracklet = arr[0]
                    if len(tracklet.P) != 1:
                        tracklet.P = set([id_])
                        tracklet.N = set(range(len(self.project.animals))) - set([id_])
                        # TODO: update this...
                        # self.decide_tracklet_callback(tracklet, id_)

    def print_conflicts(self):
        from tqdm import tqdm
        print "CONFLICTS: "
        print " A) P N intersection (len(P.intersection(N)) > 0"

        problems = []
        for t in tqdm(self.project.chm.chunk_gen()):
            if len(t.P.intersection(t.N)):
                problems.append(t)

        for t in problems:
            print "\t", t, t.P, t.N

        problemsB = []
        problemsC = []

        frame = 0
        full_set = set(range(len(self.project.animals)))
        max_f = self.video_player.total_frame_count()
        while True:
            possible_ids = set()
            used_ids = set()

            next_frame = np.inf
            for t in self.project.chm.chunks_in_frame(frame):
                if len(t.P.intersection(used_ids)):
                    problemsC.append((frame, t))

                used_ids = used_ids.union(t.P)

                possible_ids = possible_ids.union(full_set - t.N)
                next_frame = min(next_frame, t.end_frame(self.project.gm))

            if len(possible_ids) != len(self.project.animals):
                problemsB.append((frame, full_set - possible_ids))

            frame = next_frame + 1

            if frame >= max_f:
                break

        print " B) ID missing (frames where it is not possible to distribute all IDs)"
        for frame, x in problemsB:
            print "frame: {}, missing IDs: {}".format(frame, x)

        print " C) ID duplicate (frames where one ID is assigned to multiple regions)"
        for frame, x in problemsC:
            print "frame: {}, tracklet id: {} id: {}".format(frame, x.id(), x.P)




    def print_undecided(self):
        print "UNDECIDED: "
        for t in self.project.chm.chunk_gen():
            if len(t.P.union(t.N)) != len(self.project.animals):
                print t, t.P, t.N

    def split_tracklet(self):
        import random

        if self.active_tracklet_id > -1:
            tracklet = self.project.chm[self.active_tracklet_id]
            frame = self.video_player.current_frame()

            left_nodes, right_nodes = tracklet.split_at(frame, self.project.gm)
            if len(left_nodes) and len(right_nodes):
                self.project.chm.remove_chunk(tracklet, self.project.gm)

                ch1, _ = self.project.chm.new_chunk(left_nodes, self.project.gm)
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                ch1.color = QtGui.QColor.fromRgb(r, g, b)

                ch2, _ = self.project.chm.new_chunk(right_nodes, self.project.gm)
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                ch2.color = QtGui.QColor.fromRgb(r, g, b)

        self.video_player.redraw_visualisations()

    def show_idtracker_data(self):
        path = str(self.show_idtracker_i.text())
        self.idtracker_data, self.idtracker_data_permutation = load_idtracker_data(path, self.project, self._gt)

    def __head_fix(self, tracklet):
        import heapq
        rch = RegionChunk(tracklet, self.project.gm, self.project.rm)

        q = []

        # q = Queue.PriorityQueue()
        heapq.heappush(q, (0, [False]))
        heapq.heappush(q, (0, [True]))
        # q.put((0, [False]))
        # q.put((0, [True]))

        result = []
        i = 0
        max_i = 0

        cut_diff = 10

        while True:
            i += 1

            # cost, state = q.get()
            cost, state = heapq.heappop(q)
            if len(state) > max_i:
                max_i = len(state)

            if len(state) + cut_diff < max_i:
                continue

            # print i, cost, len(state), max_i

            if len(state) == tracklet.length():
                result = state
                break

            prev_r = rch[len(state) - 1]
            r = rch[len(state)]

            prev_c = prev_r.centroid()
            p1, p2 = get_region_endpoints(r)

            dist = np.linalg.norm
            d1 = dist(p1 - prev_c)
            d2 = dist(p2 - prev_c)

            prev_head, prev_tail = get_region_endpoints(prev_r)
            if state[-1]:
                prev_head, prev_tail = prev_tail, prev_head

            d3 = dist(p1 - prev_head) + dist(p2 - prev_tail)
            d4 = dist(p1 - prev_tail) + dist(p2 - prev_head)

            # state = list(state)
            state2 = list(state)
            state.append(False)
            state2.append(True)

            new_cost1 = d3
            new_cost2 = d4

            # TODO: param
            if dist(prev_c - r.centroid()) > 5:
                new_cost1 += d2 - d1
                new_cost2 += d1 - d2

            heapq.heappush(q, (cost + new_cost1, state))
            heapq.heappush(q, (cost + new_cost2, state2))
            # q.put((cost + new_cost1, state))
            # q.put((cost + new_cost2, state2))

        for b, r in zip(result, rch.regions_gen()):
            if b:
                r.theta_ += np.pi
                if r.theta_ >= 2 * np.pi:
                    r.theta_ -= 2 * np.pi

        # del q

    def head_fix(self):
        if self.active_tracklet_id < 0:
            for t in self.project.chm.chunk_gen():
                print t.id(), t.length()
                self.__head_fix(t)
        else:
            self.__head_fix(self.project.chm[self.active_tracklet_id])

    def QImageToCvMat(self, incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format_RGB32)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.constBits()
        arr = np.array(ptr).reshape(height, width, 3)  # Copies the data
        return arr

    def export_visualisation(self):
        import os

        start = 0
        end = 4500

        a = self.project.working_directory.split('/')[-1]
        wd = '/Users/flipajs/Desktop/temp/'+a+'/'

        try:
            os.mkdir(wd)
        except:
            pass

        self.video_player.goto(start)

        # qim_ = QtGui.QImage(1000, 1102, QtGui.QImage.Format_RGB32)

        # painter = QtGui.QPainter(pixmap)
        # self.video_player._scene.render(painter)
        # pixmap.save('out1.png')

        # del painter
        for i in range(end-start):
            # qim_ = QtGui.QImage(2022, 1080, QtGui.QImage.Format_RGB32)
            # qim_ = QtGui.QImage(1102, 1000, QtGui.QImage.Format_RGB32)
            qim_ = QtGui.QImage(1102, 732, QtGui.QImage.Format_RGB32)
            qim_.fill(QtGui.qRgba(0, 0, 0, 0))
            pixmap = QtGui.QPixmap.fromImage(qim_)

            painter = QtGui.QPainter(pixmap)
            self.video_player._scene.render(painter)
            name = str(self.video_player.current_frame())
            while len(name) != 5:
                name = '0'+name

            pixmap.save(wd+name+'.png')
            del painter

            self.video_player.next()
            pass

        # TODO: wait

        pass

    def __duplicate_gt(self, max_history=1):
        frame = self.video_player.current_frame()

        missing_in_current_frame = []
        found_in_frame = {}
        for i, a in enumerate(self._gt.get_clear_positions(frame)):
            if a is None:
                missing_in_current_frame.append(i)
                found_in_frame[i] = -1
            else:
                found_in_frame[i] = frame + 1

        gt_pos = {}
        for prev_frame in reversed(range(max(0, frame-max_history), frame)):
            for i, a in enumerate(self._gt.get_clear_positions(prev_frame)):
                if a is not None:
                    if found_in_frame[i] < 0:
                        found_in_frame[i] = prev_frame
                        gt_pos[i] = a

        for id_ in range(len(self.project.animals)):
            if id_ in gt_pos:
                for f in range(found_in_frame[id_], frame + 1):
                    self._gt.set_position(f, id_, gt_pos[id_][0], gt_pos[id_][1])

        self.video_player.redraw_visualisations()


    def show_movement_model(self):
        from scripts.regions_stats import hist_query, get_movement_descriptor_

        with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/movement_data.pkl', 'rb') as f:
            data = pickle.load(f)

        data = np.array(data)

        H, edges = np.histogramdd(data, bins=10)
        THRESH = 100.0

        frame = self.video_player.current_frame()

        if self.active_tracklet_id > 0:
            t = self.project.chm[self.active_tracklet_id]
            sf = t.start_frame(self.project.gm)
            vertex = t[frame - sf]
            if isinstance(vertex, int):
                vertex = self.project.gm.g.vertex(vertex)

            v1 = None
            if frame != t.start_frame(self.project.gm) and t.length() > 1:
                v1 = self.project.gm.g.vertex(t[frame-sf-1])
            else:
                best_e = None
                best_val = 0
                for e in vertex.in_edges():
                    val = self.project.gm.g.ep['score'][e]
                    if val > best_val:
                        best_val = val
                        best_e = e

                if best_e is not None:
                    v1 = best_e.source()

            if v1 is None:
                return

            v2 = vertex

            r1 = self.project.gm.region(v1)
            r2 = self.project.gm.region(v2)

            v1 = r2.centroid() - r1.centroid()

            h_ = 200
            w_ = 200
            data = np.zeros((h_, w_), dtype=np.float)

            offset = np.array([int(r2.centroid()[0] - h_/2), int(r2.centroid()[1] - w_/2)])

            for i in range(h_):
                for j in range(w_):
                    c = np.array([i, j])
                    v2 = c + offset - r2.centroid()
                    desc = get_movement_descriptor_(v1, v2)
                    val = hist_query(H, edges, desc)
                    data[i, j] = val

            qim_ = QtGui.QImage(w_, h_, QtGui.QImage.Format_ARGB32)
            qim_.fill(QtGui.qRgba(0, 0, 0, 0))

            if data.max() == 0:
                print "MAX is 0"
                return

            data /= data.max()

            for i in range(h_):
                for j in range(w_):
                    val = int(round(data[i, j] * 255))
                    c = QtGui.qRgba(0, 255, 255, val)
                    qim_.setPixel(j, i, c)

            pixmap = QtGui.QPixmap.fromImage(qim_)

            item = QtGui.QGraphicsPixmapItem(pixmap)
            item.setPos(offset[1], offset[0])
            item.setZValue(0.99)

            self.video_player.visualise_temp(item)

            # start = QtCore.QPointF(r1.centroid()[0], r1.centroid()[1])
            # end = QtCore.QPointF(r2.centroid()[0], r2.centroid()[1])
            # item2 = QtGui.QGraphicsLineItem(QtCore.QLineF(start, end))
            item2 = QtGui.QGraphicsLineItem(r1.centroid()[1], r1.centroid()[0], r2.centroid()[1], r2.centroid()[0])
            pen = QtGui.QPen(QtGui.QColor(250, 255, 0, 255), 2, QtCore.Qt.SolidLine)
            item2.setPen(pen)
            item2.setZValue(0.99)
            self.video_player.visualise_temp(item2)

    def _gt_find_permutation(self, current_frame=False):
        if self.get_separated_frame_callback or current_frame:
            if current_frame:
                frame = self.video_player.current_frame()
            else:
                frame = self.get_separated_frame_callback()
                print "SEPARATED IN: ", frame

        permutation_data = []
        for t in self.project.chm.chunks_in_frame(frame):
            if not t.is_single():
                continue

            id_ = list(t.P)[0]
            y, x = RegionChunk(t, self.project.gm, self.project.rm).centroid_in_t(frame)
            permutation_data.append((frame, id_, y, x))

        self.idtracker_data_permutation = self._gt.set_permutation_reversed(permutation_data)  # ERROR: function is not returning anything
        self.video_player.redraw_visualisations()

    def show_results_summary(self):
        from gui.learning.learning_widget import draw_region, make_item
        from tqdm import trange
        from utils.video_manager import get_auto_video_manager

        start = int(self.show_results_summary_start_i.text())
        step = int(self.show_results_summary_steps_i.text())
        end = int(self.show_results_summary_end_i.text())

        vm = get_auto_video_manager(self.project)

        from gui.img_grid.img_grid_widget import ImgGridWidget
        w = ImgGridWidget(cols=len(self.project.animals), element_width=100)

        HH = 100
        WW = 100

        num_animals = len(self.project.animals)
        for frame in trange(start, end, step):
            region_representants = []
            img_representants = []

            for i in range(num_animals):
                region_representants.append(None)
                img_representants.append(None)

            for t in self.project.chm.chunks_in_frame(frame):
                if t.is_only_one_id_assigned(num_animals):
                    a_id = list(t.P)[0]

                    v_id = t.v_id_in_t(frame, self.project.gm)
                    if v_id == 0:
                        continue

                    im = draw_region(self.project, vm, v_id)

                    img_representants[a_id] = im
                    region_representants[a_id] = self.project.gm.region(v_id)

            for aid in range(len(self.project.animals)):
                if img_representants[aid] is None:
                    im = np.zeros((100, 100, 3), dtype=np.uint8)
                else:
                    im = img_representants[aid]

                item = make_item(im, region_representants[aid], HH, WW)
                w.add_item(item)

        self.show_results_view_w = QtGui.QWidget()
        self.show_results_view_w.setLayout(QtGui.QHBoxLayout())
        self.show_results_view_w.layout().addWidget(w)
        self.show_results_view_b = QtGui.QPushButton('show info about selected')
        self.show_results_view_b.clicked.connect(partial(self.results_view_show_info_selected, w.get_selected))
        self.show_results_view_w.layout().addWidget(self.show_results_view_b)
        self.show_results_view_w.show()


        # win = QtGui.QMainWindow()
        # win.setCentralWidget(w)
        # win.show()
        # self.w = win

    def results_view_show_info_selected(self, get_selected):
        # step = int(self.show_results_summary_steps_i.text())
        regions = get_selected()
        print "SELECTED IDS:"
        for r in regions:
            print str(r)
            # frame = ((id + 1) / len(self.p.animals)) * step
            # print "selected id: {} in frame: {}".format(id, frame)

    def cs_analysis(self):
        groups = []
        unique_tracklets = set()
        frames = []

        self.p = self.project

        from utils.video_manager import get_auto_video_manager
        vm = get_auto_video_manager(self.p)
        total_frame_count = vm.total_frame_count()
        expected_t_len = len(self.p.animals) * total_frame_count
        total_single_t_len = 0
        for t in tqdm(self.p.chm.chunk_gen(), total=len(self.p.chm)):
            if t.is_single():
                total_single_t_len += t.length()

        overlap_sum = 0
        frame = 0
        i = 0


        with tqdm(total=total_frame_count) as pbar:
            while True:
                group = self.p.chm.chunks_in_frame(frame)
                if len(group) == 0:
                    break

                singles_group = filter(lambda x: x.is_single(), group)
                df = frame

                # if len(singles_group) == len(self.p.animals) and min([len(t) for t in singles_group]) >= self.min_tracklet_len:
                if len(singles_group) == len(self.p.animals) and min([len(t) for t in singles_group]) >= 1:
                    groups.append(singles_group)

                    overlap = min([t.end_frame(self.p.gm) for t in singles_group]) \
                              - max([t.start_frame(self.p.gm) for t in singles_group])

                    overlap_sum += overlap

                    frames.append((frame, overlap,))
                    for t in singles_group:
                        unique_tracklets.add(t)

                    frame = min([t.end_frame(self.p.gm) for t in singles_group]) + 1
                else:
                    frame = min([t.end_frame(self.p.gm) for t in group]) + 1

                i += 1

                pbar.update(frame - df)

        print "analysis 1) DONE"
        print "# CS: ", len(groups)

        total_len = 0
        for t in unique_tracklets:
            total_len += len(t)

        single_len = 0
        for t in self.p.chm.chunk_gen():
            if t.is_single():
                single_len += len(t)


        print "Total tracklet length {}, coverage {:.2%}, single-ID coverage {:.2%}".format(
            total_len,
            (total_len/float(len(self.p.animals)))/total_frame_count,
            (single_len / float(len(self.p.animals))) / total_frame_count,
        )

        min_lengths = []
        val_ = 0
        for g in groups:
            min_lengths.append(-min([len(t) for t in g]))

        # print sorted(min_lengths)[:10]
        ids = np.argsort(min_lengths)

        for i in range(1, 11):
            id_ = ids[i]
            print "############### "
            print -min_lengths[id_]

            for t in groups[id_]:
                print t.id(), t.length()

        print
        print
        print "WORST ONES:"
        for i in range(10):
            id_ = ids[-i]
            print "############### "
            print -min_lengths[id_]

            for t in groups[id_]:
                print t.id(), t.length()

        num_single = 0

        pivot = groups[ids[0]]

        print pivot


        g1 = groups[0]
        for g2 in groups[1:]:
            if len(set(g1).intersection(g2)) == len(self.project.animals) - 1:
                num_single += 1

                print "single ID:"
                t1 = list(set(g1).difference(set(g1).intersection(g2)))[0]
                t2 = list(set(g2).difference(set(g1).intersection(g2)))[0]
                print t1.id(), t1.length()
                print t2.id(), t2.length()


                # was added to Track
                if t1.id() not in self.project.chm.chunks_:
                    for t in self.project.chm.chunks_in_frame(t1.end_frame(self.project.gm)):
                        if t.is_track():
                            if t.is_inside(t1):
                                t1 = t
                                break

                # was added to Track, shouldn'd happen
                if t2.id() not in self.project.chm.chunks_:
                    print "T2 happened....."
                    for t in self.project.chm.chunks_in_frame(t2.start_frame(self.project.gm)):
                        if t.is_track():
                            if t.is_inside(t1):
                                t2 = t
                                break

                from core.graph.track import Track
                tt = Track([t1, t2], self.project.gm)
                self.project.chm.new_track(tt, self.project.gm)
                print "T", tt.id()

            g1 = g2

        print "# single", num_single

    def show_color_settings(self):
        from set_colors_gui import SetColorsGui

        w = SetColorsGui(self.project)
        w.show()