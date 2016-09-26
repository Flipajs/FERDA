from gui.graph_widget_loader import GraphWidgetLoader

__author__ = 'fnaiser'

import os

from PyQt4 import QtGui, QtCore
from gui.tracker.tracker_widget import TrackerWidget
from gui.correction.correction_widget import ResultsWidget
from gui.statistics.statistics_widget import StatisticsWidget

from core.background_computer import BackgroundComputer
from functools import partial
from core.graph.graph_manager import GraphManager
import time


class MainTabWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project, postpone_parallelisation=False):
        super(MainTabWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        self.project = project

        self.solver = None

        self.tracker_tab = TrackerWidget(project, show_in_visualizer_callback=self.show_in_visualizer)
        self.tabs = QtGui.QTabWidget()

        self.results_tab = QtGui.QWidget()
        self.statistics_tab = StatisticsWidget(project)
        self.graph_tab = GraphWidgetLoader(self.project).get_widget(show_tracklet_callback=self.play_and_highlight_tracklet)

        self.id_detection_tab = QtGui.QWidget()

        self.finish_callback = finish_callback

        self.tabs.addTab(self.tracker_tab, "tracking")
        self.tabs.addTab(self.results_tab, "results viewer")
        self.tabs.addTab(self.id_detection_tab, "id detection")
        self.tabs.addTab(self.statistics_tab, "stats && results")
        self.tabs.addTab(self.graph_tab, "graph")

        self.switch_to_tracking_window_action = QtGui.QAction('switch tab to tracking', self)
        self.switch_to_tracking_window_action.triggered.connect(partial(self.tabs.setCurrentIndex, 0))
        self.switch_to_tracking_window_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_T))
        self.addAction(self.switch_to_tracking_window_action)

        self.vbox.addWidget(self.tabs)

        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)

        self.ignore_tab_change = False

        self.tabs.currentChanged.connect(self.tab_changed)
        self.tabs.setCurrentIndex(1)

        self.tabs.setTabEnabled(0, False)

        self.show_results_only_around_frame = -1

        self.reload_id_data = QtGui.QAction('reload', self)
        self.reload_id_data.triggered.connect(self.reload_ids)
        self.reload_id_data.setShortcut(QtGui.QKeySequence(QtCore.Qt.ShiftModifier + QtCore.Qt.Key_R))
        self.addAction(self.reload_id_data)


        print "LOADING GRAPH..."
        if project.gm is None or project.gm.g.num_vertices() == 0:
            # project.gm = GraphManager(project, project.solver.assignment_score)
            self.bc_msers = BackgroundComputer(project, self.tracker_tab.bc_update, self.background_computer_finished, postpone_parallelisation)
            self.bc_msers.run()
        else:
            self.background_computer_finished(project.solver)

    def reload_ids(self):
        print "RELOADING"
        import cPickle as pickle
        try:
            with open(self.project.working_directory+'/temp/chunk_available_ids.pkl', 'rb') as f_:
                chunk_available_ids = pickle.load(f_)

            for ch_id in self.project.gm.chunk_list():
                animal_id = -1
                if ch_id in chunk_available_ids:
                    animal_id = chunk_available_ids[ch_id]

                self.project.chm[ch_id].animal_id_ = animal_id

        except IOError:
            pass

        try:
            self.results_tab.update_positions()
        except AttributeError:
            pass


    def show_in_visualizer(self, data):
        self.show_results_only_around_frame = data['n1'].frame_
        self.tabs.setCurrentIndex(1)
        self.show_results_only_around_frame = -1
        self.results_tab.change_frame(data['n1'].frame_)
        self.results_tab.highlight_area(data, radius=100)

    def background_computer_finished(self, solver):
        print "GRAPH LOADED"
        self.solver = solver
        self.results_tab.solver = solver
        self.tracker_tab.prepare_corrections(self.project.solver)

        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)
        self.tabs.setTabEnabled(3, True)

    def play_and_highlight_tracklet(self, tracklet, frame=-1, margin=0):
        self.tabs.setCurrentIndex(1)
        self.results_tab.play_and_highlight_tracklet(tracklet, frame=frame, margin=margin)

    def decide_tracklet(self, tracklet):
        self.tab_changed(2)
        # if not self.id_detection_tab:
        #     self.tab_changed(2)

        self.id_detection_tab.decide_tracklet_question(tracklet)

    def tab_changed(self, i):
        if self.ignore_tab_change:
            return

        if i == 1:
            if not isinstance(self.results_tab, ResultsWidget):
                self.ignore_tab_change = True
                self.tabs.removeTab(1)
                self.results_tab.setParent(None)

                self.results_tab = ResultsWidget(self.project, decide_tracklet_callback=self.decide_tracklet)
                self.results_tab.update_positions()
                self.tabs.insertTab(1, self.results_tab, 'results viewer')
                self.tabs.setCurrentIndex(1)
                self.ignore_tab_change = False

            self.results_tab.update_positions()

        if i == 2:
            # TODO: show loading or something...
            from gui.learning.learning_widget import LearningWidget
            if not isinstance(self.id_detection_tab, LearningWidget):
                self.ignore_tab_change = True
                self.id_detection_tab = LearningWidget(self.project, self.play_and_highlight_tracklet)
                self.tabs.removeTab(2)
                self.tabs.insertTab(2, self.id_detection_tab, "id detection")
                self.tabs.setCurrentIndex(2)
                self.ignore_tab_change = False

        if i == 3:
            self.statistics_tab.update_data(self.project)
        if i == 4:
            # if not isinstance(self.graph_tab, GraphWidgetLoader):
                # self.ignore_tab_change = True
                # TODO: show loading...
                # self.graph_tab = GraphWidgetLoader(self.project).get_widget()
                # self.tabs.removeTab(4)
                # self.tabs.insertTab(4, self.graph_tab, "graph")
                # self.tabs.setCurrentIndex(4)
                # self.ignore_tab_change = False
            # else:
            self.graph_tab.redraw()

        # if i == 0:
        #     # TODO: add interval to settings
        #     self.tracker_tab.autosave_timer.start(1000*60*5)
        # else:
        #     self.tracker_tab.autosave_timer.stop()

        pass