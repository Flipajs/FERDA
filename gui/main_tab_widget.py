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


class MainTabWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project, postpone_parallelisation=False):
        super(MainTabWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        self.project = project

        self.solver = None

        self.tabs = QtGui.QTabWidget()
        self.tracker_tab = TrackerWidget(project, show_in_visualizer_callback=self.show_in_visualizer)
        self.results_tab = ResultsWidget(project)
        self.statistics_tab = StatisticsWidget(project)
        self.graph_tab = GraphWidgetLoader(project).get_widget()

        self.finish_callback = finish_callback

        self.tabs.addTab(self.tracker_tab, "tracking")
        self.tabs.addTab(self.results_tab, "results viewer")
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

        self.show_results_only_around_frame = -1

        print "LOADING GRAPH..."
        if project.gm is None or project.gm.g.num_vertices() == 0:
            # project.gm = GraphManager(project, project.solver.assignment_score)
            self.bc_msers = BackgroundComputer(project, self.tracker_tab.bc_update, self.background_computer_finished, postpone_parallelisation)
            self.bc_msers.run()
        else:
            self.background_computer_finished(project.solver)

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

    def tab_changed(self, i):
        if self.ignore_tab_change:
            return

        if i == 1:
            self.ignore_tab_change = True
            self.tabs.removeTab(1)
            self.results_tab.setParent(None)

            self.results_tab = ResultsWidget(self.project)
            self.results_tab.add_data(self.project.solver, self.show_results_only_around_frame)
            self.results_tab.update_positions(self.results_tab.video.frame_number())
            # TODO
            # self.results_tab.update_positions(self.results_tab.video.frame_number(), optimized=False)
            self.tabs.insertTab(1, self.results_tab, 'results viewer')
            self.tabs.setCurrentIndex(1)
            self.ignore_tab_change = False
        if i == 2:
            self.statistics_tab.update_data(self.project)
        if i == 3:
            # TODO
            self.graph_tab.redraw()

        # if i == 0:
        #     # TODO: add interval to settings
        #     self.tracker_tab.autosave_timer.start(1000*60*5)
        # else:
        #     self.tracker_tab.autosave_timer.stop()

        pass