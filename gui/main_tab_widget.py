import sys

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
        self.tabs = DockTabWidget()

        self.results_tab = QtGui.QWidget()
        self.statistics_tab = StatisticsWidget(project)
        self.graph_tab = GraphWidgetLoader(self.project).get_widget(show_tracklet_callback=self.play_and_highlight_tracklet)

        self.id_detection_tab = QtGui.QWidget()

        self.finish_callback = finish_callback

        self.tab_widgets = [self.tracker_tab, self.results_tab, self.id_detection_tab, self.statistics_tab, self.graph_tab]
        self.tab_names = ["tracking", "results viewer", "id detection", "stats && results", "graph"]
        for i in range(len(self.tab_widgets)):
            self.tabs.addTab(self.tab_widgets[i], self.tab_names[i])
            self.tabs.setEnabled(i)

        self.switch_to_tracking_window_action = QtGui.QAction('switch tab to tracking', self)
        self.switch_to_tracking_window_action.triggered.connect(partial(self.tabs.setCurrentIndex, 0))
        self.switch_to_tracking_window_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_T))
        self.addAction(self.switch_to_tracking_window_action)

        self.vbox.addWidget(self.tabs)

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

        for i in range(len(self.tab_widgets)):
            self.tabs.setEnabled(i)

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
            self.graph_tab.redraw()
            self.detach_tab(4)

        # if i == 0:
        #     # TODO: add interval to settings
        #     self.tracker_tab.autosave_timer.start(1000*60*5)
        # else:
        #     self.tracker_tab.autosave_timer.stop()

        pass

    def detach_tab(self, tab_number):
        widget = self.tabs.widget(tab_number)
        self.tabs.removeTab(tab_number)
        window = DetachedWindow(self, widget, self, tab_number)
        window.show()

    def attach_tab(self, number):
        self.tabs.insertTab(number, self.tab_widgets[number], self.tab_names[number])


class DetachedWindow(QtGui.QMainWindow):

    def __init__(self, parent, widget, tab_widget_callback, number):
        super(DetachedWindow, self).__init__(parent)
        self.widget = widget
        self.tab_widget_callback = tab_widget_callback
        self.number = number
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle(widget.windowTitle())
        self.setCentralWidget(widget)
        self.widget.show()

    def closeEvent(self, event):
        self.tab_widget_callback.attach_tab(self.number)
        super(DetachedWindow, self).closeEvent(event)


class DockTabWidget(QtGui.QTabWidget):

    def __init__(self, parent=None):
        super(DockTabWidget, self).__init__(parent)
        self.dock_button = QtGui.QPushButton("dock")
        self.undock_button = QtGui.QPushButton("undock")
        self.dock_button.setFixedHeight(30)
        self.undock_button.setFixedHeight(30)
        self.button_group = QtGui.QButtonGroup()
        self.buttons = QtGui.QWidget()
        self.buttons.setLayout(QtGui.QHBoxLayout())
        self.buttons.layout().addWidget(self.undock_button)
        self.buttons.layout().addWidget(self.dock_button)
        self.setCornerWidget(self.buttons, QtCore.Qt.BottomRightCorner)
