import sys

from gui.graph_widget_loader import GraphWidgetLoader

__author__ = 'fnaiser'

import os

from PyQt4 import QtGui, QtCore
from gui.tracker.tracker_widget import TrackerWidget
from gui.results.results_widget import ResultsWidget
from gui.statistics.statistics_widget import StatisticsWidget

from core.background_computer import BackgroundComputer
from functools import partial
from gui.graph_widget.graph_visualizer import GraphVisualizer
from core.graph.graph_manager import GraphManager
import time
from gui.learning.learning_widget import LearningWidget


class MainTabWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project, postpone_parallelisation=False):
        super(MainTabWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        self.project = project

        self.solver = None

        self.tracker_tab = TrackerWidget(project, show_in_visualizer_callback=self.show_in_visualizer)
        self.tabs = QtGui.QTabWidget(self)

        self.undock_button = QtGui.QPushButton("Undock")
        self.undock_button.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.undock_button.pressed.connect(self.detach_tab)
        self.buttons = QtGui.QWidget()
        self.buttons.setLayout(QtGui.QHBoxLayout())
        spacer = QtGui.QWidget()
        spacer.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.buttons.layout().addWidget(spacer)
        self.buttons.layout().addWidget(self.undock_button)
        self.undock_button.setFixedHeight(30)

        self.results_tab = QtGui.QWidget()
        self.statistics_tab = StatisticsWidget(project)
        self.graph_tab = QtGui.QWidget()

        self.id_detection_tab = LearningWidget(self.project, self.play_and_highlight_tracklet)

        self.finish_callback = finish_callback

        self.tab_widgets = [self.tracker_tab, self.results_tab, self.id_detection_tab, self.statistics_tab, self.graph_tab]
        self.tab_names = ["tracking", "results viewer", "id detection", "stats && results", "graph"]
        self.tab_docked = [False] * len(self.tab_widgets)
        for i in range(len(self.tab_widgets)):
            self.tabs.addTab(self.tab_widgets[i], self.tab_names[i])
            self.tabs.setEnabled(i)

        self.switch_to_tracking_window_action = QtGui.QAction('switch tab to tracking', self)
        self.switch_to_tracking_window_action.triggered.connect(partial(self.tabs.setCurrentIndex, 0))
        self.switch_to_tracking_window_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_T))
        self.addAction(self.switch_to_tracking_window_action)

        self.vbox.addWidget(self.tabs)
        self.layout().addWidget(self.buttons)

        self.ignore_tab_change = False

        self.tabs.currentChanged.connect(self.tab_changed)
        self.tabs.setCurrentIndex(1)

        self.tabs.setTabEnabled(0, False)

        self.show_results_only_around_frame = -1

        self.reload_id_data = QtGui.QAction('reload', self)
        self.reload_id_data.triggered.connect(self.reload_ids)
        self.reload_id_data.setShortcut(QtGui.QKeySequence(QtCore.Qt.ShiftModifier + QtCore.Qt.Key_R))
        self.addAction(self.reload_id_data)

        self.update_undecided_a = QtGui.QAction('update undecided', self)
        self.update_undecided_a.triggered.connect(self.learning_widget_update_undecided)
        self.update_undecided_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_U))
        self.addAction(self.update_undecided_a)


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

            for ch in self.project.chm.chunk_gen:
                ch_id = ch.id()
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

        for i in range(len(self.tab_widgets)):
            self.tabs.setEnabled(i)

    def play_and_highlight_tracklet(self, tracklet, frame=-1, margin=0):
        self.tabs.setCurrentIndex(1)
        self.results_tab.play_and_highlight_tracklet(tracklet, frame=frame, margin=margin)

    def decide_tracklet(self, tracklet, id_=None):
        # self.tab_changed(2)
        if not self.id_detection_tab:
            self.tab_changed(2)
        self.id_detection_tab.decide_tracklet_question(tracklet, id_=id_)

    def tab_changed(self, i):
        if self.ignore_tab_change or self.project.chm is None:
            return

        if i == 1:
            if len(self.project.chm):
                if not isinstance(self.results_tab, ResultsWidget):
                    self.ignore_tab_change = True
                    self.tabs.removeTab(1)
                    self.results_tab.setParent(None)
                    self.results_tab = ResultsWidget(self.project,
                                                     callbacks={'decide_tracklet': self.decide_tracklet,
                                                                'edit_tracklet': self.id_detection_tab.edit_tracklet})
                    # self.results_tab.redraw_video_player_visualisations()
                    self.tabs.insertTab(1, self.results_tab, 'results viewer')
                    self.tabs.setCurrentIndex(1)
                    self.ignore_tab_change = False

                self.results_tab.update_visualisations()

        if i == 2:
            # TODO: show loading or something...
            if not isinstance(self.id_detection_tab, LearningWidget):
                self.ignore_tab_change = True
                self.tabs.removeTab(2)
                self.id_detection_tab.setParent(None)
                self.id_detection_tab = LearningWidget(self.project, self.play_and_highlight_tracklet)
                self.id_detection_tab.update_callback()
                self.tabs.insertTab(2, self.id_detection_tab, "id detection")
                self.tabs.setCurrentIndex(2)
                self.ignore_tab_change = False

        if i == 3:
            self.statistics_tab.update_data(self.project)
        if i == 4:
            from utils.video_manager import get_auto_video_manager
            vm = get_auto_video_manager(self.project)
            max_f = vm.total_frame_count()

            from_frame, ok = QtGui.QInputDialog.getInt(self, "show range", "From: ", 0, 0, max_f-1)
            if ok or not isinstance(self.graph_tab, GraphVisualizer):
                frames = None

                to_frame, ok = QtGui.QInputDialog.getInt(self, "show range", "From: ", from_frame+1, from_frame+1, max_f)
                if ok:
                    frames = range(from_frame, to_frame)

                self.ignore_tab_change = True
                # TODO: show loading...
                self.tabs.removeTab(4)
                self.graph_tab.setParent(None)
                self.graph_tab = GraphWidgetLoader(self.project, width=50, height=50).get_widget(show_tracklet_callback=self.play_and_highlight_tracklet, frames=frames)
                self.tabs.insertTab(4, self.graph_tab, "graph")
                self.tabs.setCurrentIndex(4)
                self.ignore_tab_change = False

                self.graph_tab.redraw()

        pass

    def detach_tab(self):
        tab_number = self.tabs.currentIndex()
        widget = self.tabs.widget(tab_number)
        self.tabs.removeTab(tab_number)
        window = DetachedWindow(self, widget, self, tab_number)
        window.show()

    def attach_tab(self, number):
        self.tabs.insertTab(number, self.tab_widgets[number], self.tab_names[number])

    def learning_widget_update_undecided(self):
        if isinstance(self.id_detection_tab, LearningWidget):
            self.id_detection_tab.update_undecided_tracklets()

class DetachedWindow(QtGui.QMainWindow):

    def __init__(self, parent, widget, widget_callback, number):
        super(DetachedWindow, self).__init__(parent)
        content = QtGui.QWidget()
        content.setLayout(QtGui.QVBoxLayout())
        self.dock_widget = QtGui.QWidget()
        dock_button = QtGui.QPushButton("Dock")
        dock_button.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        dock_button.pressed.connect(self.close)
        self.dock_widget.setLayout(QtGui.QHBoxLayout())
        spacer = QtGui.QWidget()
        spacer.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.dock_widget.layout().addWidget(spacer)
        self.dock_widget.layout().addWidget(dock_button)
        self.widget_callback = widget_callback
        self.number = number
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle(self.widget_callback.tab_names[number])
        content.layout().addWidget(widget)
        content.layout().addWidget(self.dock_widget)
        self.setCentralWidget(content)
        widget.show()

    def closeEvent(self, event):
        super(DetachedWindow, self).closeEvent(event)
        self.attach()

    def attach(self):
        self.dock_widget.hide()
        self.widget_callback.attach_tab(self.number)
        temp = self.widget_callback.ignore_tab_change
        self.widget_callback.ignore_tab_change = False
        self.widget_callback.tabs.setCurrentIndex(self.number)
        self.widget_callback.ignore_tab_change = temp


