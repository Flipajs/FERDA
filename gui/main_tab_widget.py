__author__ = 'fnaiser'

from PyQt4 import QtGui
from gui.tracker.tracker_widget import TrackerWidget
from gui.correction.correction_widget import ResultsWidget
from gui.statistics.statistics_widget import StatisticsWidget

from core.background_computer import BackgroundComputer


class MainTabWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project):
        super(MainTabWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        self.project = project

        self.solver = None

        self.tabs = QtGui.QTabWidget()
        self.tracker_tab = TrackerWidget(project)
        self.results_tab = ResultsWidget(project)
        self.statistics_tab = StatisticsWidget()

        self.finish_callback = finish_callback

        self.tabs.addTab(self.tracker_tab, "Tracking")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.statistics_tab, "Statistics")
        self.vbox.addWidget(self.tabs)

        print "LOADING GRAPH..."
        if project.saved_progress:
            self.background_computer_finished(project.saved_progress['solver'])
        else:
            self.bc_msers = BackgroundComputer(project, self.tracker_tab.bc_update, self.background_computer_finished)
            self.bc_msers.run()

    def background_computer_finished(self, solver):
        print "GRAPH LOADED"
        self.solver = solver
        self.results_tab.solver = solver
        self.results_tab.add_data(self.solver)
        self.tracker_tab.prepare_corrections(self.solver)
