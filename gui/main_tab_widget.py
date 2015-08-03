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
        self.statistics_tab = StatisticsWidget(project)

        self.finish_callback = finish_callback

        self.tabs.addTab(self.tracker_tab, "tracking")
        self.tabs.addTab(self.results_tab, "results viewer")
        self.tabs.addTab(self.statistics_tab, "stats && results")
        self.vbox.addWidget(self.tabs)

        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)

        self.tabs.currentChanged.connect(self.tab_changed)

        print "LOADING GRAPH..."
        if project.saved_progress:
            solver = project.saved_progress['solver']
            solver.update_nodes_in_t_refs()
            self.background_computer_finished(solver)
        else:
            self.bc_msers = BackgroundComputer(project, self.tracker_tab.bc_update, self.background_computer_finished)
            self.bc_msers.run()

    def background_computer_finished(self, solver):
        print "GRAPH LOADED"
        self.solver = solver
        self.results_tab.solver = solver
        self.results_tab.add_data(self.solver)
        self.tracker_tab.prepare_corrections(self.solver)

        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)

    def tab_changed(self, i):
        if i == 1:
            self.results_tab.add_data(self.solver)
            self.results_tab.update_positions(self.results_tab.video.frame_number())
        if i == 2:
            self.statistics_tab.update_data(self.solver)


        # if i == 0:
        #     # TODO: add interval to settings
        #     self.tracker_tab.autosave_timer.start(1000*60*5)
        # else:
        #     self.tracker_tab.autosave_timer.stop()

        pass