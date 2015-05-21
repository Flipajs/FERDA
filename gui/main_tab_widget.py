__author__ = 'fnaiser'

from PyQt4 import QtGui
from gui.tracker.tracker_widget import TrackerWidget
from gui.correction.correction_widget import CorrectionWidget
from gui.statistics.statistics_widget import StatisticsWidget

class MainTabWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project):
        super(MainTabWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.tabs = QtGui.QTabWidget()
        self.tracker_tab = TrackerWidget(project)
        self.correction_tab = CorrectionWidget()
        self.statistics_tab = StatisticsWidget()

        self.finish_callback = finish_callback

        self.tabs.addTab(self.tracker_tab, "Tracking")
        self.tabs.addTab(self.correction_tab, "Corrections")
        self.tabs.addTab(self.statistics_tab, "Statistics")
        self.vbox.addWidget(self.tabs)
