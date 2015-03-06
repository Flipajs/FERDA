__author__ = 'fnaiser'

from PyQt4 import QtGui
from gui.init.init_where_widget import InitWhereWidget
from gui.init.init_what_widget import InitWhatWidget
from gui.init.init_how_widget import InitHowWidget

SKIP_WHERE = True

class InitWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project, bg_model):
        super(InitWidget, self).__init__()
        self.finish_callback = finish_callback
        self.project = project
        self.bg_model = bg_model

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.tabs = QtGui.QTabWidget()
        self.where_tab = InitWhereWidget(self.widget_control, project, bg_model)
        self.what_tab = InitWhatWidget(self.widget_control, project, bg_model)
        self.how_tab = InitHowWidget(self.widget_control)

        self.finish_callback = finish_callback

        self.tabs.addTab(self.where_tab, "Where")
        self.tabs.addTab(self.what_tab, "What")
        self.tabs.addTab(self.how_tab, "How")
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)

        self.vbox.addWidget(self.tabs)

        if SKIP_WHERE:
            self.widget_control('init_where_finished')

    def widget_control(self, state, values=None):
        if state == 'init_where_finished':
            self.tabs.setTabEnabled(1, True)
            self.tabs.setCurrentIndex(1)
            self.tabs.setTabEnabled(0, False)
        return