__author__ = 'fnaiser'

import pickle
from PyQt4 import QtGui

from gui.init.init_where_widget import InitWhereWidget
from gui.init.init_what_widget import InitWhatWidget
from gui.init.init_how_widget import InitHowWidget


SKIP_WHERE = False
SKIP_WHAT = False

class InitWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project):
        super(InitWidget, self).__init__()
        self.finish_callback = finish_callback
        self.project = project

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.tabs = QtGui.QTabWidget()
        self.where_tab = InitWhereWidget(self.widget_control, project)
        self.what_tab = InitWhatWidget(self.widget_control, project)
        self.how_tab = InitHowWidget(self.widget_control, project)

        self.finish_callback = finish_callback

        self.tabs.addTab(self.where_tab, "Where")
        self.tabs.addTab(self.what_tab, "What")
        self.tabs.addTab(self.how_tab, "How")
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)

        self.vbox.addWidget(self.tabs)

        if SKIP_WHAT:
            self.tabs.setTabEnabled(2, True)
            self.tabs.setCurrentIndex(2)
            self.tabs.setTabEnabled(1, False)
        elif SKIP_WHERE:
            self.widget_control('init_where_finished')


    def widget_control(self, state, values=None):
        if state == 'init_where_finished':
            with open(self.project.working_directory+'/bg_model.pkl', 'wb') as f:
                pickle.dump(self.project.bg_model, f)
                print "SAVING BG_MODEL"

            with open(self.project.working_directory+'/arena_model.pkl', 'wb') as f:
                pickle.dump(self.project.arena_model, f)


            self.tabs.setTabEnabled(1, True)
            self.tabs.setCurrentIndex(1)
            self.tabs.setTabEnabled(0, False)

        if state == 'init_what_finished':
            with open(self.project.working_directory+'/classes.pkl', 'wb') as f:
                pickle.dump(self.project.classes, f)

            with open(self.project.working_directory+'/groups.pkl', 'wb') as f:
                pickle.dump(self.project.groups, f)

            with open(self.project.working_directory+'/animals.pkl', 'wb') as f:
                pickle.dump(self.project.animals, f)

            self.tabs.removeTab(2)
            self.how_tab = InitHowWidget(self.widget_control, self.project)
            self.tabs.addTab(self.how_tab, "How")

            self.tabs.setTabEnabled(2, True)
            self.tabs.setCurrentIndex(2)
            self.tabs.setTabEnabled(1, False)

        if state == 'init_how_finished':
            with open(self.project.working_directory+'/stats.pkl', 'wb') as f:
                pickle.dump(values[0], f)
                print "SAVING STATS"

            self.finish_callback('initialization_finished')
