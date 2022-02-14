import sys
from collections import OrderedDict
from functools import partial
import os.path

from PyQt5 import QtCore, QtGui, QtWidgets
from gui.tracking_widget import TrackingWidget
from gui.results.results_widget import ResultsWidget
from gui.statistics.statistics_widget import StatisticsWidget
from gui.graph_widget_loader import GraphWidgetLoader
from gui.learning.learning_widget import LearningWidget
from core.project.project import Project, ProjectNotFoundError
from utils.video_manager import VideoFileError
from gui.project.new_project_wizard import NewProjectWizard
from gui.settings_widgets.settings_dialog import SettingsDialog
from gui.settings import Settings as S_

from gui.generated.ui_landing_tab import Ui_landingForm


class LandingTab(QtWidgets.QWidget):
    project_ready = QtCore.pyqtSignal(object)

    def __init__(self):
        super(LandingTab, self).__init__()
        self.ui = Ui_landingForm()
        self.ui.setupUi(self)

        self.ui.newProjectButton.clicked.connect(self.show_new_project_wizard)
        self.ui.loadProjectButton.clicked.connect(lambda state: self.load_project())
        self.ui.settingsButton.clicked.connect(self.show_settings_dialog)

    def show_new_project_wizard(self):
        wizard = NewProjectWizard(self)
        if wizard.exec_() == QtWidgets.QDialog.Accepted:
            self.project_ready.emit(wizard.project)

    def show_settings_dialog(self):
        dialog = SettingsDialog(self)
        dialog.exec_()

    def load_project(self, project_dir=None):
        if project_dir is None:
            # ask for a project folder
            if os.path.isdir(S_.temp.last_wd_path):
                path = S_.temp.last_wd_path
            else:
                path = ''

            project_dir = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a project folder', directory=path))
            if not project_dir:
                return

        # project = Project()
        video_file = None
        while True:
            try:
                project = Project()
                project.load(project_dir, video_file=video_file,
                             regions_optional=True, graph_optional=True, tracklets_optional=True)
                break
            except ProjectNotFoundError as e:
                QtWidgets.QMessageBox.critical(self, 'No project found!', str(e), QtWidgets.QMessageBox.Ok)
                return
            except VideoFileError:
                video_file = str(QtWidgets.QFileDialog.getOpenFileName(
                    self, 'Project video file not found or can\'t be opened. Select new video location.',
                    '.', filter='Videos (*.mp4 *.avi *.mkv *.webm *.mpg *.mpeg *.mov);;All Files (*.*)'))[0]
                if not video_file:
                    return

        S_.temp.last_wd_path = project_dir
        self.project_ready.emit(project)


class MainTabWidget(QtWidgets.QWidget):
    def __init__(self):
        super(MainTabWidget, self).__init__()

        self.project = None
        self.ignore_tab_change = False
        self.show_results_only_around_frame = -1
        self.solver = None

        self.setWindowIcon(QtGui.QIcon('imgs/ferda.ico'))
        self.setWindowTitle('FERDA')
        # self.setGeometry(100, 100, 700, 400)

        self.vbox = QtWidgets.QVBoxLayout()
        self.setLayout(self.vbox)
        self.tabs = QtWidgets.QTabWidget(self)

        # # TODO: takes too much space
        # self.undock_button = QtGui.QPushButton("Undock")
        # self.undock_button.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        # self.undock_button.pressed.connect(self.detach_tab)
        # self.buttons = QtGui.QWidget()
        # self.buttons.setLayout(QtGui.QHBoxLayout())
        # spacer = QtGui.QWidget()
        # spacer.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        # self.buttons.layout().addWidget(spacer)
        # self.buttons.layout().addWidget(self.undock_button)
        # self.undock_button.setFixedHeight(30)

        self.widgets = OrderedDict([
            ('main', LandingTab()),
            ('tracking', None),
            ('results', None),
            # ('id_detection', QtWidgets.QWidget()),
            # ('stats', QtWidgets.QWidget()),
            ('graph', None),
        ])

        self.widgets_info = dict(
            [
                ('main', 'Main'),
                ('tracking', 'Tracking'),
                ('results', 'Results'),
                # ('id_detection', 'Id Detection'),
                # ('stats', 'Statistics'),
                ('graph', 'Graph'),
            ])
        self.reload_tabs()

        self.vbox.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.widgets['main'].project_ready.connect(self.update_project)

        self.switch_to_tracking_window_action = QtWidgets.QAction('switch tab to tracking', self)
        self.switch_to_tracking_window_action.triggered.connect(partial(self.tabs.setCurrentIndex, 0))
        self.switch_to_tracking_window_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_T))
        self.addAction(self.switch_to_tracking_window_action)

        self.reload_id_data = QtWidgets.QAction('reload', self)
        self.reload_id_data.triggered.connect(self.reload_ids)
        self.reload_id_data.setShortcut(QtGui.QKeySequence(QtCore.Qt.ShiftModifier + QtCore.Qt.Key_R))
        self.addAction(self.reload_id_data)

        self.update_undecided_a = QtWidgets.QAction('update undecided', self)
        self.update_undecided_a.triggered.connect(self.learning_widget_update_undecided)
        self.update_undecided_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_U))
        self.addAction(self.update_undecided_a)

    def update_project(self, project):
        self.project = project
        self.widgets['tracking'] = TrackingWidget(self.project)
        self.widgets['results'] = ResultsWidget(self.project,
                                         callbacks={'decide_tracklet': self.decide_tracklet,
                                                    'edit_tracklet': self.edit_tracklet,
                                                    'get_separated_frame': self.get_separated_frame,
                                                    'update_N_sets': self.update_N_sets,
                                                    'tracklet_measurements': self.tracklet_measurements})
        # self.widgets['id_detection'] = LearningWidget(self.project, self.play_and_highlight_tracklet)
        # self.widgets['id_detection'].setEnabled(True)
        # self.widgets['stats'] = StatisticsWidget(project)
        # self.widgets['stats'].setEnabled(True)
        if self.project.chm is not None and len(self.project.chm) > 0:
            self.widgets['graph'] = GraphWidgetLoader(project, tracklet_callback=self.play_and_highlight_tracklet)
        self.reload_tabs()
        self.tabs.setCurrentWidget(self.widgets['results'])

    def reload_tabs(self):
        self.tabs.clear()
        for i, (name, widget) in enumerate(self.widgets.items()):
            if widget is not None:
                self.tabs.addTab(widget, self.widgets_info[name])
                self.tabs.setTabEnabled(i, True)
            else:
                self.tabs.addTab(QtWidgets.QWidget(), self.widgets_info[name])
                self.tabs.setTabEnabled(i, False)

    def reload_ids(self):
        print("RELOADING")
        import pickle as pickle
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
        print("GRAPH LOADED")
        self.solver = solver
        self.results_tab.solver = solver

        for i in range(len(self.tab_widgets)):
            self.tabs.setEnabled(i)

    def play_and_highlight_tracklet(self, tracklet, frame=-1, margin=0):
        self.tabs.setCurrentIndex(1)
        self.results_tab.play_and_highlight_tracklet(tracklet, frame=frame, margin=margin)

    def decide_tracklet(self, tracklet, id_=None):
        # self.tab_changed(2)
        if not isinstance(self.id_detection_tab, LearningWidget):
            self.tab_changed(2)
        self.id_detection_tab.decide_tracklet_question(tracklet, id_=id_)

    def edit_tracklet(self, tracklet):
        if not isinstance(self.id_detection_tab, LearningWidget):
            self.tab_changed(2)

        self.id_detection_tab.edit_tracklet(tracklet)

    def get_separated_frame(self):
        if not isinstance(self.id_detection_tab, LearningWidget):
            self.tab_changed(2)

        return self.id_detection_tab.get_separated_frame()

    def update_N_sets(self):
        if not isinstance(self.id_detection_tab, LearningWidget):
            self.tab_changed(2)

        return self.id_detection_tab.update_N_sets()

    def tracklet_measurements(self, id_):
        if not isinstance(self.id_detection_tab, LearningWidget):
            self.tab_changed(2)

        return self.id_detection_tab.tracklet_measurements(id_)

    def tab_changed(self, i):
        if self.ignore_tab_change: #  or self.project.chm is None:
            return

        if i == 1:
            self.widgets['results'].update_visualisations()

        if i == 2:
            pass
            # TODO: show loading or something...
            # if not isinstance(self.id_detection_tab, LearningWidget):
            #     ok = False
            #     for ch in self.project.chm.tracklets_in_frame(0):
            #         if not ch.is_undefined():
            #             ok = True
            #             break
            #
            #     if not ok:
            #         QtGui.QMessageBox.information(None,
            #             "there is 0 tracklets with proper class (single-ID, multi-ID, no-ID, part-of-ID) in frame 0, most likely you need to continue to region classifier tab and do tracklet classification first. Continue with id detection only if you are aware of what you are doing.")
            #
            #     self.ignore_tab_change = True
            #     self.tabs.removeTab(2)
            #     self.id_detection_tab.setParent(None)
            #     self.id_detection_tab = LearningWidget(self.project, self.play_and_highlight_tracklet, self.progress_callback)
            #     self.tabs.insertTab(2, self.id_detection_tab, "id detection")
            #     self.tabs.setCurrentIndex(2)
            #     self.ignore_tab_change = False

            # if not len(self.id_detection_tab.lp.features):
            #     pass
                # self.id_detection_tab.disable_before_features()

        if i == 3:
            self.widgets['stats'].update_data(self.project)
        if i == 4:
            pass

    # def detach_tab(self):
    #     tab_number = self.tabs.currentIndex()
    #     widget = self.tabs.widget(tab_number)
    #     self.tabs.removeTab(tab_number)
    #     window = DetachedWindow(self, widget, self, tab_number)
    #     window.show()
    #
    # def attach_tab(self, number):
    #     self.tabs.insertTab(number, self.tab_widgets[number], self.tab_names[number])

    def learning_widget_update_undecided(self):
        if isinstance(self.widgets['id_detection'], LearningWidget):
            self.widgets['id_detection'].update_undecided_tracklets()


class DetachedWindow(QtWidgets.QMainWindow):
    def __init__(self, parent, widget, widget_callback, number):
        super(DetachedWindow, self).__init__(parent)
        content = QtWidgets.QWidget()
        content.setLayout(QtWidgets.QVBoxLayout())
        self.dock_widget = QtWidgets.QWidget()
        dock_button = QtWidgets.QPushButton("Dock")
        dock_button.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        dock_button.pressed.connect(self.close)
        self.dock_widget.setLayout(QtWidgets.QHBoxLayout())
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
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


