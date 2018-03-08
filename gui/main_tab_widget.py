from gui.graph_widget_loader import GraphWidgetLoader
from PyQt4 import QtGui, QtCore
from gui.tracker.tracker_widget import TrackerWidget
from gui.results.results_widget import ResultsWidget
from gui.statistics.statistics_widget import StatisticsWidget
from core.background_computer import BackgroundComputer
from functools import partial
from gui.learning.learning_widget import LearningWidget

__author__ = 'fnaiser'


class MainTabWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project, postpone_parallelisation=False, progress_callback=None):
        super(MainTabWidget, self).__init__()
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        self.project = project

        self.progress_callback = progress_callback

        self.solver = None

        self.tracker_tab = TrackerWidget(project, show_in_visualizer_callback=self.show_in_visualizer)
        self.tabs = QtGui.QTabWidget(self)

        # TODO: takes too much space
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
        self.region_classifier = QtGui.QWidget()

        self.id_detection_tab = QtGui.QWidget()
        # self.id_detection_tab = LearningWidget(self.project, self.play_and_highlight_tracklet,
        #                                        progressbar_callback=progress_callback)

        self.finish_callback = finish_callback

        self.tab_widgets = [self.tracker_tab, self.results_tab, self.id_detection_tab, self.statistics_tab, self.graph_tab, self.region_classifier]
        self.tab_names = ["-", "results viewer", "id detection", "stats && results", "graph", "region classifier"]
        self.tab_docked = [False] * len(self.tab_widgets)
        for i in range(len(self.tab_widgets)):
            self.tabs.addTab(self.tab_widgets[i], self.tab_names[i])
            self.tabs.setDisabled(i)
            # self.tabs.setEnabled(i)

        self.switch_to_tracking_window_action = QtGui.QAction('switch tab to tracking', self)
        self.switch_to_tracking_window_action.triggered.connect(partial(self.tabs.setCurrentIndex, 0))
        self.switch_to_tracking_window_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_T))
        self.addAction(self.switch_to_tracking_window_action)

        self.vbox.addWidget(self.tabs)
        # self.layout().addWidget(self.m)

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

        # progress bar setup

        # label showing "Part I of N: -----"
        self.progress_label = QtGui.QLabel()
        self.layout().addWidget(self.progress_label)
        # number N of processes called with progress bar
        self.total_progress_bar_usages = 8

        self.progress_bar = QtGui.QProgressBar(self)
        self.progress_bar.setGeometry(200, 80, 250, 20)
        #
        self.progress_bar_step_size = None

        self.progress_bar_part_counter = 1
        self.detailed_progress_size = None
        self.progress_bar_current_value = 0

        self.layout().addWidget(self.progress_bar)

        print "LOADING GRAPH..."
        if project.gm is None or project.gm.g.num_vertices() == 0:

            # prepare background computer
            self.bc_msers = BackgroundComputer(project, self.background_computer_finished, postpone_parallelisation)

            # connect signals (these are used instead of callbacks to work with threads/processes better)
            # bg_comp is given only one callback (self.background_computer_finished)
            self.bc_msers.next_step_progress_signal.connect(self.start_new_progress)
            self.bc_msers.update_progress_signal.connect(self.update_progress)

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

    def play_and_highlight_tracklet(self, tracklet, frame=-1, margin=0):
        self.tabs.setCurrentIndex(1)
        self.results_tab.play_and_highlight_tracklet(tracklet, frame=frame, margin=margin)

    def decide_tracklet(self, tracklet, id_=None):
        # self.tab_changed(2)
        if not self.id_detection_tab:
            self.tab_changed(2)
        self.id_detection_tab.decide_tracklet_question(tracklet, id_=id_)

    def edit_tracklet(self, tracklet):
        if not self.id_detection_tab:
            self.tab_changed(2)

        self.id_detection_tab.edit_tracklet(tracklet)

    def get_separated_frame(self):
        if not self.id_detection_tab:
            self.tab_changed(2)

        return self.id_detection_tab.get_separated_frame()

    def update_N_sets(self):
        if not self.id_detection_tab:
            self.tab_changed(2)

        return self.id_detection_tab.update_N_sets()

    def tracklet_measurements(self, id_):
        if not self.id_detection_tab:
            self.tab_changed(2)

        return self.id_detection_tab.tracklet_measurements(id_)

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
                                                                'edit_tracklet': self.edit_tracklet,
                                                                'get_separated_frame': self.get_separated_frame,
                                                                'update_N_sets': self.update_N_sets,
                                                                'tracklet_measurements': self.tracklet_measurements})
                    # self.results_tab.redraw_video_player_visualisations()
                    self.tabs.insertTab(1, self.results_tab, 'results viewer')
                    self.tabs.setCurrentIndex(1)
                    self.ignore_tab_change = False

                self.results_tab.update_visualisations()

        if i == 2:
            # TODO: show loading or something...
            if not isinstance(self.id_detection_tab, LearningWidget):
                ok = False
                for ch in self.project.chm.chunks_in_frame(0):
                    if not ch.is_undefined():
                        ok = True
                        break

                if not ok:
                    box = QtGui.QMessageBox()
                    box.setText("there is 0 tracklets with proper class (single-ID, multi-ID, no-ID, part-of-ID) in "
                                "frame 0, most likely you need to continue to region classifier tab and do tracklet "
                                "classification first. Continue with id detection only if you are aware of what you "
                                "are doing.")
                    box.setIcon(QtGui.QMessageBox.Warning)
                    box.show()

                self.ignore_tab_change = True
                self.tabs.removeTab(2)
                self.id_detection_tab.setParent(None)
                self.id_detection_tab = LearningWidget(self.project, self.play_and_highlight_tracklet, self.progress_callback)
                self.id_detection_tab.update_callback()
                self.tabs.insertTab(2, self.id_detection_tab, "id detection")
                self.tabs.setCurrentIndex(2)
                self.ignore_tab_change = False

            if not len(self.id_detection_tab.lp.features):
                self.id_detection_tab.disable_before_features()

        if i == 3:
            self.statistics_tab.update_data(self.project)
        if i == 4:
            from utils.video_manager import get_auto_video_manager
            vm = get_auto_video_manager(self.project)
            max_f = vm.total_frame_count()

            from_frame, ok = QtGui.QInputDialog.getInt(self, "show range", "From: ", 0, 0, max_f-1)
            if ok:
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

        if i == 5:
            from gui.region_classifier_tool import RegionClassifierTool

            self.ignore_tab_change = True
            self.tabs.removeTab(5)
            self.region_classifier.setParent(None)
            self.region_classifier = RegionClassifierTool(self.project)
            self.tabs.insertTab(5, self.region_classifier, "region classifier")
            self.tabs.setCurrentIndex(5)
            self.ignore_tab_change = False

            # self.region_classifier.human_iloop_classification(sort=True)
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

    def background_computer_finished(self, solver):
        print "GRAPH LOADED"
        self.solver = solver
        self.results_tab.solver = solver

        for i in range(len(self.tab_widgets)):
            self.tabs.setEnabled(i)

        self.tabs.setCurrentIndex(1)
        self.tab_changed(1)

        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

    def start_new_progress(self, num_parts, name=""):
        """
        Creates a new (empty) progress bar with specified size and name.
        
        Usage:
        The progress bar is designed to be used multiple times. The number of usages is hardcoded in this class (in
        self.total_progress_bar_usages) and is currently 8. The start_new_progress and update_progress methods are 
        called with pyqt signal.
        
        Use this method to initialize a new progress bar. Use update_progress to add one part (eg. once in every loop)
        to increase the progress bar.
        
        There are now 8 progress bars used when FERDA is creating a new project, you can find them here:
         - background_computer.py, line 85 (computing MSERs, updates are called from line 183)
         - bg_computer_assembling.py, line 11 (updates on lines 37, 79, 83)
         - bg_computer_assembling.py, line 121
           -> solver.one2one, line 84 (resetting itree)
         - bg_computer_assembling.py, line 137
           -> region_stats.add_score_to_edges, line 1343 (Calculating edge score,updates on 1358)
         - bg_computer_assembling.py, line 147 (resetting itree)
         - bg_computer_assembling.py, line 198 (resetting itree)
         - bg_computer_assembling.py, line 213
           -> chunk_manager.add_single_vertices_chunks, line 115 (resetting itree)
         - bg_computer_assembling.py, line 213
           -> chunk_manager.add_single_vertices_chunks, line 144 (resetting itree)
        
        :param num_parts: Number of updates that will fill the bar (eg. number of iterations in a loop)
        :param name: Optional name that will show above the progress bar: "Part x of y: name"
        """
        print "New progress bar: " + name

        # update label "Part 2 of 6: Computing MSERs"
        self.progress_label.setText("Part {} of {}{}".format(self.progress_bar_part_counter,
                                                             self.total_progress_bar_usages,
                                                             "" if name == "" else ": {}".format(name)))

        # update part counter
        self.progress_bar_part_counter += 1

        # compute new step for detailed progress_bar
        self.progress_bar_step_size = 100.0 / float(num_parts)

        # start progress bar from 0
        self.progress_bar_current_value = 0
        self.progress_bar.setValue(self.progress_bar_current_value)

        # update current QWidget - this is necessary to show updates from other processes
        self.update()

    def update_progress(self, jump=1):
        """
        Increase the progress bar by one chunk (they have different sizes depending on num_parts in start_new_progress)
        :param jump: optional argument to increase the bar by multiple chunks at once
        """
        # compute new progress bar value
        self.progress_bar_current_value = self.progress_bar_current_value + self.progress_bar_step_size * jump
        self.progress_bar.setValue(self.progress_bar_current_value)

        # update current QWidget - this is necessary to show updates from other processes
        QtGui.qApp.processEvents()
        self.update()


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


