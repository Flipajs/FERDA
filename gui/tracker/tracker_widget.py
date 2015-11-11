__author__ = 'fnaiser'

from PyQt4 import QtGui

from gui.correction.configurations_visualizer import ConfigurationsVisualizer
from utils.video_manager import get_auto_video_manager
from scripts.region_graph2 import NodeGraphVisualizer, visualize_nodes
from core.settings import Settings as S_
import numpy as np
from skimage.transform import rescale
from core.graph.configuration import get_length_of_longest_chunk
from utils.video_manager import optimize_frame_access
from gui.correction.global_view import GlobalView
from gui.loading_widget import LoadingWidget
from core.log import LogCategories, ActionNames


class TrackerWidget(QtGui.QWidget):
    def __init__(self, project, show_in_visualizer_callback=None):
        super(TrackerWidget, self).__init__()
        self.project = project

        self.show_in_visualizer_callback = show_in_visualizer_callback

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.solver = None
        self.certainty_visualizer = None

        # TOP row with tools
        self.top_row = QtGui.QHBoxLayout()
        self.tool_row = QtGui.QHBoxLayout()
        self.layout().addLayout(self.top_row)

        self.mode_selectbox = QtGui.QComboBox()
        self.mode_selectbox.addItem('step by step')
        self.mode_selectbox.addItem('global view')
        self.mode_selectbox.addItem('noise filter')
        self.mode_selectbox.currentIndexChanged.connect(self.mode_changed)
        self.top_row.addWidget(self.mode_selectbox)

        self.top_row.addLayout(self.tool_row)

        # noise toolbox
        self.mode_tools_noise = QtGui.QWidget()
        self.mode_tools_noise.setLayout(QtGui.QHBoxLayout())
        self.tool_row.addWidget(self.mode_tools_noise)

        self.noise_nodes_confirm_b = QtGui.QPushButton('remove selected')
        # self.noise_nodes_confirm_b.clicked.connect(self.remove_noise)
        self.mode_tools_noise.layout().addWidget(self.noise_nodes_confirm_b)

        self.noise_nodes_back_b = QtGui.QPushButton('back')
        # self.noise_nodes_back_b.clicked.connect(self.remove_noise_back)
        self.mode_tools_noise.layout().addWidget(self.noise_nodes_back_b)
        self.mode_tools_noise.hide()

        self.tool_row.addWidget(self.mode_tools_noise)

        self.tool = QtGui.QVBoxLayout()
        self.layout().addLayout(self.tool)

        self.progress_w = LoadingWidget(text='Computing MSERs and constructing graph...')
        self.layout().addWidget(self.progress_w)
        self.noise_filter = None

        self.undo_action = QtGui.QAction('undo', self)
        self.undo_action.triggered.connect(self.undo)
        self.undo_action.setShortcut(S_.controls.undo)
        self.addAction(self.undo_action)

        self.layout().setContentsMargins(0, 0, 0, 0)

    def undo(self):
        S_.general.log_graph_edits = False

        log = self.solver.project.log
        last_actions = log.pop_last_user_action()

        solver = self.solver

        i = 0
        ignore_node = False
        for a in last_actions:
            if a.action_name == ActionNames.ADD_NODE:
                solver.remove_node(a.data)
            elif a.action_name == ActionNames.REMOVE_NODE:
                solver.add_node(a.data)
            elif a.action_name == ActionNames.ADD_EDGE:
                try:
                    solver.remove_edge(a.data['n1'], a.data['n2'])
                except:
                    print "NOT EXISTING EDGE"
            elif a.action_name == ActionNames.REMOVE_EDGE:
                solver.add_edge(a.data['n1'], a.data['n2'], **a.data['data'])
            elif a.action_name == ActionNames.CHUNK_ADD_TO_REDUCED:
                a.data['chunk'].remove_from_reduced_(-1, self.solver)
                a.data['chunk'].is_sorted = False
            elif a.action_name == ActionNames.CHUNK_REMOVE_FROM_REDUCED:
                a.data['chunk'].add_to_reduced_(a.data['node'], self.solver, a.data['index'])
                a.data['chunk'].is_sorted = False
            elif a.action_name == ActionNames.CHUNK_SET_START:
                a.data['chunk'].set_start(a.data['old_start_n'], self.solver)
            elif a.action_name == ActionNames.CHUNK_SET_END:
                a.data['chunk'].set_end(a.data['old_end_n'], self.solver)
            elif a.action_name == ActionNames.IGNORE_NODE:
                del self.solver.ignored_nodes[a.data]
                ignore_node = True

            i += 1

        S_.general.log_graph_edits = True

        if ignore_node:
            self.active_node_id -= 1

        tool_w = self.tool.itemAt(0).widget()

        if isinstance(tool_w, ConfigurationsVisualizer):
            tool_w.update_content()
        elif isinstance(tool_w, GlobalView):
            tool_w.update_content()

    def clear_layout(self, layout):
        while layout.count():
            it = layout.itemAt(0)
            layout.removeItem(it)
            it.widget().setParent(None)

    def mode_changed(self):
        self.clear_layout(self.tool_row)
        self.clear_layout(self.tool)

        ct = self.mode_selectbox.currentText()
        if ct == 'step by step':
            self.show_step_by_step()
        elif ct == 'global view':
            self.show_global_view()
        elif ct == 'noise filter':
            self.mode_tools_noise.show()
            self.noise_nodes_filter()

    def show_step_by_step(self):
        """

        Returns:

        """
        step_by_step = ConfigurationsVisualizer(self.solver, get_auto_video_manager(self.project))
        self.tool.addWidget(step_by_step)
        self.tool_row.addWidget(step_by_step.tool_w)
        step_by_step.next_case()

    def show_global_view(self):
        global_view = GlobalView(self.project, self.solver, show_in_visualizer_callback=self.show_in_visualizer_callback)
        self.tool.addWidget(global_view)
        self.tool_row.addWidget(global_view.tool_w)

    def bc_update(self, val=-1, text=''):
        if len(text):
            self.progress_w.update_text(text)

        if val >= 0:
            self.progress_w.update_progress(val)

    def prepare_corrections(self, solver):
        self.solver = solver
        self.progress_w.hide()
        self.mode_changed()

        self.save_progress = QtGui.QAction('save', self)
        self.save_progress.triggered.connect(self.solver.save)
        self.save_progress.setShortcut(S_.controls.save)
        self.addAction(self.save_progress)

        self.save_progress_only_chunks_action = QtGui.QAction('save only chunks', self)
        self.save_progress_only_chunks_action.triggered.connect(self.solver.save_progress_only_chunks)
        self.save_progress_only_chunks_action.setShortcut(S_.controls.save_only_long_enough)
        self.addAction(self.save_progress_only_chunks_action)