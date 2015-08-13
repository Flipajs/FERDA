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


class TrackerWidget(QtGui.QWidget):
    def __init__(self, project):
        super(TrackerWidget, self).__init__()
        self.project = project

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.mser_progress_label = QtGui.QLabel('MSER computation progress')
        self.vbox.addWidget(self.mser_progress_label)

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

        self.global_view = None
        self.noise_filter = None

        self.graph_widget = QtGui.QWidget()
        self.layout().addWidget(self.graph_widget)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def clear_layout(self, layout):
        while layout.count():
            it = layout.itemAt(0)
            layout.removeItem(it)
            it.widget().setParent(None)

    def clear_tool_row(self):
        if self.tool_row.count():
            it = self.tool_row.itemAt(0)
            self.tool_row.removeItem(it)
            it.widget().setParent(None)

    def clear_tool_w(self):
        pass

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
        step_by_step = ConfigurationsVisualizer(self.solver, get_auto_video_manager(self.project.video_paths))
        self.tool.addWidget(step_by_step)
        self.tool_row.addWidget(step_by_step.tool_w)
        step_by_step.next_case()

    def show_global_view(self):
        global_view = GlobalView(self.project, self.solver)
        self.tool.addWidget(global_view)
        self.tool_row.addWidget(global_view.tool_w)

    def bc_update(self, text):
        self.mser_progress_label.setText('MSER computation progress'+text)

    def region_ccs_refs(self, ccs):
        refs = {}
        for c_ in ccs:
            for r in c_['c1']:
                if r in refs:
                    refs[r].append(c_)

    def update_graph_visu(self, t_start=-1, t_end=-1):
        if t_start == t_end == -1:
            sub_g = self.solver.g
        else:
            nodes = []
            for n in self.solver.g.nodes():
                if t_start <= n.frame_ < t_end:
                    nodes.append(n)

            sub_g = self.solver.g.subgraph(nodes)

        vid = get_auto_video_manager(self.project.video_paths)
        regions = {}

        optimized = optimize_frame_access(sub_g.nodes())

        for n, seq, _ in optimized:
            if n.frame_ in regions:
                regions[n.frame_].append(n)
            else:
                regions[n.frame_] = [n]

            if 'img' not in self.solver.g.node[n]:
                if seq:
                    while vid.frame_number() < n.frame_:
                        vid.move2_next()

                    im = vid.img()
                else:
                    im = vid.seek_frame(n.frame_)

                if S_.mser.img_subsample_factor > 1.0:
                    im = np.asarray(rescale(im, 1/S_.mser.img_subsample_factor) * 255, dtype=np.uint8)

                self.solver.g.node[n]['img'] = visualize_nodes(im, n)
                sub_g.node[n]['img'] = self.solver.g.node[n]['img']

        ngv = NodeGraphVisualizer(sub_g, [], regions)
        ngv.visualize()
        self.layout().removeWidget(self.graph_widget)
        self.graph_widget.setParent(None)
        self.graph_widget = ngv
        # self.layout().addWidget(self.graph_widget)
        self.graph_widget.showMaximized()
        # self.graph_widget.setFixedHeight(300)

    def prepare_corrections(self, solver):
        self.solver = solver
        self.mser_progress_label.setParent(None)
        self.mode_changed()

    def apply_actions(self, actions):
        for action_name, data in actions:
            if action_name == 'new_region':
                self.certainty_visualizer.new_region_finished(data[0], data[1])
            elif action_name == 'remove_region':
                self.certainty_visualizer.remove_region()
            elif action_name == 'choose_node':
                self.certainty_visualizer.choose_node(data)
            elif action_name == 'next':
                self.certainty_visualizer.next()
            elif action_name == 'prev':
                self.certainty_visualizer.prev()
            elif action_name == 'fitting':
                self.certainty_visualizer.fitting()
            elif action_name == 'partially_confirm':
                self.certainty_visualizer.partially_confirm()
            elif action_name == 'confirm_edges':
                self.certainty_visualizer.confirm_edges(data)
            elif action_name == 'merged':
                self.certainty_visualizer.merged(data[0], data[1], data[2])
            else:
                # TODO: raise EXCEPTION
                print "APPLY ACTIONS ERROR: unknown ACTION"