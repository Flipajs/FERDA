__author__ = 'fnaiser'

from PyQt4 import QtGui

from gui.correction.configurations_visualizer import ConfigurationsVisualizer
from utils.video_manager import get_auto_video_manager
from scripts.region_graph2 import NodeGraphVisualizer, visualize_nodes
from core.settings import Settings as S_
import numpy as np
from skimage.transform import rescale
from core.graph.configuration import get_length_of_longest_chunk


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

        self.graph_widget = QtGui.QWidget()
        self.layout().addWidget(self.graph_widget)

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
        for n in sub_g.nodes():
            if n.frame_ in regions:
                regions[n.frame_].append(n)
            else:
                regions[n.frame_] = [n]

            if 'img' not in self.solver.g.node[n]:
                im = vid.seek_frame(n.frame_)
                if S_.mser.img_subsample_factor > 1.0:
                    im = np.asarray(rescale(im, 1/S_.mser.img_subsample_factor) * 255, dtype=np.uint8)

                self.solver.g.node[n]['img'] = visualize_nodes(im, n)
                sub_g.node[n]['img'] = self.solver.g.node[n]['img']

        ngv = NodeGraphVisualizer(sub_g, [], regions)
        w_ = ngv.visualize()
        self.layout().removeWidget(self.graph_widget)
        self.graph_widget.setParent(None)
        self.graph_widget = w_
        # self.layout().addWidget(self.graph_widget)
        self.graph_widget.showMaximized()
        # self.graph_widget.setFixedHeight(300)

    def prepare_corrections(self, solver):
        self.solver = solver

        self.certainty_visualizer = ConfigurationsVisualizer(self.solver, get_auto_video_manager(self.project.video_paths), self.update_graph_visu)
        self.vbox.addWidget(self.certainty_visualizer)

        t1_nodes = []
        for n in self.solver.g.nodes():
            if n.frame_ == 0:
                t1_nodes.append(n)

        self.solver.simplify()
        self.solver.simplify_to_chunks()

        ccs = self.solver.get_ccs()
        # ccs = sorted(ccs, key=lambda k: k.regions_t1[0].frame_)
        ccs = sorted(ccs, key=lambda k: (-get_length_of_longest_chunk(self.solver, k), k.regions_t1[0].frame_))
        print "NUMBER OF CASES: ", len(ccs)

        print "MAJOR AXIS MEDIAN", self.project.stats.major_axis_median, S_.solver.max_edge_distance_in_ant_length, self.solver.max_distance

        # i = 0
        for c_ in ccs:
            c_.longest_chunk_length = get_length_of_longest_chunk(self.solver, c_)
            self.certainty_visualizer.add_configuration(c_)
            # if i > 10:
            #     break

            # i += 1

        self.certainty_visualizer.visualize_n_sorted(5)

        # self.update_graph_visu(0, 10)

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