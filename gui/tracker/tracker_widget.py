__author__ = 'fnaiser'

from PyQt4 import QtGui

from core.background_computer import BackgroundComputer
from gui.correction.certainty import CertaintyVisualizer
from utils.video_manager import get_auto_video_manager
# from scripts.region_graph2 import NodeGraphVisualizer, visualize_nodes

class TrackerWidget(QtGui.QWidget):
    def __init__(self, project):
        super(TrackerWidget, self).__init__()
        self.project = project

        self.vbox = QtGui.QVBoxLayout()
        b = QtGui.QPushButton('tracker')

        self.setLayout(self.vbox)
        self.vbox.addWidget(b)

        self.mser_progress_label = QtGui.QLabel('MSER computation progress')
        self.vbox.addWidget(self.mser_progress_label)

        self.solver = None
        self.certainty_visualizer = None

        if project.saved_progress:
            self.background_computer_finished(project.saved_progress['solver'])
            # self.apply_actions(project.saved_progress['actions'])
        else:
            self.bc_msers = BackgroundComputer(project, self.bc_update, self.background_computer_finished)
            self.bc_msers.run()


    def bc_update(self, text):
        self.mser_progress_label.setText('MSER computation progress'+text)

    def region_ccs_refs(self, ccs):
        refs = {}
        for c_ in ccs:
            for r in c_['c1']:
                if r in refs:
                    refs[r].append(c_)

    def background_computer_finished(self, solver):
        self.solver = solver
        self.certainty_visualizer = CertaintyVisualizer(self.solver, get_auto_video_manager(self.project.video_paths))
        self.vbox.addWidget(self.certainty_visualizer)

        ccs = self.solver.get_ccs()
        ccs = sorted(ccs, key=lambda k: k.regions_t1[0].frame_)

        # vid = get_auto_video_manager(self.project.video_paths)
        # imgs = {}
        # regions = {}
        # for n in self.solver.g.nodes():
        #     if n.frame_ in regions:
        #         regions[n.frame_].append(n)
        #     else:
        #         regions[n.frame_] = [n]
        #
        #     # if n.frame_ in imgs:
        #     #     im = imgs[n.frame_]
        #     # else:
        #     im = vid.seek_frame(n.frame_)
        #         # imgs[n.frame_] = im
        #
        #     self.solver.g.node[n]['img'] = visualize_nodes(im, n)
        #
        # ngv = NodeGraphVisualizer(self.solver.g, [], regions)
        # w_ = ngv.visualize()
        # self.layout().addWidget(w_)
        # w_.showMaximized()

        i = 0
        for c_ in ccs:
            if c_.certainty < 0.0001:
                print c_.regions_t1[0].frame_
            if i == 100:
                break

            self.certainty_visualizer.add_configuration(c_)
            i += 1

        self.certainty_visualizer.visualize_n_sorted()
        print "FINISHED"

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