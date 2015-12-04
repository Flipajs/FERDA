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
import math
from gui.img_grid.img_grid_widget import ImgGridWidget
from gui.correction.noise_filter_computer import NoiseFilterComputer


class NoiseFilterWidget(QtGui.QWidget):
    def __init__(self, project, steps, elem_width, cols):
        super(NoiseFilterWidget, self).__init__()

        self.project = project

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.noise_nodes_widget = ImgGridWidget()
        self.noise_nodes_widget.reshape(cols, elem_width)

        self.progress_bar = QtGui.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.vbox.addWidget(self.progress_bar)
        self.vbox.addWidget(self.noise_nodes_widget)

        self.noise_nodes_confirm_b = QtGui.QPushButton('remove selected')
        self.noise_nodes_confirm_b.clicked.connect(self.remove_noise)
        self.vbox.addWidget(self.noise_nodes_confirm_b)

        # TODO:
        # self.undo_action = QtGui.QAction('undo', self)
        # self.undo_action.triggered.connect(self.undo)
        # self.undo_action.setShortcut(S_.controls.undo)
        # self.addAction(self.undo_action)

        self.thread = NoiseFilterComputer(project.solver, project, steps)
        self.thread.part_done.connect(self.noise_part_done_)
        self.thread.proc_done.connect(self.noise_finished_)
        self.thread.set_range.connect(self.progress_bar.setMaximum)
        self.thread.start()

    def remove_noise(self):
        # TODO: add actions
        to_remove = self.noise_nodes_widget.get_selected()
        for v in to_remove:
            # if n in self.solver.g:
            self.project.solver.strong_remove(v)

        to_confirm = self.noise_nodes_widget.get_unselected()
        for n in to_confirm:
            if n in self.solver.g:
                self.solver.g.node[n]['antlikeness'] = 1.0

        self.noise_nodes_widget.hide()
        # self.next_case()

    def noise_part_done_(self, val, img, region, vertex):
        from gui.gui_utils import get_img_qlabel

        elem_width = 200
        self.progress_bar.setValue(val)
        item = get_img_qlabel(region.pts(), img, vertex, elem_width, elem_width, filled=True)
        item.set_selected(True)
        self.noise_nodes_widget.add_item(item)

    def noise_finished_(self):
        self.progress_bar.setParent(None)
        self.noise_nodes_confirm_b.show()