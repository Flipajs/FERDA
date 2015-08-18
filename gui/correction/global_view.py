__author__ = 'flipajs'

from gui.correction.noise_filter_computer import NoiseFilterComputer
from gui.gui_utils import get_img_qlabel
from utils.video_manager import get_auto_video_manager
import sys
from PyQt4 import QtGui, QtCore
import numpy as np
import pickle
from functools import partial
from core.region.fitting import Fitting
from copy import deepcopy
from case_widget import CaseWidget
from new_region_widget import NewRegionWidget
from core.region.region import Region
from core.log import LogCategories, ActionNames
from gui.img_grid.img_grid_widget import ImgGridWidget
from core.settings import Settings as S_
import math
from gui.view.graph_visualizer import call_visualizer
from gui.loading_widget import LoadingWidget



class GlobalView(QtGui.QWidget):
    def __init__(self, project, solver, show_in_visualizer_callback=None):
        super(GlobalView, self).__init__()

        self.setLayout(QtGui.QVBoxLayout())
        self.show_in_visualizer_callback = show_in_visualizer_callback
        self.project = project
        self.solver = solver

        self.chunk_len_thresholdd = None
        self.start_t = None
        self.end_t = None
        self.show_b = None
        self.node_size = None
        self.relative_margin = None

        self.save_progress = QtGui.QAction('save', self)
        self.save_progress.triggered.connect(self.solver.save)
        self.save_progress.setShortcut(S_.controls.save)
        self.addAction(self.save_progress)
        
        self.tool_w = self.create_tool_w()

    def create_tool_w(self):
        w = QtGui.QWidget()
        w.setLayout(QtGui.QHBoxLayout())

        vid = get_auto_video_manager(self.project)

        self.chunk_len_threshold = QtGui.QSpinBox()
        self.chunk_len_threshold.setMinimum(0)
        self.chunk_len_threshold.setMaximum(100000)
        self.chunk_len_threshold.setValue(10)
        w.layout().addWidget(QtGui.QLabel('min chunk length: '))
        w.layout().addWidget(self.chunk_len_threshold)

        self.start_t = QtGui.QSpinBox()
        self.start_t.setMinimum(-1)
        self.start_t.setMaximum(vid.total_frame_count()-1)
        self.start_t.setValue(0)
        w.layout().addWidget(QtGui.QLabel('start t:'))
        w.layout().addWidget(self.start_t)

        self.end_t = QtGui.QSpinBox()
        self.end_t.setMinimum(-1)
        self.end_t.setMaximum(vid.total_frame_count())
        self.end_t.setValue(vid.total_frame_count())
        w.layout().addWidget(QtGui.QLabel('end t:'))
        w.layout().addWidget(self.end_t)

        self.node_size = QtGui.QSpinBox()
        self.node_size.setMinimum(5)
        self.node_size.setMaximum(500)
        self.node_size.setValue(100)
        w.layout().addWidget(QtGui.QLabel('node size: '))
        w.layout().addWidget(self.node_size)

        self.relative_margin = QtGui.QDoubleSpinBox()
        self.relative_margin.setMinimum(0)
        self.relative_margin.setMaximum(5)
        self.relative_margin.setSingleStep(0.05)
        self.relative_margin.setValue(1.7)
        w.layout().addWidget(QtGui.QLabel('node relative margin: '))
        w.layout().addWidget(self.relative_margin)

        self.show_b = QtGui.QPushButton('show')
        self.show_b.clicked.connect(self.start_preparation)
        w.layout().addWidget(self.show_b)

        return w

    def start_preparation(self):
        # clear
        while self.layout().count():
            it = self.layout().layout().itemAt(0)
            self.layout().removeItem(it)
            it.widget().setParent(None)

        w_loading = LoadingWidget()
        self.layout().addWidget(w_loading)
        QtGui.QApplication.processEvents()

        start_t = self.start_t.value()
        end_t = self.end_t.value()
        min_chunk_len = self.chunk_len_threshold.value()

        self.project.solver_parameters.global_view_min_chunk_len = min_chunk_len

        node_size = self.node_size.value()
        margin = self.relative_margin.value()

        w = call_visualizer(start_t, end_t, self.project, self.solver, min_chunk_len, w_loading.update_progress, node_size=node_size, node_margin=margin, show_in_visualizer_callback=self.show_in_visualizer_callback)
        w_loading.hide()
        self.layout().addWidget(w)