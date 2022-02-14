__author__ = 'flipajs'

from gui.results.noise_filter_computer import NoiseFilterComputer
from gui.gui_utils import get_img_qlabel
from utils.video_manager import get_auto_video_manager
import sys
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np
import pickle
from functools import partial
from core.region.fitting import Fitting
from copy import deepcopy
from .case_widget import CaseWidget
from .new_region_widget import NewRegionWidget
from core.region.region import Region
from core.log import LogCategories, ActionNames
from gui.img_grid.img_grid_widget import ImgGridWidget
from gui.settings import Settings as S_
import math
from gui.view.graph_visualizer import call_visualizer
from gui.loading_widget import LoadingWidget



class GlobalView(QtWidgets.QWidget):
    def __init__(self, project, solver, show_in_visualizer_callback=None):
        super(GlobalView, self).__init__()

        self.setLayout(QtWidgets.QVBoxLayout())
        self.show_in_visualizer_callback = show_in_visualizer_callback
        self.project = project
        self.solver = solver

        self.chunk_len_thresholdd = None
        self.start_t = None
        self.end_t = None
        self.show_b = None
        self.node_size = None
        self.relative_margin = None

        self.tool_w = self.create_tool_w()
        self.last_margin = -1
        self.last_node_size = -1

    def create_tool_w(self):
        w = QtWidgets.QWidget()
        w.setLayout(QtWidgets.QHBoxLayout())

        vid = get_auto_video_manager(self.project)

        self.chunk_len_threshold = QtWidgets.QSpinBox()
        self.chunk_len_threshold.setMinimum(0)
        self.chunk_len_threshold.setMaximum(100000)
        self.chunk_len_threshold.setValue(10)
        w.layout().addWidget(QtWidgets.QLabel('min chunk length: '))
        w.layout().addWidget(self.chunk_len_threshold)

        self.start_t = QtWidgets.QSpinBox()
        self.start_t.setMinimum(-1)
        self.start_t.setMaximum(vid.total_frame_count()-1)
        self.start_t.setValue(0)
        w.layout().addWidget(QtWidgets.QLabel('start t:'))
        w.layout().addWidget(self.start_t)

        self.end_t = QtWidgets.QSpinBox()
        self.end_t.setMinimum(-1)
        self.end_t.setMaximum(vid.total_frame_count())
        self.end_t.setValue(vid.total_frame_count())
        w.layout().addWidget(QtWidgets.QLabel('end t:'))
        w.layout().addWidget(self.end_t)

        self.node_size = QtWidgets.QSpinBox()
        self.node_size.setMinimum(5)
        self.node_size.setMaximum(500)
        self.node_size.setValue(100)
        w.layout().addWidget(QtWidgets.QLabel('node size: '))
        w.layout().addWidget(self.node_size)

        self.relative_margin = QtWidgets.QDoubleSpinBox()
        self.relative_margin.setMinimum(0)
        self.relative_margin.setMaximum(5)
        self.relative_margin.setSingleStep(0.05)
        self.relative_margin.setValue(1.7)
        w.layout().addWidget(QtWidgets.QLabel('node relative margin: '))
        w.layout().addWidget(self.relative_margin)

        self.show_vertically = QtWidgets.QCheckBox()
        self.show_vertically.setChecked(False)
        w.layout().addWidget(QtWidgets.QLabel('show vertically:'))
        w.layout().addWidget(self.show_vertically)

        self.show_b = QtWidgets.QPushButton('show')
        self.show_b.clicked.connect(self.start_preparation)
        w.layout().addWidget(self.show_b)

        self.visualizer = None

        return w

    def start_preparation(self):
        # clear
        while self.layout().count():
            it = self.layout().layout().itemAt(0)
            self.layout().removeItem(it)
            it.widget().setParent(None)

        w_loading = LoadingWidget()
        self.layout().addWidget(w_loading)
        QtWidgets.QApplication.processEvents()

        start_t = self.start_t.value()
        end_t = self.end_t.value()
        min_chunk_len = self.chunk_len_threshold.value()

        self.project.solver_parameters.global_view_min_chunk_len = min_chunk_len

        node_size = self.node_size.value()
        margin = self.relative_margin.value()

        reset_cache = False
        if self.last_node_size != node_size or self.last_margin != margin:
            reset_cache = True

        self.last_node_size = node_size
        self.last_margin = margin

        show_vertically = self.show_vertically.isChecked()
        self.visualizer = call_visualizer(start_t, end_t, self.project, self.solver, min_chunk_len, w_loading.update_progress, node_size=node_size, node_margin=margin, show_in_visualizer_callback=self.show_in_visualizer_callback, reset_cache=reset_cache, show_vertically=show_vertically)
        w_loading.hide()
        self.layout().addWidget(self.visualizer)

    def update_content(self):
        self.start_preparation()
        # self.visualizer.update_view()
