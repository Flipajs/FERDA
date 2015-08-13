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
    def __init__(self, project, solver):
        super(GlobalView, self).__init__()

        self.setLayout(QtGui.QVBoxLayout())

        self.project = project
        self.solver = solver

        self.chunk_len_thresholdd = None
        self.start_t = None
        self.end_t = None
        self.show_b = None
        
        self.tool_w = self.create_tool_w()

    def create_tool_w(self):
        w = QtGui.QWidget()
        w.setLayout(QtGui.QHBoxLayout())

        vid = get_auto_video_manager(self.project.video_paths)

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

        self.show_b = QtGui.QPushButton('show')
        self.show_b.clicked.connect(self.start_preparation)
        w.layout().addWidget(self.show_b)

        return w

    def start_preparation(self):
        w_loading = LoadingWidget()
        self.layout().addWidget(w_loading)
        QtGui.QApplication.processEvents()

        start_t = self.start_t.value()
        end_t = self.end_t.value()
        min_chunk_len = self.chunk_len_threshold.value()
        w = call_visualizer(start_t, end_t, self.project, self.solver, min_chunk_len, w_loading.update_progress)
        w_loading.hide()
        self.layout().addWidget(w)