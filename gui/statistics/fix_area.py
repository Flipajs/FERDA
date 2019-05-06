__author__ = 'flipajs'


import sys
import cv2
import numpy as np

from PyQt4 import QtGui
from PyQt4 import QtCore
from skimage.transform import resize
from core.region.mser import get_regions_in_img
from core.project.project import Project
from gui.img_controls.gui_utils import cvimg2qtpixmap
from scripts.region_graph3 import visualize_nodes
from utils.video_manager import get_auto_video_manager
from core.region.mser import get_regions_in_img
from core.graph.reduced import Reduced
import scipy.io as sio
from scipy.spatial import ConvexHull
import time
from gui.loading_widget import LoadingWidget

class AreaUpdatingThread(QtCore.QThread):
    proc_done = QtCore.pyqtSignal(object)
    part_done = QtCore.pyqtSignal(float)

    def __init__(self, project, solver, step):
        super(AreaUpdatingThread, self).__init__()
        self.project = project
        self.solver = solver
        self.step = step

    def run(self):
        vid = get_auto_video_manager(self.project)

        register = self.chunk_register(self.solver)
        for f in range(self.step, vid.total_frame_count(), self.step):
            if f in register:
                img = vid.get_frame(f)
                msers = get_regions_in_img(img, self.project, frame=f, prefiltered=True)

                for ch in register[f]:
                    c = ch.get_centroid_in_time(f)
                    best_d = 10
                    best_match = None
                    for m in msers:
                        d = np.linalg.norm(m.centroid()-c)
                        if d < best_d:
                            best_match = m
                            best_d = d

                    if best_match:
                        ch.statistics['num_of_reasonable_regions'] += 1
                        ch.statistics['area_sum'] += best_match.area()
                        ch.statistics['area2_sum'] += best_match.area()**2

            self.part_done.emit(f/float(vid.total_frame_count()))

        self.proc_done.emit(self.solver)

    def clear_chunks_area_info(self, chunks):
        for ch in chunks:
            ch.statistics['num_of_reasonable_regions'] = 0
            ch.statistics['area_sum'] = 0
            ch.statistics['area2_sum'] = 0

    def chunk_register(self, solver):
        chunks = solver.chunk_list()
        self.clear_chunks_area_info(chunks)

        register = {}
        for ch in chunks:
            start_t = (ch.start_t() / self.step) * (self.step + 1)
            end_t = (ch.end_t() / self.step) * (self.step - 1)

            for f in range(start_t, end_t, self.step):
                register.setdefault(f, []).append(ch)

        return register


class FixArea(QtGui.QWidget):
    def __init__(self, project, solver):
        super(FixArea, self).__init__()
        self.project = project
        self.solver = solver

        self.setLayout(QtGui.QVBoxLayout())

        self.groupBox = QtGui.QGroupBox('Fix area.')
        self.layout().addWidget(self.groupBox)
        self.vbox = QtGui.QVBoxLayout()
        self.groupBox.setLayout(self.vbox)

        self.fbox = QtGui.QFormLayout()
        self.form = QtGui.QWidget()
        self.form.setLayout(self.fbox)
        self.vbox.addWidget(self.form)

        self.step = QtGui.QSpinBox()
        self.step.setMinimum(1)
        self.step.setMaximum(10000)
        self.step.setSingleStep(1)
        self.step.setValue(10)

        self.fbox.addRow('sets the sampling step', self.step)

        self.fix_b = QtGui.QPushButton('fix')
        self.fix_b.clicked.connect(self.fix)
        self.fbox.addRow('', self.fix_b)

        self.loading_w = None

    def fix(self):
        self.form.hide()

        loading_w = LoadingWidget(text='Updating area info...')
        self.loading_w = loading_w
        self.vbox.addWidget(loading_w)
        QtGui.QApplication.processEvents()

        self.area_updating_thread = AreaUpdatingThread(self.project, self.solver, self.step.value())
        self.area_updating_thread.proc_done.connect(self.updating_finished)
        self.area_updating_thread.part_done.connect(loading_w.update_progress)
        self.area_updating_thread.start()

    def updating_finished(self, solver):
        self.solver = solver
        self.loading_w.hide()
        self.solver.save()
        self.vbox.addWidget(QtGui.QLabel('AREA info is updated now... Project and progress was saved.'))
        self.project.version = "2.2.10"
        self.project.save()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    p = Project()
    p.load('/Users/flipajs/Documents/wd/eight_22/eight22.fproj')

    ex = FixArea(p, p.solver)
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()