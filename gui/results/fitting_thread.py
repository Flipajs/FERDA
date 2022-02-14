from PyQt6 import QtCore
from copy import deepcopy
from core.region.fitting import Fitting

class FittingThread(QtCore.QThread):
    proc_done = QtCore.pyqtSignal(object, object, int, object)

    def __init__(self, merged, model, pivot, s_id, num_of_iterations=10):
        super(FittingThread, self).__init__()

        self.merged = merged
        self.model = model
        self.num_of_iterations = num_of_iterations
        self.pivot = pivot
        self.s_id = s_id

    def run(self):
        f = Fitting(self.merged, self.model, self.num_of_iterations)
        result = f.fit()
        self.proc_done.emit(result, self.pivot, self.s_id, f)

class FittingThreadChunk(QtCore.QThread):
    proc_done = QtCore.pyqtSignal(object, object, int, object)

    def __init__(self, pivot, s_id, fitting):
        super(FittingThreadChunk, self).__init__()

        self.pivot = pivot
        self.s_id = s_id
        self.fitting = fitting

    def run(self):
        result = self.fitting.fit()
        self.proc_done.emit(result, self.pivot, self.s_id, self.fitting)
