from PyQt4 import QtCore
from utils.img import prepare_for_segmentation
from utils.video_manager import optimize_frame_access, get_auto_video_manager

__author__ = 'flipajs'


class NoiseFilterComputer(QtCore.QThread):
    proc_done = QtCore.pyqtSignal(bool)
    part_done = QtCore.pyqtSignal(float, object, object)
    set_range = QtCore.pyqtSignal(int)

    def __init__(self, solver, project, steps):
        super(NoiseFilterComputer, self).__init__()
        self.solver = solver
        self.steps = steps
        self.project = project

    def run(self):
        # TODO: add some settings...
        th = 0.2

        to_process = []
        for n in self.solver.g:
            prob = self.solver.get_antlikeness(n)

            if prob < th:
                to_process.append(n)

        optimized = optimize_frame_access(to_process)
        vid = get_auto_video_manager(self.project.video_paths)

        self.set_range.emit(len(optimized))

        i = 0
        for n, seq, _ in optimized:
            if seq:
                while vid.frame_number() < n.frame_:
                    vid.move2_next()

                img = vid.img()
            else:
                img = vid.seek_frame(n.frame_)

            img = prepare_for_segmentation(img, self.project, grayscale_speedup=False)

            self.part_done.emit(i, img, n)

            i += 1

            if i > self.steps:
                break

        self.proc_done.emit(True)