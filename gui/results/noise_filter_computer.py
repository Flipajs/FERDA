from PyQt5 import QtCore
from utils.img import prepare_for_segmentation
from utils.video_manager import optimize_frame_access, get_auto_video_manager

__author__ = 'flipajs'


class NoiseFilterComputer(QtCore.QThread):
    proc_done = QtCore.pyqtSignal(bool)
    part_done = QtCore.pyqtSignal(float, object, object, object)
    set_range = QtCore.pyqtSignal(int)

    def __init__(self, solver, project, steps, threshold=0.2):
        super(NoiseFilterComputer, self).__init__()
        self.solver = solver
        self.steps = steps
        self.project = project
        self.threshold = threshold

    def run(self):
        # TODO: add some settings...
        th = self.threshold

        to_process = []

        r2v_mapping = {}
        for v in self.project.gm.get_all_relevant_vertices():
            r = self.project.gm.region(v)
            r2v_mapping[r] = v
            prob = self.solver.get_antlikeness(r)

            if prob < th:
                to_process.append(r)

        optimized = optimize_frame_access(to_process)
        vid = get_auto_video_manager(self.project)

        self.set_range.emit(len(optimized))

        i = 0
        for r, seq, _ in optimized:
            if seq:
                while vid.frame_number() < r.frame_:
                    vid.next_frame()

                img = vid.img()
            else:
                img = vid.seek_frame(r.frame_)

            img = prepare_for_segmentation(img, self.project, grayscale_speedup=False)

            self.part_done.emit(i, img, r, r2v_mapping[r])

            i += 1

            if i > self.steps:
                break

        self.proc_done.emit(True)
