__author__ = 'flipajs'


import sys
import cv2
import numpy as np

from PyQt4 import QtGui
from PyQt4 import QtCore
from skimage.transform import resize
from core.region.mser import get_msers_
from core.project.project import Project
from gui.img_controls.utils import cvimg2qtpixmap
from scripts.region_graph3 import visualize_nodes
from utils.video_manager import get_auto_video_manager
from core.region.mser import get_msers_
from core.graph.reduced import Reduced
import scipy.io as sio


class RegionReconstruction(QtGui.QWidget):
    def __init__(self, project, solver):
        super(RegionReconstruction, self).__init__()
        self.project = project
        self.solver = solver

        self.setLayout(QtGui.QVBoxLayout())

        self.groupBox = QtGui.QGroupBox('Reconstruct and export regions in given frames.')
        self.layout().addWidget(self.groupBox)
        self.vbox = QtGui.QVBoxLayout()
        self.groupBox.setLayout(self.vbox)

        self.fbox = QtGui.QFormLayout()
        self.vbox.addLayout(self.fbox)

        self.out_name = QtGui.QLineEdit('out_regions')
        self.fbox.addRow('output name: ', self.out_name)

        self.query = QtGui.QLineEdit('1 2')
        self.fbox.addRow('query (frame numbers separated by space', self.query)

        self.export_results = QtGui.QPushButton('export')
        self.export_results.clicked.connect(self.export)
        self.fbox.addRow('', self.export_results)

    def export(self):
        print "reconstructin & exporting..."
        query = self.query.text()
        frames = self.process_input(query)
        reconstructed = self.reconstruct(frames)

        with open(self.project.working_directory+'/'+self.out_name.text()+'.mat', 'wb') as f:
            sio.savemat(f, {'FERDA_regions': reconstructed})

        print "done"

    def reconstruct(self, frames):
        frames = sorted(frames)

        reconstructed = []
        vid = get_auto_video_manager(self.project)

        for f in frames:
            ch_in_frame = self.solver.chunks_in_frame(f)
            im = vid.get_frame(f, auto=True)
            regions = get_msers_(im, self.project, frame=f)

            for ch in ch_in_frame:
                c = ch.get_centroid_in_time(f)
                is_virtual = ch.is_virtual_in_time(f)

                r_best_match = None
                r_best_dist = 5

                if not is_virtual:
                    for r in regions:
                        d = np.linalg.norm(r.centroid() - c)
                        if d < r_best_dist:
                            r_best_dist = d
                            r_best_match = r

                xs = []
                ys = []
                if r_best_match:
                    for p in r_best_match.pts_:
                        xs.append(p[1])
                        ys.append(p[0])

                reconstructed.append({'frame': f, 'chunk_id': ch.id, 'px': xs, 'py': ys})

        return reconstructed

    def process_input(self, s):
        """
        returns list of frames
        supported formats
        1 2 3 4
        1, 2, 3, 4
        1,2,3,4
        1:1000 (returns list of 1, 2, 3,..., 999, 1000
        1:3:1000 (returns list of 1, 4, 7, .... )
        :param s:
        :return:
        """
        frames = []

        # test 1:1000 or 1:3:1000 format
        if ':' in s:
            query = s.split(':')
            if len(query) == 2:
                start_t = int(query[0])
                end_t = int(query[1]) + 1
                frames = range(start_t, end_t)
            elif len(query) == 3:
                start_t = int(query[0])
                step = int(query[1])
                end_t = int(query[2]) + 1
                frames = range(start_t, end_t, step)
        else:
            frames = s.split(',')
            if len(frames) == 1:
                frames = s.split(' ')

            frames = map(int, frames)

        return frames


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    p = Project()
    p.load('/Users/flipajs/Documents/wd/eight/eight.fproj')

    ex = RegionReconstruction(p, p.saved_progress['solver'])
    print ex.process_input('1 2 3 4')
    print ex.process_input('1, 2, 3, 4')
    print ex.process_input('1,2,3,4')
    print ex.process_input('1:30')
    print ex.process_input('1:3:30')
    # print ex.reconstruct([100, 3, 500])
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()