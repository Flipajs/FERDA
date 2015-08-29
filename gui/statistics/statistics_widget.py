__author__ = 'fnaiser'

from PyQt4 import QtGui
import csv
import scipy.io as sio
import numpy as np
from region_reconstruction import RegionReconstruction
from fix_area import FixArea


class StatisticsWidget(QtGui.QWidget):
    def __init__(self, project):
        super(StatisticsWidget, self).__init__()

        self.project = project
        self.solver = None

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        self.fbox = QtGui.QFormLayout()
        self.vbox.addLayout(self.fbox)

        self.num_of_single_nodes = QtGui.QLabel('-1')
        self.fbox.addRow('Nodes num:', self.num_of_single_nodes)

        self.num_of_chunks = QtGui.QLabel('-1')
        self.fbox.addRow('Chunks num:', self.num_of_chunks)

        self.fbox.addRow('Min certainty value: ', QtGui.QLabel(str(self.project.solver_parameters.certainty_threshold)))

        self.export_fbox = QtGui.QFormLayout()
        self.vbox.addLayout(self.export_fbox)

        self.export_name = QtGui.QLineEdit('out')
        self.export_fbox.addRow('output name', self.export_name)

        # self.export_trajectories = QtGui.QCheckBox('')
        # self.export_trajectories.setChecked(True)
        # self.export_fbox.addRow('export trajectories', self.export_trajectories)

        self.x_axis_first = QtGui.QCheckBox('')
        self.x_axis_first.setChecked(True)
        self.export_fbox.addRow('x axis first', self.x_axis_first)

        self.file_type = QtGui.QComboBox()
        self.file_type.addItem('.csv')
        self.file_type.addItem('.mat')
        self.file_type.addItem('.txt')
        self.file_type.setCurrentIndex(1)

        self.export_fbox.addRow('file type', self.file_type)

        self.export_b = QtGui.QPushButton('export')
        self.export_b.clicked.connect(self.export)
        self.export_fbox.addRow(self.export_b)

        self.region_reconstruction = RegionReconstruction(project, solver=None)
        self.vbox.addWidget(self.region_reconstruction)

        if project.version_is_le('2.2.9'):
            self.fix_area = FixArea(project, solver=None)
            self.vbox.addWidget(self.fix_area)

    def export(self):
        print "exporting..."
        ftype = self.file_type.currentText()
        if ftype == '.txt':
            self.export_txt()
        elif ftype == '.csv':
            self.export_csv()
        elif ftype == '.mat':
            self.export_mat()

        print "done"

    def write_line_csv(self, f, r):
        a, b = self.centroid_in_right_order(r)
        f.writerow([str(r.frame_), round(a, 2), round(b, 2)])

    def export_csv(self):
        chunks = self.solver.chunk_list()
        chunks = sorted(chunks, key=lambda x: x.start_n.frame_)

        with open(self.get_out_path()+'.csv', 'wb') as f:
            csv_f = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            a = "y"
            b = "x"
            if self.x_axis_first.isChecked():
                a, b = b, a

            csv_f.writerow(["frame", a, b])

            id_ = 0
            for ch in chunks:
                csv_f.writerow(["CHUNK ID",  str(id_)])
                self.write_line_csv(csv_f, ch.start_n)
                for r in ch.reduced:
                    self.write_line_csv(csv_f, r)

                self.write_line_csv(csv_f, ch.end_n)
                id_ += 1

                csv_f.writerow([])

    def export_mat(self):
        chunks = self.solver.chunk_list()
        chunks = sorted(chunks, key=lambda x: x.start_n.frame_)

        obj_arr = []

        id_ = 0
        for ch in chunks:
            d = {'x': [], 'y': [], 'frame': []}

            self.add_line_mat(d, ch.start_n)
            for r in ch.reduced:
                self.add_line_mat(d, r)

            self.add_line_mat(d, ch.end_n)
            if self.project.other_parameters.store_area_info:
                mean, std = ch.get_area_stats()
                obj_arr.append({'id': id_, 'x': np.array(d['x']), 'y': np.array(d['y']), 'frame': np.array(d['frame']), 'area_mean': mean, 'area_std': std})
            else:
                obj_arr.append({'id': id_, 'x': np.array(d['x']), 'y': np.array(d['y']), 'frame': np.array(d['frame'])})
            id_ += 1

        with open(self.get_out_path()+'.mat', 'wb') as f:
            sio.savemat(f, {'FERDA': obj_arr})

    def add_line_mat(self, d, r):
        y, x = r.centroid()
        d['x'].append(x)
        d['y'].append(y)
        d['frame'].append(r.frame_)

    def get_out_path(self):
        return self.project.working_directory + '/' + self.export_name.text()

    def centroid_in_right_order(self, r):
        c = r.centroid()
        if self.x_axis_first.isChecked():
            b = c[0]
            a = c[1]
        else:
            a = c[0]
            b = c[1]

        return a, b

    def write_line_txt(self, f, r):
        a, b = self.centroid_in_right_order(r)
        f.write('#' + str(r.frame_) + '\t' + str(round(a, 2)) + '\t' + str(round(b, 2)) + '\n')

    def export_txt(self):
        chunks = self.solver.chunk_list()
        chunks = sorted(chunks, key=lambda x: x.start_n.frame_)

        with open(self.get_out_path()+'.txt', 'wb') as f:
            if self.x_axis_first.isChecked():
                f.write("FRAME\tx\ty\n\n")
            else:
                f.write("FRAME\ty\tx\n\n")

            id = 0
            for ch in chunks:
                f.write("CHUNK ID: " + str(id) + "\n")
                self.write_line_txt(f, ch.start_n)
                for r in ch.reduced:
                    self.write_line_txt(f, r)

                self.write_line_txt(f, ch.end_n)
                id += 1

                f.write("\n")

    def update_data(self, solver):
        self.solver = solver
        self.num_of_single_nodes.setText(str(len(solver.g.nodes())))
        self.num_of_chunks.setText(str(len(solver.chunk_list())))

        self.region_reconstruction.solver = solver
        self.fix_area.solver = solver
