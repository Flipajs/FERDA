__author__ = 'fnaiser'

from PyQt4 import QtGui
import csv
import scipy.io as sio
import numpy as np
from region_reconstruction import RegionReconstruction
from fix_area import FixArea
import sys
from core.graph.region_chunk import RegionChunk
from pympler import asizeof
import gc
from scripts.export.export_part import Exporter, export_arena


class StatisticsWidget(QtGui.QWidget):
    def __init__(self, project):
        super(StatisticsWidget, self).__init__()

        self.project = project

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        self.fbox = QtGui.QFormLayout()
        self.vbox.addLayout(self.fbox)

        self.num_of_single_nodes = QtGui.QLabel('-1')
        self.fbox.addRow('Nodes num:', self.num_of_single_nodes)

        self.num_of_chunks = QtGui.QLabel('-1')
        self.fbox.addRow('Chunks num:', self.num_of_chunks)

        self.mean_ch_len = QtGui.QLabel('-1')
        self.fbox.addRow('Chunks mean len', self.mean_ch_len)

        self.mean_ch_area = QtGui.QLabel('-1')
        self.fbox.addRow('Mean of means of chunks area', self.mean_ch_area)

        self.med_ch_area = QtGui.QLabel('-1')
        self.fbox.addRow('Med of means of chunks area', self.med_ch_area)


        self.fbox.addRow('Min certainty value: ', QtGui.QLabel(str(self.project.solver_parameters.certainty_threshold)))

        self.tracklet_coverage_step = QtGui.QLineEdit()
        self.tracklet_coverage_step.setText('10')

        self.show_tracklet_coverage_b = QtGui.QPushButton('show coverage')
        self.show_tracklet_coverage_b.clicked.connect(self.show_tracklet_coverage)
        self.fbox.addWidget(self.tracklet_coverage_step)
        self.fbox.addWidget(self.show_tracklet_coverage_b)

        self.export_fbox = QtGui.QFormLayout()
        self.vbox.addLayout(self.export_fbox)

        self.export_name = QtGui.QLineEdit('out')
        self.export_fbox.addRow('output name', self.export_name)

        # self.export_trajectories = QtGui.QCheckBox('')
        # self.export_trajectories.setChecked(True)
        # self.export_fbox.addRow('export trajectories', self.export_trajectories)

        # self.include_id = QtGui.QCheckBox('')
        # self.include_id.setChecked(True)
        # self.export_fbox.addRow('include id', self.include_id)

        # self.include_orientation = QtGui.QCheckBox('')
        # self.include_orientation.setChecked(True)
        # self.export_fbox.addRow('include orientation', self.include_orientation)

        # self.include_area = QtGui.QCheckBox('')
        # self.include_area.setChecked(True)
        # self.export_fbox.addRow('include area', self.include_area)

        # self.include_axes = QtGui.QCheckBox('')
        # self.include_axes.setChecked(True)
        # self.export_fbox.addRow('include axes (major/minor)', self.include_axes)

        self.include_region_points = QtGui.QCheckBox('')
        self.include_region_points.setChecked(True)
        self.export_fbox.addRow('include region points', self.include_region_points)

        self.include_region_contour = QtGui.QCheckBox('')
        self.export_fbox.addRow('include region contour', self.include_region_contour)

        self.export_chunks_only = QtGui.QCheckBox('')
        self.export_fbox.addRow('export chunks only', self.export_chunks_only)

        self.file_type = QtGui.QComboBox()
        # self.file_type.addItem('.csv')
        self.file_type.addItem('.mat')
        # self.file_type.addItem('.txt')
        self.file_type.setCurrentIndex(0)

        self.export_fbox.addRow('file type', self.file_type)

        # self.memory_limit_mb = QtGui.QLineEdit()
        # self.memory_limit_mb.setText('1000')
        # self.export_fbox.addRow('memory approx. limit (MB)', self.memory_limit_mb)

        self.export_b = QtGui.QPushButton('export')
        self.export_b.clicked.connect(self.export)
        self.export_fbox.addRow(self.export_b)

        # self.region_reconstruction = RegionReconstruction(project, solver=None)
        # self.vbox.addWidget(self.region_reconstruction)

        # self.fix_area = FixArea(project, solver=None)
        # self.vbox.addWidget(self.fix_area)
        # if not project.version_is_le('2.2.9'):
        #     self.fix_area.vbox.addWidget(QtGui.QLabel('AREA WAS ALREADY UPDATED!'))

    def export(self):
        print "exporting..."


        ex = Exporter(self.project.chm, self.project.gm, self.project.rm,
                      pts_export=self.include_region_points.isChecked(),
                      contour_pts_export=self.include_region_contour.isChecked())

        ex.export(self.get_out_path(), min_tracklet_length=1)
        export_arena(self.get_out_path(), self.project)

        # ftype = self.file_type.currentText()
        # if ftype == '.txt':
        #     self.export_txt()
        # elif ftype == '.csv':
        #     self.export_csv()
        # elif ftype == '.mat':
        #     self.export_mat()

        print "done"

    def write_line_csv(self, f, r):
        a, b = self.centroid_in_right_order(r)
        f.writerow([str(r.frame_), round(a, 2), round(b, 2)])

    def export_csv(self):
        chunks = self.project.chm.chunk_list()
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

    def obj_arr_append_(self, obj_arr, d):
        new_d = {}
        for key, val in d.iteritems():
            if key != 'frame' and key != 'region_id':
                val = np.array(val)

            new_d[key] = val

        obj_arr.append(d)

    def get_approx_region_size(self):
        ch_test_num = min(10, len(self.project.chm.chunks_))

        size_sum = 0
        for i in range(1, ch_test_num+1):
            rch = RegionChunk(self.project.chm[i], self.project.gm, self.project.rm)
            # so we have some idea about uncompressed pts size
            rch[0]  #!!!! BUG, ONE HAS TO ASK FOR THE SAME REGION TWICE IF THE CACHE IS CLEARED OR HAS LIMITED SIZE!
            rch[0].pts()
            size_sum += asizeof.asizeof(rch[0])

        return int(size_sum / ch_test_num)

    def export_mat(self):
        import time

        self.project.rm.cache_size_limit_ = 1;

        t = time.time()

        approx_reg_size = self.get_approx_region_size()
        print "APPROX REG SIZE", approx_reg_size

        obj_arr = []

        # bytes to Mb * 1000 * 1000
        limit = int(self.memory_limit_mb.text()) * 1000 * 1000
        curr_size = 0

        t1 = time.time()
        if not self.export_chunks_only.isChecked():
            for _, vs in self.project.gm.vertices_in_t.iteritems():
                for v in vs:
                    ch, _ = self.project.gm.is_chunk(v)

                    if not ch:
                        r = self.project.gm.region(v)
                        d = self.init_struct_(r)

                        curr_size += asizeof.asizeof(d)
                        self.add_line_mat(d, r)

                        self.obj_arr_append_(obj_arr, d)

        print "single regions t:", time.time() - t1

        t2 = time.time()
        file_num = 0
        chunNum  = 0
        for _, ch in self.project.chm.chunks_.iteritems():
            chunNum += 1

            rch = RegionChunk(ch, self.project.gm, self.project.rm)


            rch[0] #!!!! BUG, ONE HAS TO ASK FOR THE SAME REGION TWICE IF THE CACHE IS CLEARED OR HAS LIMITED SIZE!
            d = self.init_struct_(rch[0])

            #rs_ = rch[:]
            #for r in rs_:

            for regionNum in range(len(rch)):
                rch[regionNum] #!!!! BUG, ONE HAS TO ASK FOR THE SAME REGION TWICE IF THE CACHE IS CLEARED OR HAS LIMITED SIZE!
                r = rch[regionNum]
                self.add_line_mat(d, r)

            curr_size += asizeof.asizeof(d)
            self.obj_arr_append_(obj_arr, d)

            
            if (curr_size > limit):
                with open(self.get_out_path()+str(file_num)+'.mat', 'wb') as f:
                    print "saving ", str(file_num)
                    print(str(chunNum)+"\n")
                    sio.savemat(f, {'FERDA': obj_arr}, do_compression=True)

                curr_size = 0

                obj_arr = []
                #reset_selective d
                del d
                del rch
                del obj_arr
                obj_arr = []

                gc.collect()

                file_num += 1

        # save the rest
        with open(self.get_out_path()+str(file_num)+'.mat', 'wb') as f:
            sio.savemat(f, {'FERDA': obj_arr}, do_compression=True)

        print "chunks regions t:", time.time() - t2

        t3 = time.time()
        with open(self.get_out_path()+'_arena.mat', 'wb') as f:
            arena = None
            if self.project.arena_model:
                am = self.project.arena_model
                try:
                    c = am.center
                    radius = am.radius
                except AttributeError:
                    center = np.array([0, 0])
                    num = 0
                    # estimate center:
                    for y in range(am.im_height):
                        for x in range(am.im_width):
                            if am.mask_[y, x]:
                                center += np.array([y, x])
                                num += 1

                    c = center / num
                    radius = round((num / np.pi) ** 0.5)

                arena = {'cx': c[1], 'cy': c[0], 'radius': radius}

            sio.savemat(f, {'arena': arena}, do_compression=True)

        print "save t:", time.time()-t3

        print "WHOLE EXPORT t: ", time.time() - t

    def append_pts_(self, d, key, pts):
        px = []
        py = []
        for pt in pts:
            py.append(pt[0])
            px.append(pt[1])

        d[key].append({'x': np.array(px), 'y': np.array(py)})

    def add_line_mat(self, d, r):
        y, x = r.centroid()
        d['x'].append(x)
        d['y'].append(y)

        if self.include_id.isChecked():
            d['region_id'].append(r.id_)

        if self.include_orientation.isChecked():
            d['orientation'].append(r.theta_)

        if self.include_area.isChecked():
            d['area'].append(r.area())

        if self.include_axes.isChecked():
            d['major_axis'].append(r.ellipse_major_axis_length())
            d['minor_axis'].append(r.ellipse_minor_axis_length())

        if self.include_region_points.isChecked():
            pts = r.pts()
            self.append_pts_(d, 'region', pts)

        if self.include_region_contour.isChecked():
            pts = r.contour_without_holes()
            self.append_pts_(d, 'region_contour', pts)

    def init_struct_(self, region):
        d = {'x': [], 'y': [], 'frame_offset': region.frame()}
        if self.include_id.isChecked():
            d['region_id'] = []

        if self.include_orientation.isChecked():
            d['orientation'] = []

        if self.include_area.isChecked():
            d['area'] = []

        if self.include_axes.isChecked():
            d['major_axis'] = []
            d['minor_axis'] = []

        if self.include_region_points.isChecked():
            d['region'] = []

        if self.include_region_contour.isChecked():
            d['region_contour'] = []

        return d

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
        chunks = self.project.chm.chunk_list()
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

    def update_data(self, project):
        self.project = project
        if project.gm:
            self.num_of_single_nodes.setText(str(project.gm.g.num_vertices()))
        if project.chm:
            self.num_of_chunks.setText(str(len(project.chm)))

        # TODO: takes too much time... Compute statistics during project creation

        # lens_ = []
        # mean_areas_ = []
        # for ch in self.project.chm.chunk_gen():
        #     lens_.append(ch.length())
        #     areas_ = []
        #     rch = RegionChunk(ch, self.project.gm, self.project.rm)
        #
        #     for r in rch.regions_gen():
        #         areas_.append(r.area())
        #
        #     mean_areas_.append(np.mean(areas_))
        #
        #
        #
        # mean_ = np.mean(lens_)
        # mean_mean_areas_ = np.mean(mean_areas_)
        # med_ch_area = np.median(mean_areas_)
        #
        # self.mean_ch_len.setText('{:.2f}'.format(mean_))
        # self.mean_ch_area.setText('{:.2f}'.format(mean_mean_areas_))
        # self.med_ch_area.setText('{:.2f}'.format(med_ch_area))

    def show_tracklet_coverage(self):
        frames = self.project.gm.end_t - self.project.gm.start_t
        try:
            step = int(self.tracklet_coverage_step.text())
        except:
            step = 1


        import matplotlib.pyplot as plt

        vals = []
        ff = range(0, frames, step)
        for f in ff:
            vals.append(len(self.project.chm.tracklets_in_frame(f)))

        ind = np.arange(len(vals))
        ff = np.array(ff)

        width = 1.0
        fig, ax = plt.subplots()
        ax.bar(ind, np.array(vals), width, color='r')

        how_many_labels_do_we_want = 30
        labels_step = max(1, int(len(vals) / how_many_labels_do_we_want))

        ax.set_xticks(ind[::labels_step])
        ax.set_xticklabels(map(str, ff[::labels_step]))

        plt.ion()
        plt.show()