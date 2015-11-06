__author__ = 'fnaiser'

import numpy as np
from utils.video_manager import get_auto_video_manager
import multiprocessing as mp
from core.region.mser import get_msers_
from PyQt4 import QtGui, QtCore
import time
from functools import partial
import sys
import os
import errno
from core.settings import Settings as S_
import cPickle as pickle
import networkx as nx
from core.graph.solver import Solver
from core.graph.reduced import Reduced
import cProfile
from core.region.region_manager import RegionManager
from core.graph.chunk_manager import ChunkManager
from core.graph.chunk import Chunk
from gui.graph_view.vis_loader import VisLoader
from utils.img_manager import ImgManager


class BackgroundComputer:
    WAITING = 0
    RUNNING = 1
    FINISHED = 2

    def __init__(self, project, update_callback, finished_callback):
        self.project = project
        self.process_n = S_.parallelization.processes_num
        self.results = []
        self.update_callback = update_callback
        self.finished_callback = finished_callback
        self.start = 0

        # TODO: Settings
        self.frames_in_row = project.solver_parameters.frames_in_row
        self.frames_in_row_last = -1
        self.part_num = -1

        self.processes = []

        self.set_frames_in_row()
        self.finished = np.array([False for i in range(self.part_num)])

        self.solver = None

        self.check_parallelization_timer = QtCore.QTimer()
        self.check_parallelization_timer.timeout.connect(self.check_parallelization)
        self.check_parallelization_timer.start(100)
        self.precomputed = False

    def set_frames_in_row(self):
        vid = get_auto_video_manager(self.project)
        frame_num = int(vid.total_frame_count())

        self.part_num = int(frame_num / self.frames_in_row)
        self.frames_in_row_last = self.frames_in_row + (frame_num - (self.frames_in_row * self.part_num))

    def run(self):
        if not os.path.exists(self.project.working_directory + '/temp/part0.pkl'):
            if not S_.general.log_in_bg_computation:
                S_.general.log_graph_edits = False
            self.start = time.time()

            # change this if parallelisation stopped working and you want to run it from given part
            skip_n_first_parts = 0

            for i in range(skip_n_first_parts):
                self.processes.append(None)

            for i in range(skip_n_first_parts, self.part_num):
                p = QtCore.QProcess()

                p.finished.connect(partial(self.onFinished, i))
                p.readyReadStandardError.connect(partial(self.OnProcessErrorReady, i))
                p.readyReadStandardOutput.connect(partial(self.OnProcessOutputReady, i))

                f_num = self.frames_in_row

                last_n_frames = 0
                if i == self.part_num - 1:
                    last_n_frames = self.frames_in_row_last - self.frames_in_row

                ex_str = str(sys.executable) + ' "' + os.getcwd() + '/core/parallelization.py" "' + str(
                    self.project.working_directory) + '" "' + str(self.project.name) + '" ' + str(i) + ' ' + str(
                    f_num) + ' ' + str(last_n_frames)
                print ex_str
                status = self.WAITING
                if i < skip_n_first_parts + self.process_n:
                    status = self.RUNNING
                    p.start(str(sys.executable) + ' "' + os.getcwd() + '/core/parallelization.py" "' + str(
                        self.project.working_directory) + '" "' + str(self.project.name) + '" ' + str(i) + ' ' + str(
                        f_num) + ' ' + str(last_n_frames))

                self.processes.append([p, ex_str, status])

                # self.update_callback('DONE: '+str(i+1)+' out of '+str(self.process_n))

            S_.general.log_graph_edits = True
        else:
            self.precomputed = True

    def check_parallelization(self):
        if self.finished.all() or self.precomputed:
            self.check_parallelization_timer.stop()
            self.piece_results_together()

    def merge_parts(self, new_gm, old_g, old_g_relevant_vertices, project, old_rm, old_chm):
        """
        merges all parts (from parallelisation)
        we want to merge all these structures (graph, region and chunk managers) into one

        in the beginning there were separate graphs(for given time bounds) with ids starting from 0
        ids in region manager also starts with 0, the same for chunk manager
        -> reindexing is needed

        :param new_gm:
        :param old_g:
        :param old_g_relevant_vertices:
        :param project:
        :param old_rm:
        :param old_chm:
        :return:
        """

        new_chm = project.chm
        new_rm = project.rm

        vertex_map = {}
        used_chunks_ids = set()
        # reindex vertices
        for v_id in old_g_relevant_vertices:
            old_v = old_g.vertex(v_id)
            old_reg = old_rm[old_g.vp['region_id'][old_v]]
            new_rm.add(old_reg)

            new_v = new_gm.add_vertex(old_reg)
            vertex_map[old_v] = new_v

            used_chunks_ids.add(old_g.vp['chunk_start_id'][old_v])
            used_chunks_ids.add(old_g.vp['chunk_end_id'][old_v])

        # because 0 id means - no chunk assigned!
        used_chunks_ids.remove(0)

        # go through all edges and copy them with all edge properties...
        for old_e in old_g.edges():
            v1_old = old_e.source()
            v2_old = old_e.target()
            old_score = old_g.ep['score'][old_e]

            if v1_old in vertex_map and v2_old in vertex_map:
                v1_new = vertex_map[v1_old]
                v2_new = vertex_map[v2_old]
            else:
                # this means there was some outdated edge, it is fine to ignore it...
                continue

            # ep['score'] is assigned in add_edge call
            new_e = new_gm.add_edge(v1_new, v2_new, old_score)
            new_gm.g.ep['certainty'][new_e] = old_g.ep['certainty'][old_e]

        # chunk id = 0 means no chunk assigned
        chunks_map = {0: 0}
        # update chunks
        for old_id_ in used_chunks_ids:
            print old_id_

            if old_id_ == 79:
                print "test"

            ch = old_chm[old_id_]

            new_list = []
            for old_v in ch.nodes_:
                if old_v in vertex_map:
                    new_list.append(int(vertex_map[old_v]))
                else:
                    id_ = new_rm.add(old_rm[old_g.vp['region_id'][old_g.vertex(old_v)]])
                    # list of ids is returned [id] ...
                    id_ = id_[0]

                    # this happens in case when the vertex will not be in new graph, but we wan't to keep the region in
                    # RM (e. g. for inner points of chunks)
                    new_list.append(-id_)

            _, new_id_ = new_chm.new_chunk(new_list, new_gm)

            chunks_map[old_id_] = new_id_

        for old_v, new_v in vertex_map.iteritems():
            new_gm.g.vp['chunk_start_id'][new_v] = chunks_map[old_g.vp['chunk_start_id'][old_v]]
            new_gm.g.vp['chunk_end_id'][new_v] = chunks_map[old_g.vp['chunk_end_id'][old_v]]

    def piece_results_together(self):
        self.project.rm = RegionManager(db_wd=self.project.working_directory)
        self.project.chm = ChunkManager()
        self.solver = Solver(self.project)

        self.update_callback(0, 're-indexing...')

        # switching off... We don't want to log following...
        S_.general.log_graph_edits = False

        part_num = self.part_num
        # TODO: remove this line
        part_num = 2

        for i in range(part_num):
            rm_old = RegionManager(db_wd=self.project.working_directory + '/temp',
                                   db_name='part' + str(i) + '_rm.sqlite3')

            with open(self.project.working_directory + '/temp/part' + str(i) + '.pkl', 'rb') as f:
                up = pickle.Unpickler(f)
                g_ = up.load()
                relevant_vertices = up.load()
                chm_ = up.load()

                self.merge_parts(self.solver.gm, g_, relevant_vertices, self.project, rm_old, chm_)

            self.update_callback((i + 1) / float(part_num))
        # TODO: remove this
        self.project.solver_parameters.frames_in_row = 10

        fir = self.project.solver_parameters.frames_in_row


        self.update_callback(-1, 'joining parts...')

        self.solver.gm = self.project.gm
        for part_end_t in range(fir, fir*part_num):
            t_v = self.solver.gm.get_vertices_in_t(part_end_t-1)
            t1_v = self.solver.gm.get_vertices_in_t(part_end_t)

            self.connect_graphs(t_v, t1_v, self.project.gm, self.project.rm)
            # self.solver.simplify(map(int, t_v), [self.solver.adaptive_threshold])

        S_.general.log_graph_edits = True

        self.project.solver = self.solver

        from utils.color_manager import colorize_project
        colorize_project(self.project)

        # self.solver.save()

        self.project.gm.project = self.project
        # self.project.save()

        app = QtGui.QApplication(sys.argv)
        im_manager = ImgManager(self.projec, max_size_mb=0, max_num_of_instances=0)
        l = VisLoader(self.project)
        l.visualise()
        app.exec_()

        self.finished_callback(self.solver)

    def connect_graphs(self, vertices1, vertices2, gm, rm):
        print "edges before ", gm.g.num_edges()
        for v1 in vertices1:
            r1 = rm[int(v1)]
            for v2 in vertices2:
                r2 = rm[int(v2)]

                d = np.linalg.norm(r1.centroid() - r2.centroid())

                if d < gm.max_distance:
                    s, ds, multi, antlike = self.solver.assignment_score(r1, r2)
                    gm.add_edge_fast(v1, v2, s)

        print "edges after ", gm.g.num_edges()

    def OnProcessOutputReady(self, p_id):
        while True:
            try:
                codec = QtCore.QTextCodec.codecForName("UTF-8")
                str_ = str(codec.toUnicode(self.processes[p_id][0].readAllStandardOutput().data()))
                if p_id == self.process_n - 1:
                    try:
                        i = int(str_)
                        s = str((int(i) / float(self.frames_in_row_last) * 100))
                        # self.update_callback(' '+s[0:4]+'%')
                    except:
                        print str_
                break
            except IOError, e:
                if e.errno != errno.EINTR:
                    raise

    def OnProcessErrorReady(self, p_id):
        codec = QtCore.QTextCodec.codecForName("UTF-8")
        print p_id, codec.toUnicode(self.processes[p_id][0].readAllStandardError().data())

    def onFinished(self, p_id):
        while True:
            try:
                end = time.time()
                self.finished[p_id] = True
                num_finished = 0
                for i in self.finished:
                    if i:
                        num_finished += 1

                self.update_callback(num_finished / float(self.part_num))

                print "PART " + str(p_id + 1) + "/" + str(self.part_num) + " FINISHED MSERS, takes ", round(
                    end - self.start, 2), " seconds which is ", round((end - self.start) / (
                    self.process_n * self.frames_in_row * int((p_id + self.process_n) / self.process_n)),
                                                                      4), " seconds per frame"

                self.processes[p_id][2] = self.FINISHED

                new_id = p_id + self.process_n
                if new_id < len(self.processes):
                    it = self.processes[new_id]
                    it[0].start(it[1])
                    self.processes[new_id][2] = self.RUNNING

                break
            except IOError, e:
                if e.errno != errno.EINTR:
                    raise
