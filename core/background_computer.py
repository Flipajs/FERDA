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

class BackgroundComputer():
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
        if not os.path.exists(self.project.working_directory+'/temp/g_simplified0.pkl'):
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

                ex_str = str(sys.executable) + ' "'+os.getcwd()+'/core/parallelization.py" "'+ str(self.project.working_directory)+'" "'+str(self.project.name)+'" '+str(i)+' '+str(f_num)+' '+str(last_n_frames)
                print ex_str
                status = self.WAITING
                if i < skip_n_first_parts + self.process_n:
                    status = self.RUNNING
                    p.start(str(sys.executable) + ' "'+os.getcwd()+'/core/parallelization.py" "'+ str(self.project.working_directory)+'" "'+str(self.project.name)+'" '+str(i)+' '+str(f_num)+' '+str(last_n_frames))

                self.processes.append([p, ex_str, status])

                # self.update_callback('DONE: '+str(i+1)+' out of '+str(self.process_n))

            S_.general.log_graph_edits = True
        else:
            self.precomputed = True

    def check_parallelization(self):
        if self.finished.all() or self.precomputed:
            self.check_parallelization_timer.stop()
            self.piece_results_together()

    def piece_results_together(self):
        end_nodes_prev = []
        nodes_to_process = []

        self.solver = Solver(self.project)

        self.update_callback(0, 'Assembling solution...')

        part_num = self.part_num
        # compability...
        if self.project.version_is_le("2.0.1"):
            part_num = self.process_n

        for i in range(part_num):
            with open(self.project.working_directory+'/temp/g_simplified'+str(i)+'.pkl', 'rb') as f:
                up = pickle.Unpickler(f)
                g_ = up.load()
                start_nodes = up.load()
                end_nodes = up.load()

                # This is much faster then nx.union()
                for n, d in g_.nodes(data=True):
                    self.solver.g.add_node(n, d)

                for n1, n2, d in g_.edges(data=True):
                    self.solver.g.add_edge(n1, n2, d)

                nodes_to_process += end_nodes

                # check last and start frames...
                start_t = self.project.solver_parameters.frames_in_row * i
                for n in end_nodes_prev[:]:
                    if n.frame_ != start_t - 1:
                        end_nodes_prev.remove(n)

                for n in start_nodes[:]:
                    if n.frame_ != start_t:
                        start_nodes.remove(n)

                self.connect_graphs(self.solver.g, end_nodes_prev, start_nodes)
                end_nodes_prev = end_nodes

            self.update_callback((i+1)/float(part_num))

        S_.general.log_graph_edits = False
        self.solver.update_nodes_in_t_refs()
        self.update_callback(-1, 'Simplifying...')
        self.solver.simplify(nodes_to_process)
        self.update_callback(-1, 'CHUNKS reduction')
        self.solver.simplify_to_chunks(nodes_to_process)
        self.update_callback(-1, 'DONE...')

        S_.general.log_graph_edits = True

        self.finished_callback(self.solver)

    def connect_graphs(self, g, t1, t2):
        for r_t1 in t1:
            for r_t2 in t2:
                d = np.linalg.norm(r_t1.centroid() - r_t2.centroid())

                if d < self.solver.max_distance:
                    s, ds, multi, antlike = self.solver.assignment_score(r_t1, r_t2)
                    self.solver.add_edge_fast(r_t1, r_t2, type='d', score=-s)

    def OnProcessOutputReady(self, p_id):
        while True:
            try:
                codec = QtCore.QTextCodec.codecForName("UTF-8")
                str_ = str(codec.toUnicode(self.processes[p_id][0].readAllStandardOutput().data()))
                if p_id == self.process_n - 1:
                    try:
                        i = int(str_)
                        s = str((int(i) / float(self.frames_in_row_last)*100))
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

                self.update_callback(num_finished/float(self.part_num))

                print "PART "+str(p_id+1)+"/"+str(self.part_num)+" FINISHED MSERS, takes ", round(end - self.start, 2), " seconds which is ", round((end-self.start) / (self.process_n * self.frames_in_row * int((p_id+self.process_n)/self.process_n)), 4), " seconds per frame"

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