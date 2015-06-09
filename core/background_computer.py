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

class BackgroundComputer():
    def __init__(self, project, update_callback, finished_callback):
        self.project = project
        self.process_n = S_.parallelization.processes_num
        self.results = []
        self.update_callback = update_callback
        self.finished_callback = finished_callback
        self.processes = []
        self.start = 0
        self.frames_in_row = -1
        self.frames_in_row_last = -1
        self.set_frames_in_row()
        self.finished = np.array([False for i in range(self.process_n)])

        self.solver = None

        self.check_parallelization_timer = QtCore.QTimer()
        self.check_parallelization_timer.timeout.connect(self.check_parallelization)
        self.check_parallelization_timer.start(100)


    def set_frames_in_row(self):
        vid = get_auto_video_manager(self.project.video_paths)
        frame_num = int(vid.total_frame_count())

        self.frames_in_row = int(frame_num / self.process_n)
        self.frames_in_row_last = self.frames_in_row + (frame_num - (self.frames_in_row * self.process_n))

    def run(self):
        if not os.path.exists(self.project.working_directory+'/temp/g_simplified0.pkl'):
            self.start = time.time()
            for i in range(self.process_n):
                p = QtCore.QProcess()

                p.finished.connect(partial(self.onFinished, i))
                p.readyReadStandardError.connect(partial(self.OnProcessErrorReady, i))
                p.readyReadStandardOutput.connect(partial(self.OnProcessOutputReady, i))

                f_num = self.frames_in_row
                last_n_frames = 0
                if i == self.process_n - 1:
                    last_n_frames = self.frames_in_row_last - self.frames_in_row

                p.start(str(sys.executable) + ' "'+os.getcwd()+'/core/parallelization.py" "'+ str(self.project.working_directory)+'" "'+str(self.project.name)+'" '+str(i)+' '+str(f_num)+' '+str(last_n_frames))
                print str(sys.executable) + ' "'+os.getcwd()+'/core/parallelization.py" "'+ str(self.project.working_directory)+'" "'+str(self.project.name)+'" '+str(i)+' '+str(f_num)+' '+str(last_n_frames)
                self.processes.append(p)

                self.update_callback('DONE: '+str(i+1)+' out of '+str(self.process_n))
        else:
            self.piece_results_together()
            self.check_parallelization_timer.stop()

    def check_parallelization(self):
        if self.finished.all():
            self.check_parallelization_timer.stop()
            self.piece_results_together()
            self.update_callback(' MSER COMPUTATION FINISHED, preparing data for correction')

    def piece_results_together(self):
        end_nodes_prev = []
        nodes_to_process = []

        self.solver = Solver(self.project)

        for i in range(S_.parallelization.processes_num):
            with open(self.project.working_directory+'/temp/g_simplified'+str(i)+'.pkl', 'rb') as f:
                up = pickle.Unpickler(f)
                g_ = up.load()
                start_nodes = up.load()
                end_nodes = up.load()

                self.solver.g = nx.union(self.solver.g, g_)

                if i < self.process_n - 1:
                    nodes_to_process += end_nodes

                self.connect_graphs(self.solver.g, end_nodes_prev, start_nodes)
                end_nodes_prev = end_nodes

        self.solver.update_nodes_in_t_refs()
        self.solver.simplify(nodes_to_process)
        self.solver.simplify_to_chunks()

        self.finished_callback(self.solver)

    def connect_graphs(self, g, t1, t2):
        for r_t1 in t1:
            for r_t2 in t2:
                d = np.linalg.norm(r_t1.centroid() - r_t2.centroid())

                if d < self.solver.max_distance:
                    s, ds, multi, antlike = self.solver.assignment_score(r_t1, r_t2)
                    g.add_edge(r_t1, r_t2, type='d', score=-s)


    def OnProcessOutputReady(self, p_id):
        while True:
            try:
                codec = QtCore.QTextCodec.codecForName("UTF-8")
                str_ = str(codec.toUnicode(self.processes[p_id].readAllStandardOutput().data()))
                if p_id == self.process_n - 1:
                    try:
                        i = int(str_)
                        s = str((int(i) / float(self.frames_in_row_last)*100))
                        self.update_callback(' '+s[0:4]+'%')
                    except:
                        print str_
                break
            except IOError, e:
                if e.errno != errno.EINTR:
                    raise

    def OnProcessErrorReady(self, p_id):
        codec = QtCore.QTextCodec.codecForName("UTF-8")
        print p_id, codec.toUnicode(self.processes[p_id].readAllStandardError().data())

    def onFinished(self, p_id):
        while True:
            try:
                end = time.time()
                self.finished[p_id] = True
                print "FINISHED MSERS, takes ", end - self.start, " seconds which is ", (end-self.start) / (self.process_n * self.frames_in_row), " seconds per frame"

                break
            except IOError, e:
                if e.errno != errno.EINTR:
                    raise