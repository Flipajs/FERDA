__author__ = 'fnaiser'

import errno
import sys
import time
from functools import partial

import numpy as np
import os
from PyQt4 import QtCore

from graph_assembly import graph_assembly
from core.config import config
from utils.video_manager import get_auto_video_manager
from core.graph.solver import Solver


class BackgroundComputer:
    WAITING = 0
    RUNNING = 1
    FINISHED = 2

    def __init__(self, project, update_callback, finished_callback):
        self.project = project
        self.process_n = config['general']['n_jobs']
        self.results = []
        self.update_callback = update_callback
        self.finished_callback = finished_callback
        self.start = []

        # TODO: Settings
        self.frames_in_row = config['segmentation']['frames_in_row']
        self.frames_in_row_last = -1
        self.n_parts = -1

        self.processes = []

        vid = get_auto_video_manager(self.project)
        self.frame_num = int(vid.total_frame_count())
        self.finished = np.array([False for i in range(self.n_parts)])

        self.solver = None

        self.check_parallelization_timer = QtCore.QTimer()
        self.check_parallelization_timer.timeout.connect(self.check_parallelization)
        self.check_parallelization_timer.start(100)
        self.precomputed = False

        self.first_part = 0
        # is True when semi merge is done on cluster... e.g. parts 0-9, 10-19 etc...
        self.do_semi_merge = False

    def run(self):
        if not os.path.exists(self.project.working_directory + '/temp'):
            os.mkdir(self.project.working_directory + '/temp')
            
        if not os.path.exists(self.project.working_directory + '/temp/part0.pkl'):

            if not config['general']['log_in_bg_computation']:
                config['general']['log_graph_edits'] = False

            self.n_parts = len(range(0, self.frame_num, config['segmentation']['frames_in_row']))
            self.start = [0] * self.n_parts

            for i, frame_start in enumerate(range(0, self.frame_num, config['segmentation']['frames_in_row'])):
                p = QtCore.QProcess()

                p.finished.connect(partial(self.onFinished, i))
                p.readyReadStandardError.connect(partial(self.OnProcessErrorReady, i))
                p.readyReadStandardOutput.connect(partial(self.OnProcessOutputReady, i))

                commandline = '{executable} "{cwd}/core/segmentation.py" ' \
                              'part_segmentation "{project_file}" {part_id} {frame_start}'.format(
                    executable=sys.executable,
                    cwd=os.getcwd(),
                    project_file=self.project.project_file,
                    part_id=i,
                    frame_start=frame_start,
                    )
                print(commandline)

                if i < self.process_n:
                    p.start(commandline)

                self.start[i] = time.time()

                if i < self.process_n:
                    status = self.RUNNING
                else:
                    status = self.WAITING

                self.processes.append([p, commandline, status])

                # self.update_callback('DONE: '+str(i+1)+' out of '+str(self.process_n))

            config['general']['log_graph_edits'] = True
        else:
            self.precomputed = True

    def check_parallelization(self):
        if self.finished.all() or self.precomputed:
            self.check_parallelization_timer.stop()
            self.project.load(self.project.working_directory)
            self.solver = Solver(self.project)
            graph_assembly(self.project, self.solver, self.do_semi_merge)
            self.project.region_cardinality_classifier.classify_project(self.project)

            # from utils.color_manager import colorize_project
            # import time
            # s = time.time()
            # colorize_project(self.project)
            # print "color manager took %f seconds" % (time.time() - s)

    def OnProcessOutputReady(self, p_id):
        while True:
            try:
                if p_id == self.process_n - 1:
                    codec = QtCore.QTextCodec.codecForName("UTF-8")
                    str_ = str(codec.toUnicode(self.processes[p_id][0].readAllStandardOutput().data()))

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

                self.update_callback(num_finished / float(self.n_parts))

                print "PART " + str(p_id + 1) + "/" + str(self.n_parts) + " FINISHED MSERS, takes ", round(
                    end - self.start[p_id], 2), " seconds which is ", round((end - self.start[p_id]) / (
                    self.process_n * self.frames_in_row), 4), " seconds per frame"

                self.processes[p_id][2] = self.FINISHED

                new_id = p_id + self.process_n
                if new_id < len(self.processes):
                    it = self.processes[new_id]
                    it[0].start(it[1])
                    self.start[new_id] = time.time()
                    self.processes[new_id][2] = self.RUNNING

                break
            except IOError, e:
                if e.errno != errno.EINTR:
                    raise
