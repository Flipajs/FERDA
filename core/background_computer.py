from PyQt4.QtCore import QObject, pyqtSignal
import numpy as np
from PyQt4 import QtGui, QtCore
import time
from functools import partial
import sys
import os
import errno
from core.settings import Settings as S_
from utils.video_manager import get_auto_video_manager
from bg_computer_assembling import assembly_after_parallelization

__author__ = 'fnaiser'


class BackgroundComputer(QObject):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2
    next_step_progress_signal = pyqtSignal(int, str)
    update_progress_signal = pyqtSignal()

    def __init__(self, project, finished_callback, postpone_parallelisation):
        super(BackgroundComputer, self).__init__()
        self.project = project
        self.process_n = S_.parallelization.processes_num
        self.results = []
        self.finished_callback = finished_callback
        self.start = []

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
        self.postpone_parallelisation = postpone_parallelisation

        self.first_part = 0
        # is True when semi merge is done on cluster... e.g. parts 0-9, 10-19 etc...
        self.do_semi_merge = False

        # self.bg_computer_signal = QtCore.pyqtSignal(int, name="bg_computer_signal")

    def set_frames_in_row(self):
        vid = get_auto_video_manager(self.project)
        frame_num = int(vid.total_frame_count())

        self.part_num = int(int(frame_num) / int(self.frames_in_row))
        self.frames_in_row_last = self.frames_in_row + (frame_num - (self.frames_in_row * self.part_num))

    def run(self):
        if not os.path.exists(self.project.working_directory + '/temp'):
            os.mkdir(self.project.working_directory + '/temp')

        if not os.path.exists(self.project.working_directory + '/temp/part0.pkl'):
            # if self.postpone_parallelisation:
                # f = open(self.project.working_directory+'/limits.txt', 'w')

            if not S_.general.log_in_bg_computation:
                S_.general.log_graph_edits = False

            # change this if parallelisation stopped working and you want to run it from given part
            skip_n_first_parts = 0
            # TODO: also update progress_bar

            self.start = [0] * self.part_num

            for i in range(skip_n_first_parts):
                self.processes.append(None)

            if self.postpone_parallelisation:
                limitsFile = open(str(self.project.working_directory)+"/limits.txt","w")

            self.next_step_progress_signal.emit(self.part_num + 1, "Computing MSERs")
            self.update_progress_signal.emit()

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
                # print ex_str

                if self.postpone_parallelisation:
                    limitsFile.write(str(i)+'\t'+str(f_num)+'\t'+str(last_n_frames)+'\n')

                status = self.WAITING
                if i < skip_n_first_parts + self.process_n:
                    status = self.RUNNING

                    if not self.postpone_parallelisation:
                        p.start(str(sys.executable) + ' "' + os.getcwd() + '/core/parallelization.py" "' + str(
                            self.project.working_directory) + '" "' + str(self.project.name) + '" ' + str(i) + ' ' + str(
                            f_num) + ' ' + str(last_n_frames))

                self.start[i] = time.time()

                status = self.WAITING
                if i < skip_n_first_parts + self.process_n:
                    status = self.RUNNING

                self.processes.append([p, ex_str, status])

            if self.postpone_parallelisation:
                self.precomputed = True

            S_.general.log_graph_edits = True
            if self.postpone_parallelisation:
                limitsFile.close()
                sys.exit()  # Comment for cluster usage
            
        else:
            self.precomputed = True

    def check_parallelization(self):
        if not self.postpone_parallelisation and (self.finished.all() or self.precomputed):
            self.check_parallelization_timer.stop()
            self.project.load(self.project.working_directory+'/'+self.project.name+'.fproj')
            assembly_after_parallelization(self)

            from utils.color_manager import colorize_project
            import time
            s = time.time()
            colorize_project(self.project)
            print "color manager takes %f seconds" % (time.time() - s)
            self.finished_callback(self.solver)

    def OnProcessOutputReady(self, p_id):
        while True:
            try:
                if p_id == self.process_n - 1:
                    codec = QtCore.QTextCodec.codecForName("UTF-8")
                    str_ = str(codec.toUnicode(self.processes[p_id][0].readAllStandardOutput().data()))

                    try:
                        i = int(str_)
                        s = str((int(i) / float(self.frames_in_row_last) * 100))
                    except:
                        # TODO: this causes unusual outputs
                        # print str_
                        pass
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

                self.update_progress_signal.emit()

                print "PART " + str(p_id + 1) + "/" + str(self.part_num) + " FINISHED MSERS, takes ",\
                    round(end - self.start[p_id], 2), " seconds which is ",\
                    round((end - self.start[p_id]) / (self.process_n * self.frames_in_row), 4), " seconds per frame"

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
