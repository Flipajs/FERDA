from fitting_thread import FittingThread, FittingThreadChunk
from copy import deepcopy
from PyQt4.QtCore import QProcess
import sys, os
from functools import partial
from PyQt4 import QtCore
import cPickle as pickle
from core.region.fitting import Fitting


class FittingSession:
    def __init__(self, id, fitting_thread, pivot, callback):
        self.id = id
        self.locked_vertices = []
        self.fp = fitting_thread
        self.pivot = pivot
        self.callback = callback


class FittingSessionChunk(FittingSession):
    def __init__(self, id, project, step_callback, model, regions, ch_vertices, vertices_after_chunk):
        # super(self.__class__, self).__init__(id, None)
        self.id = id
        self.locked_vertices = []
        self.fp = []

        self.project = project
        self.step_callback = step_callback
        self.model = model
        self.ch_regions = regions
        self.ch_vertices = ch_vertices
        self.vertices_after_chunk = vertices_after_chunk

    def get_file_name(self):
        return self.project.working_directory+'/temp/sess_data_'+str(self.id)+'.pkl'

    def on_process_error_ready(self, s_id):
        print s_id, self.fp[-1].readAllStandardError().data()

    def start(self):
        merged = self.ch_regions[0]
        self.pivot = self.ch_vertices[0]
        self.ch_regions = self.ch_regions[1:]
        self.ch_vertices = self.ch_vertices[1:]

        model = deepcopy(self.model)
        for m in model: m.frame_ += 1


        fp = QProcess()

        fp.finished.connect(partial(self.process_next))
        fp.readyReadStandardError.connect(partial(self.on_process_error_ready, self.id))

        file_name = self.get_file_name()

        f = Fitting(merged, model)
        with open(file_name, 'wb') as f_:
            pickle.dump({'fitting': f}, f_, -1)

        ex_str = str(sys.executable) + ' "' + os.getcwd() + '/core/region/fitting_script_chunk.py" '+str(self.id)+' "'+file_name+'"'

        self.fp.append(fp)
        self.fp[-1].start(ex_str)

    def process_next(self):
        file_name = self.get_file_name()

        pivot = self.pivot
        s_id = self.id

        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        result = data['results']
        fitting = data['fitting']

        if len(self.ch_regions) == 0:
            # reconnect the end...
            # self.project.gm.add_edges_()

            os.remove(file_name)
            self.step_callback(result, pivot, s_id, None)
        else:
            # -s_id is a flag - do not release this session

            # it is important to call deepcopy before merged_chunk, where pts_ are rounded...
            model = deepcopy(result)
            self.step_callback(result, pivot, -s_id, None)

            merged = self.ch_regions[0]
            self.pivot = self.ch_vertices[0]
            self.ch_regions = self.ch_regions[1:]
            self.ch_vertices = self.ch_vertices[1:]

            fitting.region = merged
            from core.region.distance_map import DistanceMap
            fitting.d_map_region = DistanceMap(merged.pts())

            for a in fitting.animals:
                a.frame_ += 1

            fp = QProcess()

            fp.finished.connect(partial(self.process_next))
            fp.readyReadStandardError.connect(partial(self.on_process_error_ready, s_id))

            with open(file_name, 'wb') as f_:
                pickle.dump({'fitting': fitting}, f_, -1)

            ex_str = str(sys.executable) + ' "' + os.getcwd() + '/core/region/fitting_script_chunk.py" '+str(self.id)+' "'+file_name+'"'

            self.fp.append(fp)
            self.fp[-1].start(ex_str)


class FittingThreadingManager:
    def __init__(self):
        self.locked_vertices = set()
        self.fitting_sessions = {}
        self.session_id = 1

    def get_file_name(self, project, s_id):
        return project.working_directory+'/temp/sess_data_'+str(s_id)+'.pkl'

    def add_simple_session(self, done_callback, region, model, pivot, project):
        s_id = self.session_id
        self.session_id += 1

        fp = QProcess()

        fp.finished.connect(partial(self.on_finished, s_id, project))
        fp.readyReadStandardError.connect(partial(self.on_process_error_ready, s_id))

        file_name = self.get_file_name(project, s_id)

        with open(file_name, 'wb') as f:
            pickle.dump({'merged': region, 'model': model}, f, -1)

        ex_str = str(sys.executable) + ' "' + os.getcwd() + '/core/region/fitting_script.py" '+str(s_id)+' "'+file_name+'"'

        fs = FittingSession(s_id, fp, pivot, done_callback)
        fs.locked_vertices.extend(list(set(pivot.in_neighbours())))
        fs.locked_vertices.append(pivot)
        fs.locked_vertices.extend(list(set(pivot.out_neighbours())))

        self.fitting_sessions[s_id] = fs
        for v in fs.locked_vertices:
            self.locked_vertices.add(int(v))

        fs.fp.start(ex_str)

        print "STARTING ", s_id

    def on_finished(self, s_id, project):
        file_path = self.get_file_name(project, s_id)
        with open(file_path, 'rb') as f:
            results = pickle.load(f)

        os.remove(file_path)

        pivot = self.fitting_sessions[s_id].pivot
        self.fitting_sessions[s_id].callback(results, pivot, s_id, None)

    def on_process_error_ready(self, s_id):
        print s_id, self.fitting_sessions[s_id].fp.readAllStandardError().data()

    def release_session(self, s_id):
        fs = self.fitting_sessions[s_id]
        for v in fs.locked_vertices:
            if int(v) in self.locked_vertices:
                self.locked_vertices.remove(int(v))

        print "RELEASING: ", s_id
        del self.fitting_sessions[s_id]

    def add_chunk_session(self, project, done_callback, chunk):
        ch_s = project.gm.g.vertex(chunk.start_vertex_id())
        ch_e = project.gm.g.vertex(chunk.end_vertex_id())

        vertices_before_chunk = list(set([v for v in ch_s.in_neighbours()]))
        vertices_after_chunk = list(set([v for v in ch_e.out_neighbours()]))

        chunk_vertices = []

        i = 0
        while chunk.length() > 0 and (i < 20 or chunk.length() < 3):
            chunk_vertices.append(chunk.pop_first(project.gm))
            i += 1

        s_id = self.session_id
        self.session_id += 1

        regions_before_chunk = map(project.gm.region, vertices_before_chunk)
        chunk_regions = map(project.gm.region, chunk_vertices)

        fs = FittingSessionChunk(s_id, project, done_callback, regions_before_chunk, chunk_regions, chunk_vertices, vertices_after_chunk)
        fs.locked_vertices.extend(vertices_before_chunk)
        fs.locked_vertices.extend(chunk_vertices)
        fs.locked_vertices.extend(vertices_after_chunk)

        for v in fs.locked_vertices:
            self.locked_vertices.add(int(v))

        self.fitting_sessions[s_id] = fs
        fs.start()

        print "STARTING ", s_id

    def add_lock(self, s_id, vertices):
        vertices = map(int, vertices)
        self.fitting_sessions[s_id].locked_vertices.extend(vertices)
        for v in vertices:
            self.locked_vertices.add(v)
