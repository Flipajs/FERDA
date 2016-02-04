from fitting_thread import FittingThread, FittingThreadChunk
from copy import deepcopy


class FittingSession:
    def __init__(self, id, fitting_thread):
        self.id = id
        self.locked_vertices = []
        self.ft = fitting_thread


class FittingSessionChunk(FittingSession):
    def __init__(self, id, project, step_callback, model, regions, ch_vertices, vertices_after_chunk):
        # super(self.__class__, self).__init__(id, None)
        self.id = id
        self.locked_vertices = []
        self.ft = []

        self.project = project
        self.step_callback = step_callback
        self.model = model
        self.ch_regions = regions
        self.ch_vertices = ch_vertices
        self.vertices_after_chunk = vertices_after_chunk

    def start(self):
        merged = self.ch_regions[0]
        pivot = self.ch_vertices[0]
        self.ch_regions = self.ch_regions[1:]
        self.ch_vertices = self.ch_vertices[1:]

        model = deepcopy(self.model)
        for m in model: m.frame_ += 1

        ft = FittingThread(merged, model, pivot, self.id)
        self.ft.append(ft)
        self.ft[-1].proc_done.connect(self.process_next)
        self.ft[-1].start()

    def process_next(self, result, pivot, s_id, fitting):
        if len(self.ch_regions) == 0:
            # reconnect the end...
            # self.project.gm.add_edges_()

            self.step_callback(result, pivot, s_id, None)
        else:
            # -s_id is a flag - do not release this session

            # it is important to call deepcopy before merged_chunk, where pts_ are rounded...
            model = deepcopy(result)
            self.step_callback(result, pivot, -s_id, None)

            merged = self.ch_regions[0]
            pivot = self.ch_vertices[0]
            self.ch_regions = self.ch_regions[1:]
            self.ch_vertices = self.ch_vertices[1:]

            fitting.region = merged
            from core.region.distance_map import DistanceMap
            fitting.d_map_region = DistanceMap(merged.pts())

            for a in fitting.animals:
                a.frame_ += 1

            ft = FittingThreadChunk(pivot, self.id, fitting)
            # ft = FittingThread(merged, model, pivot, self.id)
            self.ft.append(ft)
            self.ft[-1].proc_done.connect(self.process_next)
            self.ft[-1].start()


class FittingThreadingManager():
    def __init__(self):
        self.locked_vertices = set()
        self.fitting_sessions = {}
        self.session_id = 1

    def add_simple_session(self, done_callback, region, model, pivot):
        s_id = self.session_id
        self.session_id += 1

        fs = FittingSession(s_id, FittingThread(region, model, pivot, s_id))
        fs.locked_vertices.extend(list(set(pivot.in_neighbours())))
        fs.locked_vertices.append(pivot)
        fs.locked_vertices.extend(list(set(pivot.out_neighbours())))

        self.fitting_sessions[s_id] = fs
        for v in fs.locked_vertices:
            self.locked_vertices.add(int(v))

        fs.ft.proc_done.connect(done_callback)
        fs.ft.start()

        print "STARTING ", s_id

    def release_session(self, s_id):
        fs = self.fitting_sessions[s_id]
        for v in fs.locked_vertices:
            self.locked_vertices.remove(int(v))

        print "RELEASING: ", s_id
        del self.fitting_sessions[s_id]

    def add_chunk_session(self, project, done_callback, chunk):
        ch_s = project.gm.g.vertex(chunk.start_vertex_id())
        ch_e = project.gm.g.vertex(chunk.end_vertex_id())

        vertices_before_chunk = list(set([v for v in ch_s.in_neighbours()]))
        vertices_after_chunk = list(set([v for v in ch_e.out_neighbours()]))

        chunk_vertices = []
        while chunk.length() > 0:
            chunk_vertices.append(chunk.pop_first(project.gm))




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


# def fitting_get_model(project, pivot):
#     region = project.gm.region(active_cw.active_node)
#
#     merged_t = region.frame() - active_cw.frame_t
#     model_t = merged_t - 1
#
#     objects = []
#     vertices = []
#
#     if len(active_cw.vertices_groups[model_t]) > 0 and len(active_cw.vertices_groups[merged_t]) > 0:
#         t1_ = active_cw.vertices_groups[model_t]
#
#         for c1 in t1_:
#             vertices.append(c1)
#             a = deepcopy(project.gm.region(c1))
#             project.rm.add(a)
#
#             a.frame_ += 1
#
#             objects.append(a)
#
#     return objects, vertices




