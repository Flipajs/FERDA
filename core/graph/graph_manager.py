__author__ = 'flipajs'

import graph_tool
from core.log import LogCategories, ActionNames
import numpy as np
from scipy.spatial.distance import cdist
from core.region.fitting_logger import FittingLogger


class GraphManager:
    def __init__(self, project, assignment_score):
        self.project = project
        self.rm = project.rm
        self.g = graph_tool.Graph(directed=True)
        self.g.set_fast_edge_removal(fast=True)
        self.graph_add_properties()
        self.vertices_in_t = {}
        self.start_t = np.inf
        self.end_t = -1
        # multiply 2 times to get ant length
        self.major_axis_median = project.stats.major_axis_median
        self.max_distance = project.solver_parameters.max_edge_distance_in_ant_length * self.major_axis_median
        self.assignment_score = assignment_score
        self.fitting_logger = FittingLogger()

    def graph_add_properties(self):
        # In these cases the id 0 means unassigned
        # thus it is important to start indexing from 1 in region and chunk manager
        self.g.vp['region_id'] = self.g.new_vertex_property("int")
        self.g.vp['active'] = self.g.new_vertex_property("bool")
        self.g.vp['chunk_start_id'] = self.g.new_vertex_property("int")
        self.g.vp['chunk_end_id'] = self.g.new_vertex_property("int")

        self.g.ep['score'] = self.g.new_edge_property("float")
        self.g.ep['certainty'] = self.g.new_edge_property("float")

    def add_vertex(self, region):
        self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.ADD_NODE, region)
        self.start_t = min(self.start_t, region.frame_)
        self.end_t = max(self.end_t, region.frame_)

        vertex = self.g.add_vertex()

        self.vertices_in_t.setdefault(region.frame(), []).append(int(vertex))
        self.g.vp['region_id'][vertex] = region.id()
        self.g.vp['active'][vertex] = True

        return vertex

    def add_vertices(self, regions):
        # v_list = self.g.add_vertex(len(regions))
        for region in regions:
            vertex = self.add_vertex(region)
            # self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.ADD_NODE, region)
            # self.start_t = min(self.start_t, region.frame_)
            # self.end_t = max(self.end_t, region.frame_)
            #
            # self.vertices_in_t.setdefault(region.frame_, []).append(int(vertex))
            # self.g.vp['region_id'][vertex] = region.id()

    def remove_vertex(self, vertex_id, disassembly=True):
        if vertex_id < 0:
            print "id < 0"
        vertex = self.g.vertex(vertex_id)
        region = self.project.rm[self.g.vp['region_id'][vertex]]

        if region.frame() not in self.vertices_in_t:
            return []

        if int(vertex) not in self.vertices_in_t[region.frame()]:
            return []

        affected = []

        if region is None:
            print "remove node n is None"
            return

        if disassembly:
            ch, chunk_end = self.is_chunk(vertex)
            if ch:
                ch.pop_last(self) if chunk_end else ch.pop_first(self)

        # save all edges
        for e in vertex.in_edges():
            affected.append(e.source())
            self.remove_edge_(e)

        for e in vertex.out_edges():
            affected.append(e.target())
            self.remove_edge_(e)

        self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.REMOVE_NODE, region)

        self.vertices_in_t[region.frame()].remove(int(vertex))
        if not self.vertices_in_t[region.frame_]:
            del self.vertices_in_t[region.frame_]

        # do not remove the vertex it is very slow O(N)
        # if using fast=True, the node id will be corrupted...
        # self.g.remove_vertex(vertex)

        self.g.vp['chunk_start_id'][vertex] = 0
        self.g.vp['chunk_end_id'][vertex] = 0
        self.g.vp['active'][vertex] = False

        # maybe we need to shrink time boundaries...
        if self.end_t == region.frame_ or self.start_t == region.frame_:
            self.update_time_boundaries()

        return affected

    def is_chunk(self, vertex):
        """
        check whether the vertex is start or end of chunk
        :param n: ref to vertex in g
        :return: (chunk_ref (ref or None), is_chunk_end (True if it is chunk_end))
        """
        if isinstance(vertex, int):
            vertex = self.g.vertex(vertex)

        chunk_start = self.g.vp['chunk_start_id'][vertex]
        # 0 means not a chunk
        if chunk_start:
            return self.project.chm[chunk_start], False

        chunk_end = self.g.vp['chunk_end_id'][vertex]
        if chunk_end:
            return self.project.chm[chunk_end], True

        return None, False

    def update_nodes_in_t_refs(self):
        self.vertices_in_t = {}
        for v in self.g.vertices():
            if self.g.vp['active'][v]:
                v_id = int(v)
                n = self.region(v_id)
                self.vertices_in_t.setdefault(n.frame_, []).append(v_id)

        self.update_time_boundaries()

    def add_regions_in_t(self, regions, t, fast=False):
        self.add_vertices(regions)

        if t-1 in self.vertices_in_t and t in self.vertices_in_t:
            self.add_edges_(self.vertices_in_t[t-1], self.vertices_in_t[t], fast=fast)

    def region(self, vertex):
        if int(vertex) < 0:
            id_ = -int(vertex)
        else:
            id_ = self.g.vp['region_id'][self.g.vertex(vertex)]

        return self.project.rm[id_]

    def chunk_start(self, vertex_id):
        id_ = self.g.vp['chunk_start_id'][self.g.vertex(vertex_id)]
        return self.project.chm[id_]

    def chunk_end(self, vertex_id):
        id_ = self.g.vp['chunk_end_id'][self.g.vertex(vertex_id)]
        return self.project.chm[id_]

    def add_edges_(self, vertices_t1, vertices_t2, fast=False):
        if vertices_t1 and vertices_t2:
            regions_t1 = [self.region(v) for v in vertices_t1]
            regions_t2 = [self.region(v) for v in vertices_t2]

            centroids_t1 = np.array([r.centroid() for r in regions_t1])
            centroids_t2 = np.array([r.centroid() for r in regions_t2])

            dists = cdist(centroids_t1, centroids_t2)
            for i, v_t1, r_t1 in zip(range(len(vertices_t1)), vertices_t1, regions_t1):
                for j, v_t2, r_t2 in zip(range(len(vertices_t2)), vertices_t2, regions_t2):
                    d = dists[i, j]

                    if d < self.max_distance:
                        if self.chunk_start(v_t1) or self.chunk_end(v_t2):
                            continue

                        s, ds, multi, _ = self.assignment_score(r_t1, r_t2)

                        if fast:
                            self.add_edge_fast(v_t1, v_t2, s)
                        else:
                            self.add_edge(v_t1, v_t2, s)

    def update_time_boundaries(self):
        self.start_t = np.inf
        self.end_t = -1

        for frame in self.vertices_in_t:
            self.start_t = min(self.start_t, frame)
            self.end_t = max(self.end_t, frame)

    def remove_edge(self, source_vertex, target_vertex):
        # source_vertex = self.match_if_reconstructed(source_vertex)
        # target_vertex = self.match_if_reconstructed(target_vertex)

        if source_vertex is None or target_vertex is None:
            if source_vertex is None:
                print "ERROR in (graph_manager.py) remove_edge source_vertex is None, target_vertex: ", target_vertex
            else:
                print "ERROR in (graph_manager.py) remove_edge target_vertex is None, source_vertex: ", source_vertex
            return

        for e in source_vertex.out_edges():
            if e.target == target_vertex:
                self.remove_edge_(e)

    def remove_edge_(self, edge):
        s = self.g.ep['score'][edge]

        self.project.log.add(LogCategories.GRAPH_EDIT,
                             ActionNames.REMOVE_EDGE,
                             {'v1': edge.source(),
                              'v2': edge.target(),
                              's': s})

        self.g.remove_edge(edge)

    def add_edge(self, source_vertex, target_vertex, score=1):
        # source_vertex = self.match_if_reconstructed(source_vertex)
        # target_vertex = self.match_if_reconstructed(target_vertex)
        if source_vertex is None or target_vertex is None:
            if source_vertex is None:
                print "add_edge source_vertex is None, target_vertex: ", target_vertex
            else:
                print "add_edge target_vertex is None, source_vertex: ", source_vertex
            return

        return self.add_edge_fast(source_vertex, target_vertex, score)

    def add_edge_fast(self, source_vertex, target_vertex, score):
        self.project.log.add(LogCategories.GRAPH_EDIT,
                             ActionNames.ADD_EDGE,
                             {'v1': source_vertex,
                              'v2': target_vertex,
                              's': score})

        e = self.g.add_edge(source_vertex, target_vertex)
        self.g.ep['score'][e] = float(score)
        return e

    def chunk_list(self):
        chunks = []
        for v in self.g.vertices():
            ch = self.g.vp['chunk_start_id'][v]
            if ch:
                chunks.append(ch)

        return chunks

    def chunks_in_frame(self, frame):
        chunks = self.chunk_list()

        in_frame = []
        for ch in chunks:
            if ch.start_t() <= frame <= ch.end_t():
                in_frame.append(ch)

        return in_frame

    def start_nodes(self):
        return self.vertices_in_t[self.start_t]

    def end_nodes(self):
        return self.vertices_in_t[self.end_t]

    def get_all_relevant_vertices(self):
        vertices = []
        for _, vs_ in self.vertices_in_t.iteritems():
            vertices.extend(vs_)

        return vertices

    def get_2_best_out_vertices(self, vertex, order='desc'):
        vertices = []
        scores = []

        for e in vertex.out_edges():
            vertices.append(e.target())
            scores.append(self.g.ep['score'][e])

        scores = np.array(scores)
        vertices = np.array(vertices)

        if order == 'asc':
            ids = np.argsort(scores)
        else:
            ids = np.argsort(-scores)

        r = min(2, len(ids))
        best_scores = [0, 0]
        best_vertices = [None, None]
        for i in range(r):
            best_scores[i] = scores[ids[i]]
            best_vertices[i] = vertices[ids[i]]

        return best_scores, best_vertices

    def get_2_best_in_vertices(self, vertex, order='desc'):
        vertices = []
        scores = []

        for e in vertex.in_edges():
            vertices.append(e.source())
            scores.append(self.g.ep['score'][e])

        scores = np.array(scores)
        vertices = np.array(vertices)

        if order == 'asc':
            ids = np.argsort(scores)
        else:
            ids = np.argsort(-scores)

        best_scores = [0, 0]
        best_vertices = [None, None]

        r = min(2, len(ids))
        for i in range(r):
            best_scores[i] = scores[ids[i]]
            best_vertices[i] = vertices[ids[i]]

        return best_scores, best_vertices

    def get_cc(self, vertex):
        s_t1 = set()
        s_t2 = set()

        process = [(vertex, 1)]

        while True:
            if not process:
                break

            n_, t_ = process.pop()

            s_test = s_t2
            if t_ == 1:
                s_test = s_t1

            if n_ in s_test:
                continue

            s_test.add(n_)

            if t_ == 1:
                for n2 in n_.out_neighbours():
                    process.append((n2, 2))
            else:
                for n2 in n_.in_neighbours():
                    process.append((n2, 1))

        return list(s_t1), list(s_t2)

    def get_2_best_matchings(self, vertices1, vertices2, order='desc'):
        matchings = []
        scores = []

        self.get_matchings(vertices1, vertices2, [], 0, matchings, scores)

        if len(scores) < 2:
            return scores, matchings

        best_scores = [0, 0]
        best_matchings = [None, None]

        if order == 'asc':
            ids = np.argsort(scores)
        else:
            ids = np.argsort(-np.array(scores))

        for i in range(2):
            best_scores[i] = scores[ids[i]]
            best_matchings[i] = matchings[ids[i]]

        return best_scores, best_matchings

    def get_matchings(self, vertices1, vertices2, m, s, matchings, conf_scores, use_undefined=True, undefined_edge_cost=0):
        if vertices1:
            v1 = vertices1.pop(0)
            for i in range(len(vertices2)):
                v2 = vertices2.pop(0)
                e = self.g.edge(v1, v2)
                if e:
                    self.get_matchings(vertices1, vertices2, m + [(v1, v2)], s+self.g.ep['score'][e], matchings, conf_scores)
                vertices2.append(v2)

            # undefined state
            if use_undefined:
                self.get_matchings(vertices1, vertices2, m + [(v1, None)], s + undefined_edge_cost, matchings, conf_scores)

            vertices1.append(v1)
        else:
            matchings.append(m)
            conf_scores.append(s)

    def get_vertices_in_t(self, t):
        if t in self.vertices_in_t:
            return self.vertices_in_t[t]

        return []

    def all_vertices_and_regions(self, start_frame=-1, end_frame=np.inf):
        l = []
        for t, v_ids in self.vertices_in_t.iteritems():
            if start_frame <= t <= end_frame:
                for v_id in v_ids:
                    l.append((v_id, self.region(v_id)))

        return l

    def get_cc_rec(self, vertex, depth, node_groups):
        # TODO: add max depth param!
        if depth > 10 or not self.g.vp['active'][vertex]:
            return

        r = self.region(vertex)
        if r.frame_ in node_groups and vertex in node_groups[r.frame_]:
            return

        node_groups.setdefault(r.frame_, []).append(vertex)

        for v_ in vertex.in_neighbours():
            ch, ch_end = self.is_chunk(v_)
            if ch and not ch_end:
                continue

            self.get_cc_rec(v_, depth-1, node_groups)

        for v_ in vertex.out_neighbours():
            ch, ch_end = self.is_chunk(v_)
            if ch and ch_end:
                continue

            self.get_cc_rec(v_, depth+1, node_groups)

    def get_cc_from_vertex(self, vertex):
        node_groups = {}
        self.get_cc_rec(vertex, 0, node_groups)

        keys = node_groups.keys()
        keys = sorted(keys)

        g = []
        for k in keys:
            g.append(node_groups[k])

        return g
