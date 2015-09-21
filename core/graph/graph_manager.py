__author__ = 'flipajs'

import graph_tool
from core.log import LogCategories, ActionNames
import numpy as np


class GraphManager:
    def __init__(self, project):
        self.project = project
        self.g = graph_tool.Graph(directed=True)
        self.graph_add_properties()
        self.nodes_in_t = {}
        self.start_t = np.inf
        self.end_t = -1

    def graph_add_properties(self):
        self.g.vp.region = self.g.new_vertex_property("object")
        self.g.vp.chunk_start = self.g.new_vertex_property("object")
        self.g.vp.chunk_end = self.g.new_vertex_property("object")
        self.g.ep.cost = self.g.new_edge_property("float")

    def add_vertex(self, region):
        self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.ADD_NODE, region)
        self.start_t = min(self.start_t, region.frame_)
        self.end_t = max(self.end_t, region.frame_)

        vertex = self.g.add_vertex()

        self.nodes_in_t.setdefault(region.frame_, []).append(vertex)
        self.g.vp.region[vertex] = region

    def remove_vertex(self, vertex, disassembly=True):
        region = self.graph.vp.regions[vertex]
        region = self.match_if_reconstructed(region)
        if region is None:
            print "remove node n is None"
            return

        if disassembly:
            ch, chunk_end = self.is_chunk(vertex)
            if ch:
                ch.pop_last(self) if chunk_end else ch.pop_first(self)

        # save all edges
        for e in vertex.in_edges():
            self.project.log.add(LogCategories.GRAPH_EDIT,
                                 ActionNames.REMOVE_EDGE,
                                 {'n1': e.source(), 'n2': e.target(), 'data': self.g.ep.cost[e]})

        for e in vertex.out_edges():
            self.project.log.add(LogCategories.GRAPH_EDIT,
                                 ActionNames.REMOVE_EDGE,
                                 {'n1': e.soure(), 'n2': e.target(), 'data': self.g.ep.cost[e]})

        self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.REMOVE_NODE, region)

        self.nodes_in_t[region.frame_].remove(vertex)
        if not self.nodes_in_t[region.frame_]:
            del self.nodes_in_t[region.frame_]

        self.g.remove_vertex(vertex)

        # maybe we need to shrink time boundaries...
        if self.end_t == region.frame_ or self.start_t == region.frame_:
            self.update_time_boundaries()

    def is_chunk(self, vertex):
        """
        check whether the vertex is start or end of chunk
        :param n: ref to vertex in g
        :return: (chunk_ref (ref or None), is_chunk_end (True if it is chunk_end))
        """
        chunk_start = self.g.vp.chunk_start[vertex]
        if chunk_start:
            return chunk_start, False

        chunk_end = self.g.vp.chunk_end[vertex]
        if chunk_end:
            return chunk_end, True

        return None, False

    def update_time_boundaries(self):
        self.start_t = np.inf
        self.end_t = -1

        for frame in self.nodes_in_t:
            self.start_t = min(self.start_t, frame)
            self.end_t = max(self.end_t, frame)
