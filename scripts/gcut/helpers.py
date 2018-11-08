from __future__ import unicode_literals
from builtins import object
__author__ = 'flipajs'


class TWeights(object):
    def __init__(self):
        self.nodes = []
        self.capacities_source = []
        self.capacities_sink = []
        self.refs = {}

    def add(self, node, source_capacity, sink_capacity):
        self.refs[node] = len(self.nodes)
        self.nodes.append(node)
        self.capacities_source.append(source_capacity)
        self.capacities_sink.append(sink_capacity)

    def plus_weights(self, node, plus_source, plus_sink):
        self.capacities_source[self.refs[node]] += plus_source
        self.capacities_sink[self.refs[node]] += plus_sink

    def source_edges_sum(self):
        return sum(self.capacities_source)

    def sink_edges_sum(self):
        return sum(self.capacities_sink)

class Edges(object):
    def __init__(self):
        self.n1 = []
        self.n2 = []
        self.cost = []

    def add(self, n1, n2, c):
        self.n1.append(n1)
        self.n2.append(n2)
        self.cost.append(c)

    def num(self):
        return len(self.n1)
