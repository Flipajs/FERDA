from __future__ import print_function
from __future__ import unicode_literals
from builtins import zip
from builtins import range
__author__ = 'flipajs'

import graph_tool
from core.project.project import Project


if __name__ == "__main__":
    project = Project()
    project.load('/Users/flipajs/Documents/wd/eight/eight.fproj')

    g = project.solver.g

    nodes = set()
    for n in g.nodes()[0:10]:
        nodes.add(n)

    edges = []
    nodes2 = set()
    for n in nodes:
        for _, n2, d in g.out_edges(n, data=True):
            nodes2.add(n2)
            edges.append((n, n2, d))

        for n1, _, d in g.in_edges(n, data=True):
            nodes2.add(n1)
            edges.append((n, n1, d))

    nodes = nodes.union(nodes2)

    g2 = graph_tool.Graph(directed=True)
    g2.vp['region'] = g2.new_vertex_property("int")
    # g2.vp['chunk_start'] = g2.new_vertex_property("object")
    # g2.vp['chunk_end'] = g2.new_vertex_property("object")
    g2.ep['cost'] = g2.new_edge_property("float")

    ids = {}
    for i, n in zip(list(range(len(nodes))), nodes):
        ids[n] = i

    region_vertex_refs = {}
    nodes = list(nodes)
    vertices = []
    for n in nodes:
        vertex = g2.add_vertex()
        vertices.append(vertex)
        g2.vp['region'][vertex] = ids[n]
        region_vertex_refs[n] = vertex

    edges_ = []
    for n1, n2, d in edges:
        e_ = g2.add_edge(region_vertex_refs[n1], region_vertex_refs[n2])
        g2.ep['cost'][e_] = d['score']
        edges_.append(e_)

    # import cPickle as pickle
    # with open('/Users/flipajs/Documents/wd/test.pkl', 'wb') as f:
    #     pickle.dump(g2, f, -1)

    # for e in edges_:
    #     try:
    #         print g2.vp['region'][e.source()], " -> ", g2.vp['region'][e.target()], "COST: ", g2.ep['cost'][e]
    #     except:
    #         pass
    #
    # print "REMOVING NODE", nodes[4]
    my_node = vertices[-1]
    print(my_node, g2.vp['region'][my_node])
    g2.remove_vertex(vertices[1], fast=True)
    g2.remove_vertex(vertices[6], fast=True)
    g2.remove_vertex(vertices[4], fast=True)
    print(my_node, g2.vp['region'][my_node])

    # for e in edges_:
    #     try:
    #         print g2.vp['region'][e.source()], " -> ", g2.vp['region'][e.target()], "COST: ", g2.ep['cost'][e]
    #     except:
    #         pass

