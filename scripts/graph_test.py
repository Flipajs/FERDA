import networkx as nt
import graph_tool.all as gt
import igraph as ig
import numpy as np
import random

"""
               graph-tool                                                igraph                                           networkx
docs:          https://graph-tool.skewed.de/static/doc/graph_tool.html   http://igraph.org/python/doc/igraph-module.html
pyqt:          No (cairo interactive popup)                              No (low quality image output)
new graph:     g = Graph(directed=True)                                  g = Graph(directed=True)
new edge:      g.add_edge(Vertex from, Vertex to)                        g.add_edge(int id1, int id2)
new node:      g.add_vertex()                                            g.add_vertex()

random graph creation without visualisation (results are in seconds, average from at least 5 measurings):
600 vertices   0.017                                                     0.03                                             0.0047
1000 vertices  0.027                                                     0.077                                            0.0055
4000 vertices  0.12                                                      1.1                                              0.028
20000 vertices 0.57                                                      26.6                                             0.25
"""

# TODO: check the remove_vertex and find_vertex functions in graph-tool





class Node():
    def __init__(self, id, i, j):
        self.id = id
        self.i = i
        self.j = j


def create_networkx(nodes, lenght):
    g = nt.DiGraph()
    frames = []
    id = 0
    for i in range(0, lenght):
        n = []
        for j in range(0, nodes):
            n.append(id)
            g.add_node(id)
            id += 1
        frames.append(n)

    print "Nodes are ready"

    vertices = []
    id = 0
    for i in range(0, lenght-1):
        for j in range(0, nodes):
            d1 = random.randint(1, nodes-1)
            vertices.append(g.add_edge(frames[i][j], frames[i+1][d1]))
            id += 1

    print "Graph complete!"
    return g


def create_igraph(nodes, lenght):
    g = ig.Graph(directed=True)
    frames = []
    id = 0
    for i in range(0, lenght):
        n = []
        for j in range(0, nodes):
            n.append(id)
            g.add_vertex(id, shape='hidden')
            id += 1
        frames.append(n)

    print "Nodes are ready"

    vertices = []
    id = 0
    for i in range(0, lenght-1):
        for j in range(0, nodes):
            d1 = random.randint(1, nodes-1)
            vertices.append(g.add_edge(frames[i][j], frames[i+1][d1]))
            id += 1

    print "Graph complete!"
    #layout = g.layout("rt")
    #ig.plot(g, layout = layout, )
    return g

def create_graph_tool(nodes, lenght):
    g = gt.Graph(directed=True)
    frames = []
    id = 0
    for i in range(0, lenght):
        n = []
        for j in range(0, nodes):
            n.append(g.add_vertex())
            id += 1
        frames.append(n)

    print "Nodes are ready"

    vertices = []
    id = 0
    for i in range(0, lenght-1):
        for j in range(0, nodes):
            n1 = frames[i][j]
            d1 = random.randint(1, nodes-1)
            n2 = frames[i+1][d1]
            vertices.append(g.add_edge(n1, n2))
            id += 1

    print "Graph complete!"
    #gt.graph_draw(g)
    return g

def randoms(array):
    for i in range(0, len(array)):
        yield array[i]


import time
t = time.time()
igraph = create_igraph(2, 100)
print "igraph: %s" % (time.time()-t)
"""
t = time.time()
graph_tool = create_graph_tool(2, 10000)
print "graph-tool: %s" % (time.time()-t)
"""
t = time.time()
networkx = create_networkx(2, 10000)
print "networkx: %s" % (time.time()-t)


# 100 random numbers from range 0 to 100
rnd_ints = np.random.rand(0, 100, 100)
igraph.delete_edges(randoms(rnd_ints))
