import networkx as nt
import graph_tool.all as gt
import igraph as ig
import random

"""
               graph-tool                                                igraph                                           networkx
docs:          https://graph-tool.skewed.de/static/doc/graph_tool.html   http://igraph.org/python/doc/igraph-module.html
pyqt:          No (cairo interactive popup)                              No (low quality image output)
new graph:     g = Graph(directed=True)                                  g = Graph(directed=True)
new edge:      g.add_edge(Vertex from, Vertex to)                        g.add_edge(int id1, int id2)
new node:      g.add_vertex()                                            g.add_vertex()

small random graph creation without visualisation (results are in seconds, average from at least 5 measurings):
600 vertices   0.017                                                     0.03                                             0.0047
1000 vertices  0.027                                                     0.077                                            0.0055
4000 vertices  0.12                                                      1.1                                              0.028
20000 vertices 0.57                                                      26.6                                             0.25
"""


def create_igraph(nodes, lenght):
    # vertices is a list of ids
    # edges is a list of ids
    g = ig.Graph(directed=True)
    vertices = []
    id = 0
    for i in range(0, lenght):
        n = []
        for j in range(0, nodes):
            n.append(id)
            g.add_vertex(id, shape='hidden')
            id += 1
        vertices.append(n)

    # print "Nodes are ready"

    edges = []
    id = 0
    for i in range(0, lenght-1):
        for j in range(0, nodes):
            d1 = random.randint(1, nodes-1)
            edges.append(id)
            g.add_edge(vertices[i][j], vertices[i+1][d1])
            id += 1

    # print "Graph complete!"
    #layout = g.layout("rt")
    #ig.plot(g, layout = layout, )
    return g, vertices, edges


def search_igraph(graph):
    for vertex in graph.vs():
        k = graph.neighbors(vertex, mode="IN")


def delete_edges_igraph(graph, edges, ids):
    for i in range(0, len(ids)):
        graph.delete_edges(edges[ids[i]])
        #graph.remove_edge(edges[ids[i]][0], edges[ids[i]][1])


def create_graph_tool(nodes, lenght):
    # vertices is a list of Vertex objects
    # edges is a list of Edge objects
    g = gt.Graph(directed=True)
    vertices = []
    id = 0
    for i in range(0, lenght):
        n = []
        for j in range(0, nodes):
            n.append(g.add_vertex())
            id += 1
        vertices.append(n)

    # print "Nodes are ready"

    edges = []
    id = 0
    for i in range(0, lenght-1):
        for j in range(0, nodes):
            n1 = vertices[i][j]
            d1 = random.randint(1, nodes-1)
            n2 = vertices[i+1][d1]
            edges.append(g.add_edge(n1, n2))
            id += 1

    # print "Graph complete!"
    #gt.graph_draw(g)
    return g, vertices, edges


def search_graph_tool(graph):
    for vertex in graph.vertices():
        k = vertex.out_neighbours()


def delete_edges_graph_tool(graph, edges, ids):
    for i in range(0, len(ids)):
        graph.remove_edge(edges[ids[i]])
        #graph.remove_edge(edges[ids[i]][0], edges[ids[i]][1])


def create_networkx(nodes, lenght):
    # vertices is a list of vertex ids
    # edges is a list of int tuples (from_vertex id, to_vertex id)
    g = nt.DiGraph()
    vertices = []
    id = 0
    for i in range(0, lenght):
        n = []
        for j in range(0, nodes):
            n.append(id)
            g.add_node(id)
            id += 1
        vertices.append(n)

    # print "Nodes are ready"

    edges = []
    id = 0
    for i in range(0, lenght-1):
        for j in range(0, nodes):
            n1 = vertices[i][j]
            d2 = random.randint(1, nodes-1)
            n2 = vertices[i+1][d2]
            g.add_edge(n1, n2)
            edges.append((n1, n2))
            id += 1

    # print "Graph complete!"
    return g, vertices, edges


def search_networkx(graph):
    for vertex in graph.nodes():
        k = graph.out_edges(vertex)


def delete_edges_networkx(graph, edges, ids):
    for i in range(0, len(ids)):
        graph.remove_edge(edges[ids[i]][0], edges[ids[i]][1])


def randoms(array):
    for i in range(0, len(array)):
        yield array[i]


def get_rnd(max, length):
    result = []
    for i in range(0, length):
        rnd = random.randint(0, max)
        while rnd in result:
            rnd = random.randint(0, max)
        result.append(rnd)
    return result


import time
rnd_array = get_rnd(1000, 500)
print "Random done"

lim = 30*60*60

t = time.time()
graph_tool, graph_tool_vertices, graph_tool_edges = create_graph_tool(10, lim)
print "graph_tool creation: %s" % (time.time()-t)
t = time.time()
search_graph_tool(graph_tool)
print "graph_tool search: %s" % (time.time()-t)
t = time.time()
delete_edges_graph_tool(graph_tool, graph_tool_edges, rnd_array)
print "graph_tool edges deletion: %s" % (time.time()-t)

t = time.time()
networkx, networkx_vertices, networkx_edges = create_networkx(10, lim)
print "networkx creation: %s" % (time.time()-t)
t = time.time()
search_networkx(networkx)
print "networkx search: %s" % (time.time()-t)
t = time.time()
delete_edges_networkx(networkx, networkx_edges, rnd_array)
print "networkx edges deletion: %s" % (time.time()-t)

t = time.time()
igraph, igraph_vertices, igraph_edges = create_igraph(10,lim)
print "igraph creation: %s" % (time.time()-t)
t = time.time()
search_igraph(igraph)
print "igraph search: %s" % (time.time()-t)
t = time.time()
delete_edges_igraph(igraph, igraph_edges, rnd_array)
print "igraph edges deletion: %s" % (time.time()-t)
