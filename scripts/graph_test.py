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
    # g.vp['region'] = g.new_vertex_property("int")
    # g.vp['chunk_start'] = g.new_vertex_property("int")
    # g.vp['chunk_end'] = g.new_vertex_property("int")
    # g.ep['cost'] = g.new_edge_property("float")

    vertices = []
    id = 0
    for i in range(0, lenght):
        n = []
        for j in range(0, nodes):
            n.append(g.add_vertex())
            # if i % 3:
                # g.vp['region'][n[-1]] = i
                # g.vp['chunk_start'][n[-1]] = i*13
                # g.vp['chunk_end'][n[-1]] = i*14
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
        k = vertex.out_neighbors()


def delete_edges_graph_tool(graph, edges, ids):
    for i in range(0, len(ids)):
        graph.remove_edge(edges[ids[i]])
        #graph.remove_edge(edges[ids[i]][0], edges[ids[i]][1])

def delete_nodes_graph_tool(graph, nodes, ids):
    for i in range(0, len(ids)):
        graph.remove_vertex(nodes[i])


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

def delete_nodes_networkx(graph, nodes, ids):
    for i in range(0, len(ids)):
        graph.remove_node(nodes[i])

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

lim = 30*60*10

t = time.time()
graph_tool, graph_tool_vertices, graph_tool_edges = create_graph_tool(50, lim)
print "graph_tool creation: %s" % (time.time()-t)
t = time.time()
search_graph_tool(graph_tool)
print "graph_tool search: %s" % (time.time()-t)
t = time.time()
delete_edges_graph_tool(graph_tool, graph_tool_edges, rnd_array)
print "graph_tool edges deletion: %s" % (time.time()-t)
t = time.time()
delete_nodes_graph_tool(graph_tool, graph_tool_vertices, rnd_array)
print "graph_tool nodes deletion: %s" % (time.time()-t)

import cPickle as pickle
s = time.time()
with open('/Users/flipajs/Documents/wd/graph_test_graph_tool.pkl', 'wb') as f:
    pickle.dump(graph_tool, f, -1)
print "DUMP TIME: ", time.time() - s

t = time.time()
networkx, networkx_vertices, networkx_edges = create_networkx(50, lim)
print "networkx creation: %s" % (time.time()-t)
t = time.time()
search_networkx(networkx)
print "networkx search: %s" % (time.time()-t)
t = time.time()
delete_edges_networkx(networkx, networkx_edges, rnd_array)
print "networkx edges deletion: %s" % (time.time()-t)

t = time.time()
delete_nodes_networkx(networkx, networkx_vertices, rnd_array)
print "networkx nodes deletion: %s" % (time.time()-t)

s = time.time()
with open('/Users/flipajs/Documents/wd/graph_test_networkx.pkl', 'wb') as f:
    pickle.dump(networkx, f, -1)
print "DUMP TIME: ", time.time() - s

# t = time.time()
# igraph, igraph_vertices, igraph_edges = create_igraph(10,lim)
# print "igraph creation: %s" % (time.time()-t)
# t = time.time()
# search_igraph(igraph)
# print "igraph search: %s" % (time.time()-t)
# t = time.time()
# delete_edges_igraph(igraph, igraph_edges, rnd_array)
# print "igraph edges deletion: %s" % (time.time()-t)

# alternativy metod z networkx v graph_tool
"""
g = DiGraph() -> g = Graph(directed=True)
g.add_node(hashable n) -> v = g.add_vertex() # graph_tool's add vertex method returns the vertex. It can be used to
  create multiple (M) vertices at a time by calling g.add_vertex(n=M). In that case, it returns iterator over them.
g.add_edge(n1, n2) -> e = g.add_node(v1, v2)
g.in_edges(hasheble n) -> v.in_edges() # return an iterator over the in-edges
g.out_edges(hashable n) -> v.out_edges() # return an iterator over the out-edges
g.remove_node(hashable n) -> g.remove_vertex(v) # v can also be an iterator
g.get_edge_data(n1, n2) -> # there is no such method in graph_tool, since it's edges have no data but source and target
g.remove_edge(n1, n2) -> g.remove_edge(e)
g.in_degree(n) -> v.in_degree()
g.out_degree(n) -> v.out_degree(
g.nodes() -> g.vertices()
Edge/vertex can be found by id using g.vertex(id)







"""






















