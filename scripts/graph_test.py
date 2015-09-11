import networkx
import random
import graph_tool.all as gt
import igraph


class Node():
    def __init__(self, id, i, j):
        self.id = id
        self.i = i
        self.j = j


def create_igraph(nodes, lenght):
    g = igraph.Graph()
    frames = []
    id = 0
    for i in range(0, lenght):
        n = []
        for j in range(0, nodes):
            n.append(Node(id, i, j))
            g.add_vertex(id, shape='hidden')
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
            vertices.append(g.add_edge(n1.id, n2.id))
            id += 1

    print "Graph complete!"
    import time
    t = time.time()
    layout = g.layout("rt")
    igraph.plot(g, layout = layout, )
    print time.time()-t

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
    import time
    t = time.time()
    gt.graph_draw(g)
    print time.time()-t



create_igraph(10, 30)
#create_graph_tool(10, 300)
