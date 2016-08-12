from graph_tool.draw import graph_draw

from core.project.project import Project


class GraphSupplier:
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager
        self.graph = graph_manager.g

    def visualize_graph(self):
        graph_draw(self.graph, vertex_text=self.graph.vertex_index, fit_view=True)

    def get_nodes_tuples(self):
        ret = []
        queue = [self.graph.vertex(x) for x in self.graph_manager.start_nodes()]
        visited = set(queue)
        for vertex in queue:
            for neighbour in vertex.out_neighbours():
                ret.append((vertex, neighbour))
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.add(neighbour)
        return ret


if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/CompleteGraph/CompleteGraph.fproj")

    supplier = GraphSupplier(project.gm)
    print supplier.get_nodes_tuples()
