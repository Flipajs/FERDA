

class FittingLogger:
    def __init__(self):
        self.log_ = {}

    def add(self, new_vertices, model_vertices, merged_vertices):
        new_vertices = list(map(int, new_vertices))
        model_vertices = list(map(int, model_vertices))
        merged_vertices = list(map(int, merged_vertices))

        for v in new_vertices:
            self.log_[v] = {'new_vertices': new_vertices,
                            'model_vertices': model_vertices,
                            'merged_vertices': merged_vertices
                            }

    def undo_recipe(self, new_vertex):
        new_vertex = int(new_vertex)

        if new_vertex in self.log_:
            recipe = self.log_[new_vertex]
            del self.log_[new_vertex]

            return recipe
        else:
            return None