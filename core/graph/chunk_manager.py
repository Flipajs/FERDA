__author__ = 'flipajs'

from chunk import Chunk


class ChunkManager:
    def __init__(self):
        # default value in graph properties will be 0, so we can easily test...
        self.id_ = 1
        self.chunks_ = {}
        pass

    # TODO:
    # https://docs.python.org/2/reference/datamodel.html#emulating-container-types
    # at least slice access

    def __getitem__(self, index):
        return self.chunks_.get(index, None)

    def new_chunk(self, v1, v2, project):
        self.chunks_[self.id_] = Chunk([v1, v2], self.id_, project)
        self.id_ += 1