__author__ = 'flipajs'

import chunk


class ChunkManager:
    def __init__(self):
        # default value in graph properties will be 0, so we can easily test...
        self.id = 1
        self.chunks_ = {}
        pass

    # TODO:
    # https://docs.python.org/2/reference/datamodel.html#emulating-container-types
    # at least slice access

    def __getitem__(self, index):
        return self.chunks_.get(index, None)

    def new_chunk(self):
        # Chunk(start_n=None, end_n=None, solver=None, store_area=False, id=-1)
        # we have to provide also nm - NodeManager
        pass