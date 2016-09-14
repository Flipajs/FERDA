__author__ = 'flipajs'

from chunk import Chunk


class ChunkManager:
    def __init__(self):
        # default value in graph properties will be 0, so we can easily test...
        self.id_ = 1
        self.chunks_ = {}
        pass

    def __getitem__(self, index):
        return self.chunks_.get(index, None)

    def new_chunk(self, vertices_ids, gm, assign_color=True):
        ch = Chunk(vertices_ids, self.id_, gm)
        self.chunks_[self.id_] = ch

        # assign chunk color
        r1 = gm.region(vertices_ids[0])
        rend = gm.region(vertices_ids[-1])
        # if gm.project.color_manager:
        #     ch.color, _ = gm.project.color_manager.new_track(r1.frame_, rend.frame_)

        self.id_ += 1

        return ch, self.id_ - 1

    def chunk_list(self):
        l = []
        for _, ch in self.chunks_.iteritems():
            l.append(ch)

        return l

    def chunk_gen(self):
        for ch in self.chunks_.itervalues():
            yield ch