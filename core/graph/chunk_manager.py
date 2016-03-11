__author__ = 'flipajs'

from chunk import Chunk
from intervaltree import IntervalTree

class ChunkManager:
    def __init__(self):
        # default value in graph properties will be 0, so we can easily test...
        self.id_ = 1
        self.chunks_ = {}
        self.itree = IntervalTree()
        self.eps1 = 0.01
        self.eps2 = 0.1

    def __getitem__(self, index):
        return self.chunks_.get(index, None)

    def new_chunk(self, vertices_ids, gm, assign_color=True):
        ch = Chunk(vertices_ids, self.id_, gm)
        self.chunks_[self.id_] = ch

        self.itree.addi(ch.start_frame(gm)-self.eps1, ch.end_frame(gm)+self.eps1, ch)

        # assign chunk color
        r1 = gm.region(vertices_ids[0])
        rend = gm.region(vertices_ids[-1])
        if gm.project.color_manager:
            ch.color, _ = gm.project.color_manager.new_track(r1.frame_, rend.frame_)

        self.id_ += 1

        return ch, self.id_ - 1

    def chunk_list(self):
        l = []
        for _, ch in self.chunks_.iteritems():
            l.append(ch)

        return l

    def remove_chunk(self, ch, gm):
        if isinstance(ch, int):
            ch = self.chunks_[ch]

        try:
            self.itree.removei(ch.start_frame(gm)-self.eps1, ch.end_frame(gm)+self.eps1, ch)
        except ValueError:
            pass

        del self.chunks_[ch.id_]

    def get_chunks_from_intervals_(self, intervals):
        chunks = [i.data for i in intervals]

        return chunks

    def chunks_in_frame(self, frame):
        intervals = self.itree[frame-self.eps2:frame+self.eps2]

        return self.get_chunks_from_intervals_(intervals)

    def chunks_in_interval(self, start_frame, end_frame):
        intervals = self.itree[start_frame-self.eps2:end_frame+self.eps2]

        return self.get_chunks_from_intervals_(intervals)

