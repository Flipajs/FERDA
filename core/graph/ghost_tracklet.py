from chunk import Chunk

class GhostTracklet(Chunk):
    def __init__(self, start, end):
        self.start = start
        self.end = end

        self.nodes_ = []

    def __len__(self):
        return self.end - self.start

    def start_frame(self, gm=None):
        return self.start

    def end_frame(self, gm=None):
        return self.end
