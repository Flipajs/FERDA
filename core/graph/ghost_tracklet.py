from chunk import Chunk

class GhostTracklet(Chunk):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.id_ = -1

        self.nodes_ = []

    def __len__(self):
        return self.end - self.start

    def start_frame(self, gm=None):
        return self.start

    def end_frame(self, gm=None):
        return self.end

    def __getitem__(self, item):
        return None

    def is_ghost(self):
        return True

    def is_tracklet(self):
        return False