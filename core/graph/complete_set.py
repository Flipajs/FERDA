import numpy as np
from utils.visualization_utils import generate_colors


class CompleteSet(object):
    # TODO: rename to TrackList / TrackletList
    def __init__(self, tracks_or_tracklets):
        self.tracklets = tracks_or_tracklets

    def start_frame(self):
        return max([t.start_frame() for t in self.tracklets])

    def end_frame(self):
        return min([t.end_frame() for t in self.tracklets])

    def get_start_end(self):
        return self.start_frame(), self.end_frame()

    def __getitem__(self, item):
        return self.tracklets[item]

    def draw(self, rm):
        for t, color in zip(self.tracklets, generate_colors(len(self.tracklets))):
            t.draw(rm, c=color, label=t.id())

    def get_intervals(self):
        return [t.get_interval() for t in self.tracklets]









