from __future__ import unicode_literals


from builtins import range
from builtins import object
class TrackletTypes(object):
    BLOB, SINGLE, OTHER = list(range(3))

    def __init__(self):
        self.tracklet_types = {}

    def contains(self, tracklet_id):
        return tracklet_id in self.tracklet_types

    def __len__(self):
        return len(self.tracklet_types)

    def get_unlabeled(self, tracklets):
        return [t for t in tracklets if t.id() not in self.tracklet_types]

    def get_labeled_blobs(self, tracklets):
        return [t for t in tracklets if self.tracklet_types.get(t.id(), self.SINGLE) == self.BLOB]

    def insert(self, tracklet_id, type):
        self.tracklet_types[tracklet_id] = type