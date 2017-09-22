from collections import namedtuple
import time

BlobInfo = namedtuple("BlobInfo", "region_id frame tracklet_id")
BlobData = namedtuple("BlobData", "ants date")


class AntBlobs:

    def __init__(self):
        self.blobs = {}

    def feed_blobs(self):
        for k in sorted(self.blobs.iterkeys()):
            yield k, self.blobs[k]

    def __len__(self):
        return len(self.blobs)

    def all_blobs(self):
        return self.blobs.items()

    def contains(self, region_id, frame, tracklet_id):
        return BlobInfo(region_id, frame, tracklet_id) in self.blobs

    def insert(self, region_id, frame, tracklet_id, ants):
        self.blobs[BlobInfo(region_id, frame, tracklet_id)] = BlobData(ants, time.strftime("%d %b %Y %H:%M:%S"))




