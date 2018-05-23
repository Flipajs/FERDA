from collections import namedtuple
import time

BlobInfo = namedtuple("BlobInfo", "region_id frame tracklet_id")
BlobData = namedtuple("BlobData", "ants date")


class AntBlobs:
    """
    {key: ( (BlobInfo, BlobData), key: (BlobInfo, BlobData), ...}

    BlobInfo(region_id=63729, frame=14431, tracklet_id=70922)
    BlobData(ants=[array([[23, 54], ...]),
                   array([[45, 88], ...]),
                   ...],
             date='22 z\xc3\xa1\xc5\x99 2017 16:59:46')
    """

    def __init__(self):
        self.blobs = {}

    def feed_blobs(self):
        for k in sorted(self.blobs.iterkeys()):
            yield k, self.blobs[k]

    def __len__(self):
        return len(self.blobs)

    def all_blobs(self):
        """
        Returns list of annotated blobs.

        :return: list of AntBlobs()
        """
        return self.blobs.items()

    def contains(self, region_id, frame, tracklet_id):
        return BlobInfo(region_id, frame, tracklet_id) in self.blobs

    def insert(self, region_id, frame, tracklet_id, ants):
        self.blobs[BlobInfo(region_id, frame, tracklet_id)] = BlobData(ants, time.strftime("%d %b %Y %H:%M:%S"))

    def filter_labeled_tracklets(self, tracklets):
        labeled_tracklets = set()
        for k in self.blobs.keys():
            labeled_tracklets.add(k.tracklet_id)
        return list(filter(lambda x: x.id() not in labeled_tracklets, tracklets))
