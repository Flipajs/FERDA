from __future__ import unicode_literals

from builtins import object
class TrackletCompleteSet(object):
    def __init__(self, tracklets, id, left_neighbour=None, right_neighbour=None):
        self.tracklets = tracklets
        self.id = id
        self.left_neighbour = left_neighbour
        self.right_neighbour = right_neighbour

