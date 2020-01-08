from PyQt4 import QtGui

HASH_PRIME = 100663319


class LineType:
    TRACKLET, LINE, PARTIAL_TRACKLET = list(range(3)) # line cannot be partial (takes only two frames)

    @staticmethod
    def valid_type(type):
        return type in range(3)

class Overlap:

    def __init__(self, left=False, right=False):
        self.left = left
        self.right = right


class GraphLine:

    def __init__(self, id, region_from, region_to, type=LineType.LINE, overlap=Overlap(), sureness=0, color=QtGui.QColor(0, 0, 0, 120), appearance_score=0, movement_score=0):
        self.id = id
        self.region_from = region_from
        self.region_to = region_to
        self.type = type
        self.overlap = overlap
        self.set_type(type)
        self.sureness = sureness
        self.color = color
        self.appearance_score = appearance_score
        self.movement_score = movement_score

    def __hash__(self):
        if self.type == LineType.TRACKLET:
            return self.id
        else:
            return self.region_from.id() * HASH_PRIME + self.region_to.id()

    def set_type(self, type):
        if LineType.valid_type(type):
            self.type = type
        else:
            raise ValueError("Line type is invalid!")

    def overlaps_left(self):
        return self.overlap.left

    def overlaps_right(self):
        return self.overlap.right

    def set_overlap_left(self, p):
        self.overlap.left = p

    def set_overlap_right(self, p):
        self.overlap.right = p
