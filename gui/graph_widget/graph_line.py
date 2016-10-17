from PyQt4 import QtGui

HASH_PRIME = 100663319


class LineType:
    TRACKLET, LINE, PARTIAL = range(3)

    @staticmethod
    def valid_type(type):
        return type in range(3)

class GraphLine:

    def __init__(self, id, region_from, region_to, type=LineType.LINE, sureness=0, color=QtGui.QColor(0, 0, 0, 120)):
        self.id = id
        self.region_from = region_from
        self.region_to = region_to
        self.type = None
        self.set_type(type)
        self.sureness = sureness
        self.color = color

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


