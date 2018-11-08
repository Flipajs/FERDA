from __future__ import unicode_literals


from builtins import object
class Line(object):

    def __init__(self, from_region, to_region, type, sureness, color, source_start_id):
        self.from_region = from_region
        self.to_region = to_region
        self.type = type
        self.sureness = sureness
        self.color = color
        self.source_start_id = source_start_id
