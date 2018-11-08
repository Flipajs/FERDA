from __future__ import unicode_literals
from builtins import object
class FakeBGComp(object):
    def __init__(self, project, first_part, part_num, frames_in_row=100):
        self.project = project
        self.part_num = part_num
        self.first_part = first_part
        self.do_semi_merge = True
        self.frames_in_row = frames_in_row

    def update_callback(self, fake1=None, fake2=None):
        pass

    def finished_callback(self, fake1=None, fake2=None):
        pass