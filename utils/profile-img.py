from __future__ import unicode_literals
from builtins import range
from core.project.project import Project
from core.id_detection.features import get_colornames_hists
from core.graph.region_chunk import RegionChunk

p = Project()
p.load('/Users/flipajs/Documents/wd/FERDA/rep1-cam3')

for frame in range(100):
    for ch in p.chm.tracklets_in_frame(frame):
        rch = RegionChunk(ch, p.gm, p.rm)
        r = rch.region_in_t(frame)

        f = get_colornames_hists(r, p)
