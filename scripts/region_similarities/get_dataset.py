from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
import pickle as pickle
from core.project.project import Project
from core.graph.region_chunk import RegionChunk


if __name__ == '__main__':
    p = Project()
    p.load('/Users/flipajs/Documents/wd/C210/C210.fproj')

    chunks = []
    for v_id in p.gm.get_all_relevant_vertices():
        ch_id = p.gm.g.vp['chunk_start_id'][p.gm.g.vertex(v_id)]
        if ch_id > 0:
            r_ch = RegionChunk(p.chm[ch_id], p.gm, p.rm)
            chunks.append(r_ch[:])

    with open('/Users/flipajs/Documents/dev/ferda/scripts/datasets/c210-few_chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f, -1)
