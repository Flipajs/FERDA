__author__ = 'flipajs'


class RegionChunk:
    """
    this class is kind of wrapper which simplifies access to the regions when one gets chunk (which stores
    vertices_ids if >= 0 else regions ids.
    """

    def __init__(self, chunk, gm, rm):
        self.chunk_ = chunk
        self.gm_ = gm
        self.rm_ = rm

    def __len__(self):
        return self.chunk_.length()

    def __str__(self):
        s = "RegionChunk start: "+str(self.start_frame())+" end: "+str(self.end_frame())
        return s

    def __getitem__(self, key):
        ids = self.chunk_[key]
        if isinstance(ids, int):
            return self.get_region_(ids)

        new_ids = []
        for id in ids:
            if id < 0:
                id_ = -id
            else:
                id_ = self.gm_.g.vp['region_id'][self.gm_.g.vertex(id)]

            new_ids.append(id_)

        return self.rm_[new_ids]

    def get_region_(self, id):
        # if there is direct link to region, the id is negative
        if id < 0:
            return self.rm_[-id]
        else:
            return self.gm_.region(id)

    def start_frame(self):
        return self[0].frame_

    def end_frame(self):
        return self[-1].frame_

    def region_in_t(self, t):
        t = t-self.start_frame()
        if -1 < t < len(self.chunk_.nodes_):
            return self[t]
        else:
            return None

    def centroid_in_t(self, t):
        r = self.region_in_t(t)
        if r:
            return r.centroid()
        else:
            return None

    def regions_gen(self):
        i = 0
        while i < self.chunk_.length():
            yield self[i]
            i += 1

    def rid_gen(self):
        for r in self.regions_gen():
            yield r.id()