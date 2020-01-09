__author__ = 'flipajs'
import numpy as np
import numbers


class RegionChunk(object):
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
        if ids is None:
            return None

        if isinstance(ids, numbers.Integral):
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
        return self.chunk_.start_frame()
        # return self[0].frame_

    def end_frame(self):
        return self.chunk_.end_frame()
        # return self[-1].frame_

    def region_in_t(self, t):
        t = t-self.start_frame()
        if -1 < t < len(self.chunk_):
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
            x = self[i]
            if x is not None:
                yield x
            i += 1

    def rid_gen(self):
        for r in self.regions_gen():
            yield r.id()

    def fix_regions_orientation(self):
        import heapq
        from core.region.region import get_region_endpoints
        import numpy as np
        q = []
        # q = Queue.PriorityQueue()
        heapq.heappush(q, (0, [False]))
        heapq.heappush(q, (0, [True]))
        # q.put((0, [False]))
        # q.put((0, [True]))

        result = []
        i = 0
        max_i = 0

        cut_diff = 10

        while True:
            i += 1

            # cost, state = q.get()
            cost, state = heapq.heappop(q)
            if len(state) > max_i:
                max_i = len(state)

            if len(state) + cut_diff < max_i:
                continue

            # print i, cost, len(state), max_i

            if len(state) == len(self):
                result = state
                break

            prev_r = self[len(state) - 1]
            r = self[len(state)]

            prev_c = prev_r.centroid()
            p1, p2 = get_region_endpoints(r)

            dist = np.linalg.norm
            d1 = dist(p1 - prev_c)
            d2 = dist(p2 - prev_c)

            prev_head, prev_tail = get_region_endpoints(prev_r)
            if state[-1]:
                prev_head, prev_tail = prev_tail, prev_head

            d3 = dist(p1 - prev_head) + dist(p2 - prev_tail)
            d4 = dist(p1 - prev_tail) + dist(p2 - prev_head)

            # state = list(state)
            state2 = list(state)
            state.append(False)
            state2.append(True)

            new_cost1 = d3
            new_cost2 = d4

            # TODO: param
            if dist(prev_c - r.centroid()) > 5:
                new_cost1 += d2 - d1
                new_cost2 += d1 - d2

            heapq.heappush(q, (cost + new_cost1, state))
            heapq.heappush(q, (cost + new_cost2, state2))
            # q.put((cost + new_cost1, state))
            # q.put((cost + new_cost2, state2))

        # fix tracklet
        n_swaps = 0
        for do_orientation_swap, r in zip(result, self):
            if do_orientation_swap:
                n_swaps += 1
                r.theta_ += np.pi
                if r.theta_ >= 2 * np.pi:
                    r.theta_ -= 2 * np.pi
        return n_swaps

    @property
    def centroids(self):
        return np.array([r.centroid() for r in self.regions_gen()])

    @property
    def speed(self):
        return np.linalg.norm(np.diff(self.centroids, axis=0), axis=1).mean()

    def draw(self):
        import matplotlib.pylab as plt
        import numpy as np
        polyline_xy = np.array([r.centroid()[::-1] for r in self.regions_gen()])
        plt.plot(polyline_xy[:, 0], polyline_xy[:, 1])

