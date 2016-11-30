import numpy as np

class Solver2:
    def __init__(self, project):
        self.p = project

    def prune_distant_connections(self, max_dist):
        """
        removes all edges with distance > threshold
        Returns:

        """

        # TODO: add global parameter:
        max_dist_increased = max_dist * 1.2
        g = self.p.gm.g
        print "_______________________________________________________"
        print "pruning edges based on distance threshold"
        print "max_dist: {:.1f}, max_dist_increased: {:.1f}".format(max_dist, max_dist_increased)
        print "BEFORE"
        print "avg vertex out degree before {:.1f}".format(np.mean([v.out_degree() for v in g.vertices()]))
        num_edges = g.num_edges()
        print "#edges: {}".format(num_edges)

        for v in self.p.gm.active_v_gen():
            r1 = self.p.gm.region(v)
            to_prune = []
            for e in v.out_edges():
                r2 = self.p.gm.region(e.target())

                if r1.is_ignorable(r2, max_dist):
                    to_prune.append(e)

            for e in to_prune:
                self.p.gm.remove_edge_(e)

        degrees = [v.out_degree() for v in g.vertices()]
        print "AFTER"
        print "#edges: {} (removed: {})".format(g.num_edges(), num_edges - g.num_edges())
        print "avg vertex out degree after {:.1f}".format(np.mean(degrees))
        print "------------------------------------"

    # TODO:
    """
    priority queue with update option
    A) https://docs.python.org/2/library/heapq.html
    B) dict with item and timestamp.. if not valid, ignore and continue... + maybe already solved dict
    """


