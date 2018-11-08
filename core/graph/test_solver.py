import unittest

from core.graph.solver import Solver
from core.project.project import dummy_project
from core.graph.graph_manager import GraphManager
from core.region.region import Region
from core.graph.chunk_manager import ChunkManager


class TestSolver(unittest.TestCase):

    def setUp(self):
        self.p = dummy_project()
        self.p.chm = ChunkManager()
        self.p.gm = GraphManager(self.p, None)
        self.solver = Solver(self.p)

        gm = self.p.gm

        self.frames_ = [0, 0, 0, 1, 1, 2, 2, 2]
        self.regions_ = [Region() for i in range(8)]
        for r, f, i in zip(self.regions_, self.frames_, range(len(self.regions_))):
            self.p.rm.add(r)
            r.frame_ = f
            r.id_ = i + 1

            gm.add_vertex(r)

        # simple graph with following structure
        # 0 - 3 - 5
        #   X   X
        # 1 - 4 - 6
        #   /   \
        # 2       7
        #
        # each edge with score 1

        gm.add_edge(0, 3)
        gm.add_edge(0, 4)
        gm.add_edge(1, 3)
        gm.add_edge(1, 4)
        gm.add_edge(2, 4)
        gm.add_edge(3, 5)
        gm.add_edge(3, 6)
        gm.add_edge(4, 5)
        gm.add_edge(4, 6)
        gm.add_edge(4, 7)

    def test_confirm_edge(self):
        gm = self.p.gm
        num_edges = gm.g.num_edges()

        self.solver.confirm_edges([[gm.g.vertex(0), gm.g.vertex(4)]])
        # this should remove 0-3, 1-4 and also 2-4 edge, thus there should be 3 edges less

        self.assertEqual(num_edges-3, gm.g.num_edges())

        # there should be one chunk
        self.assertEqual(1, len(self.p.chm.chunks_))

    def test_confim_multiple_edges(self):
        gm = self.p.gm
        num_edges = gm.g.num_edges()

        self.solver.confirm_edges([
            [gm.g.vertex(0), gm.g.vertex(3)],
            [gm.g.vertex(3), gm.g.vertex(6)]
        ])

        # there should be one chunk
        self.assertEqual(1, len(self.p.chm.chunks_))

        self.assertEqual(num_edges-5, gm.g.num_edges())

    def test_remove_vertex(self):
        gm = self.p.gm
        num_edges = gm.g.num_edges()

        self.solver.strong_remove(gm.g.vertex(4))

        # this should remove 6 edges
        self.assertEqual(num_edges-6, gm.g.num_edges())

    def test_get_vertices_around_t(self):
        v_t_minus, v_t, v_t_plus = self.solver.get_vertices_around_t(1)

        self.assertListEqual([0, 1, 2], v_t_minus)
        self.assertListEqual([3, 4], v_t)
        self.assertListEqual([5, 6, 7], v_t_plus)

    def test_strong_remove(self):
        gm = self.p.gm
        num_edges = gm.g.num_edges()

        self.solver.strong_remove(gm.g.vertex(4))

        # this should remove 6 edges
        self.assertEqual(num_edges-6, gm.g.num_edges())

    # def test_get_ccs(self):
    #     ccs = self.solver.get_ccs([0])
    #
    #     self.assertEqual(1, len(ccs))