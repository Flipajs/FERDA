import unittest
from chunk import Chunk
from core.graph.graph_manager import GraphManager
from core.project.project import Project
from core.region.region import Region


class TestChunk(unittest.TestCase):
    def setUp(self):
        self.p = Project()
        # region ids:   0  1  2  3  4  5  6  7   8  9
        self.frames_ = [1, 2, 3, 3, 4, 5, 6, 90, 0, 12]
        self.regions_ = [Region(None, frame) for frame in self.frames_]
        for r in self.regions_:
            self.p.rm.append(r)  # RegionManager.append takes care about id
        self.gm = self.p.gm
        self.gm.add_vertices(self.regions_)

    # exceptions disabled in core/graph/chunk.py:16
    # def test_not_list_init(self):
    #     self.assertRaises(Exception, Chunk, (1, 2), 1, self.gm)
    #
    # def test_not_enough_vertices(self):
    #     self.assertRaises(Exception, Chunk, [1], 1, self.gm)

    def test_init(self):
        ch = Chunk([1, 2], 1, self.gm)

        self.assertEqual(2, len(ch.nodes_))

    def test_append_left_raise_discontinuity_test(self):
        ch = Chunk([0, 1], 1, self.gm)
        self.assertRaises(Exception, ch.append_left, self.gm.g.vertex(3), self.gm)

    def test_append_right_raise_discontinuity_test(self):
        ch = Chunk([0, 1], 1, self.gm)
        self.assertRaises(Exception, ch.append_right, self.gm.g.vertex(4), self.gm)

    def test_append_left(self):
        ch = Chunk([1, 2], 1, self.gm)
        ch.append_left(self.gm.g.vertex(0))
        self.assertEqual(ch.start_vertex_id(), 0)

    def test_append_right(self):
        ch = Chunk([0, 1], 1, self.gm)
        ch.append_right(self.gm.g.vertex(2))
        self.assertEqual(self.gm.g.vertex(2), ch.end_vertex_id())

    def test_merge(self):
        ch1 = Chunk([0, 1], 1, self.gm)
        ch2 = Chunk([2, 3, 4, 5, 6], 2, self.gm)

        ch1.merge(ch2)

        self.assertEqual(7, ch1.length())
        self.assertEqual(ch1.start_vertex_id(), self.gm.g.vertex(0))
        self.assertEqual(ch1.end_vertex_id(), self.gm.g.vertex(6))
