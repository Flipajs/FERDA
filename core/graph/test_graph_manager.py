import unittest
from core.graph.graph_manager import GraphManager
from core.project.project import dummy_project
from core.region.region import Region


class TestGraphManager(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.p = dummy_project()

        self.frames_ = [8, 102, 3, 3, 4, 1, 0, 90, 0, 12]
        self.regions_ = [Region() for i in range(10)]
        for r, f, i in zip(self.regions_, self.frames_, range(len(self.regions_))):
            self.p.rm.add(r)
            r.frame_ = f
            r.id_ = i + 1

    def setUp(self):
        self.gm = GraphManager(self.p, None)

    def test_add_vertex_num_of_vertices(self):
        r = Region()

        num_vertices_before = self.gm.g.num_vertices()
        self.gm.add_vertex(r)
        self.assertEqual(num_vertices_before+1, self.gm.g.num_vertices(), "Unexpected number of vertices in graph manager.")

    def test_range_update(self):
        r10 = Region()
        r10.frame_ = 10

        r9 = Region()
        r9.frame_ = 9

        r11 = Region()
        r11.frame_ = 11

        r12 = Region()
        r12.frame_ = 12

        self.gm.add_vertex(r10)
        self.assertEqual(10, self.gm.start_t, "The frame range was not updated!")

        self.gm.add_vertex(r12)
        self.assertEqual(10, self.gm.start_t, "The start_t should stay = 10!")
        self.assertEqual(12, self.gm.end_t, "The frame range was not updated!")

        self.gm.add_vertex(r9)
        self.assertEqual(12, self.gm.end_t, "The end_t should stay = 11")
        self.assertEqual(9, self.gm.start_t, "The start_t was not updated!")

        self.gm.add_vertex(r11)
        self.assertEqual(9, self.gm.start_t, "The start_t should stay == 9")
        self.assertEqual(12, self.gm.end_t, "The end_t should stay == 12")


    def test_add_multiple_vertices(self):
        num_before = self.gm.g.num_vertices()
        self.gm.add_vertices(self.regions_)

        self.assertEqual(self.gm.g.num_vertices(), num_before+len(self.regions_), "Unexpected number of vertices after adding multiple vertices")

        self.assertEqual(self.gm.start_t, min(self.frames_), "start_t should be equal to min of frames")
        self.assertEqual(self.gm.end_t, max(self.frames_), "end_t should be equal to max of frames")


    def test_remove_vertices(self):
        v_id = 3

        self.gm.add_vertices(self.regions_)

        num_before = self.gm.g.num_vertices()
        v_ = self.gm.g.vertex(v_id)
        self.gm.remove_vertex(v_id)

        is_active = self.gm.g.vp['active'][v_]

        self.assertEqual(False, is_active, "The vertex must have active flag = 0 once it was removed.")


    def test_start_nodes(self):
        self.gm.add_vertices(self.regions_)

        vertices = self.gm.start_nodes()

        self.assertEqual(2, len(vertices), "There should be 2 starting vertices")
        self.assertEqual(0, self.gm.region(vertices[0]).frame_)
        self.assertEqual(0, self.gm.region(vertices[1]).frame_)

    def test_end_nodes(self):
        self.gm.add_vertices(self.regions_)

        vertices = self.gm.end_nodes()

        self.assertEqual(1, len(vertices))
        self.assertEqual(102, self.gm.region(vertices[0]).frame_)
