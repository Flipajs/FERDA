import unittest
from core.project.project import Project
import core.global_cardinality_classifier as gcc


class GTTestCase(unittest.TestCase):
    def setUp(self):
        self.p = Project('test/project/Sowbug3_cut_300_frames')

    def test_get_tracklet_cardinalities(self):
        cardinalities = gcc.get_tracklet_cardinalities(self.p)
        pass

    def test_fill_tracklet_cardinalites(self):
        pre = [(None, t.segmentation_class) for t in self.p.chm.chunk_gen()]
        gcc.fill_tracklet_cardinalites(self.p)
        post = [(t.cardinality, t.segmentation_class) for t in self.p.chm.chunk_gen()]
        pass
