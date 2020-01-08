import unittest
from core.graph.track import Track
from core.graph.graph_manager import GraphManager
from core.project.project import Project
from intervals.interval import IntInterval
from core.region.region import Region
from functools import reduce


class TestTrack(unittest.TestCase):
    def setUp(self):
        self.p = Project('test/project/Sowbug3_cut_300_frames')
        """
        [(t.id_, t.get_interval()) for t in self.p.chm.tracklet_gen()]
        [(1, IntInterval('[0, 300]')),
         (2, IntInterval('[0, 300]')),
         (749, IntInterval('[243, 252]')),
         (4, IntInterval('[0, 141]')),
         (5, IntInterval('[0, 193]')),
         (737, IntInterval('[241, 242]')),
         (827, IntInterval('[259, 300]')),
         (747, IntInterval('[243, 258]')),
         (738, IntInterval('[241, 252]')),
         (694, IntInterval('[232, 242]')),
         (693, IntInterval('[232, 240]')),
         (502, IntInterval('[194, 231]')),
         (503, IntInterval('[194, 231]')),
         (346, IntInterval('[142, 240]')),
         (347, IntInterval('[142, 193]')),
         (828, IntInterval('[259, 300]')),
         (798, IntInterval('[253, 300]')),
         (799, IntInterval('[253, 258]'))]
        """
        self.track = Track([self.p.chm[i] for i in [4, 346]], self.p.gm)

    def test_get_interval(self):
        interval = self.track.get_interval()
        self.assertEqual(len(interval), 2)
        self.assertFalse(reduce(IntInterval.is_connected, self.track.get_interval()))

    def test_is_overlapping(self):
        t1 = Track([self.p.chm[i] for i in [5]], self.p.gm)
        self.assertEqual(t1.get_interval()[0].lower, 0)
        self.assertEqual(t1.get_interval()[0].upper, 193)
        t2 = Track([self.p.chm[i] for i in [749, 798]], self.p.gm)
        self.assertEqual(t2.get_interval()[0].lower, 243)
        self.assertEqual(t2.get_interval()[0].upper, 252)
        self.assertEqual(t2.get_interval()[1].lower, 253)
        self.assertEqual(t2.get_interval()[1].upper, 300)
        self.assertFalse(t1.is_overlapping(t2))
        self.assertFalse(t2.is_overlapping(t1))

        t1 = Track([self.p.chm[i] for i in [749]], self.p.gm)
        self.assertEqual(t1.get_interval()[0].lower, 243)
        self.assertEqual(t1.get_interval()[0].upper, 252)
        t2 = Track([self.p.chm[i] for i in [798]], self.p.gm)
        self.assertEqual(t2.get_interval()[0].lower, 253)
        self.assertEqual(t2.get_interval()[0].upper, 300)
        self.assertFalse(t1.is_overlapping(t2))
        self.assertFalse(t2.is_overlapping(t1))

        t1 = Track([self.p.chm[i] for i in [502]], self.p.gm)
        self.assertEqual(t1.get_interval()[0].lower, 194)
        self.assertEqual(t1.get_interval()[0].upper, 231)
        t2 = Track([self.p.chm[i] for i in [346]], self.p.gm)
        self.assertEqual(t2.get_interval()[0].lower, 142)
        self.assertEqual(t2.get_interval()[0].upper, 240)
        self.assertTrue(t1.is_overlapping(t2))
        self.assertTrue(t2.is_overlapping(t1))





