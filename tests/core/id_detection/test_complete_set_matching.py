import unittest
from numpy.testing import assert_array_equal
from core.id_detection.complete_set_matching import get_csm
from core.project.project import Project
from mock import MagicMock
import copy
import numpy as np


class MockTracklet(object):
    def __init__(self, p, n):
        self.P = p
        self.N = n

    def get_track_id(self):
        return next(iter(self.P))

    def is_only_one_id_assigned(self, foo):
        return True


class CompleteSetMatchingTestCase(unittest.TestCase):
    def setUp(self):
        self.p = Project('test/project/Sowbug3_cut_300_frames')
        self.csm = get_csm(self.p)
        # {key: t.get_interval() for key, t in self.csm.tracks_obj.iteritems()}
        # {0: [IntInterval('[0, 300]')],
        #  1: [IntInterval('[0, 300]')],
        #  2: [IntInterval('[243, 252]')],
        #  3: [IntInterval('[0, 193]')],
        #  4: [IntInterval('[241, 242]')],
        #  5: [IntInterval('[259, 300]')],
        #  6: [IntInterval('[243, 258]')],
        #  7: [IntInterval('[241, 252]')],
        #  8: [IntInterval('[232, 242]')],
        #  9: [IntInterval('[232, 240]')],
        #  10: [IntInterval('[194, 231]')],
        #  11: [IntInterval('[194, 231]')],
        #  12: [IntInterval('[142, 240]')],
        #  13: [IntInterval('[142, 193]')],
        #  14: [IntInterval('[259, 300]')],
        #  15: [IntInterval('[253, 300]')],
        #  16: [IntInterval('[253, 258]')]}

    def test_find_track_cs(self):
        css = self.csm.find_track_cs()
        pass

    def test_remap_ids_from_0(self):
        # single P members, overlapping old and new id (0), missing support for id 60
        tracklets = [
            MockTracklet({10}, {20, 0, 40, 50, 60}),
            MockTracklet({20}, {10, 0, 40, 50, 60}),
            MockTracklet({20}, {10, 0, 40, 50, 60}),
            MockTracklet({0}, {10, 20, 40, 50, 60}),
            MockTracklet({0}, {10, 20, 40, 50, 60}),
            MockTracklet({0}, {10, 20, 40, 50, 60}),
            MockTracklet({40}, {10, 20, 0, 50, 60}),
            MockTracklet({40}, {10, 20, 0, 50, 60}),
            MockTracklet({40}, {10, 20, 0, 50, 60}),
            MockTracklet({40}, {10, 20, 0, 50, 60}),
            MockTracklet({50}, {10, 20, 0, 40, 60}),
            MockTracklet({50}, {10, 20, 0, 40, 60}),
            MockTracklet({50}, {10, 20, 0, 40, 60}),
            MockTracklet({50}, {10, 20, 0, 40, 60}),
            MockTracklet({50}, {10, 20, 0, 40, 60}),
        ]
        self.p.chm.chunk_gen = MagicMock(return_value=copy.deepcopy(tracklets))
        self.csm.remap_ids_from_0({10: 1, 20: 2, 0: 3, 40: 4, 50: 5})
        tracklets_new = self.p.chm.chunk_gen()
        self.assertEqual(len(tracklets), len(tracklets_new))
        self.assertEqual(tracklets_new[0].get_track_id(), 4)
        self.assertEqual(tracklets_new[1].get_track_id(), 3)
        self.assertEqual(tracklets_new[3].get_track_id(), 2)
        self.assertEqual(tracklets_new[6].get_track_id(), 1)
        self.assertEqual(tracklets_new[10].get_track_id(), 0)
        assert_array_equal(list(tracklets_new[0].N), [0, 1, 2, 3, 5])
        assert_array_equal(list(tracklets_new[-1].N), [1, 2, 3, 4, 5])

        # multiple P members
        tracklets = [
            MockTracklet({10, 20}, {0, 40, 50, 60}),
            MockTracklet({20}, {10, 0, 40, 50, 60}),
            MockTracklet({20}, {10, 0, 40, 50, 60}),
            MockTracklet({0}, {10, 20, 40, 50, 60}),
            MockTracklet({0}, {10, 20, 40, 50, 60}),
            MockTracklet({0}, {10, 20, 40, 50, 60}),
            MockTracklet({40}, {10, 20, 0, 50, 60}),
            MockTracklet({40}, {10, 20, 0, 50, 60}),
            MockTracklet({40}, {10, 20, 0, 50, 60}),
            MockTracklet({40}, {10, 20, 0, 50, 60}),
            MockTracklet({50}, {10, 20, 0, 40, 60}),
            MockTracklet({50}, {10, 20, 0, 40, 60}),
            MockTracklet({50}, {10, 20, 0, 40, 60}),
            MockTracklet({50}, {10, 20, 0, 40, 60}),
            MockTracklet({50}, {10, 20, 0, 40, 60}),
        ]
        self.p.chm.chunk_gen = MagicMock(return_value=copy.deepcopy(tracklets))
        self.csm.remap_ids_from_0({10: 1, 20: 2, 0: 3, 40: 4, 50: 5})
        tracklets_new = self.p.chm.chunk_gen()
        self.assertEqual(len(tracklets), len(tracklets_new))
        self.assertEqual(tracklets_new[1].get_track_id(), 3)
        self.assertEqual(tracklets_new[3].get_track_id(), 2)
        self.assertEqual(tracklets_new[6].get_track_id(), 1)
        self.assertEqual(tracklets_new[10].get_track_id(), 0)
        assert_array_equal(list(tracklets_new[0].P), [3, 4])
        assert_array_equal(list(tracklets_new[0].N), [0, 1, 2, 5])
        assert_array_equal(list(tracklets_new[-1].N), [1, 2, 3, 4, 5])

    def test_get_overlap_matrix(self):
        self.csm.find_track_cs()

        # not overlapping
        cs1 = [3]
        #  3: [IntInterval('[0, 193]')],
        cs2 = [9, 10]
        #  9: [IntInterval('[232, 240]')],
        #  10: [IntInterval('[194, 231]')],
        P = self.csm.get_overlap_matrix(cs1, cs2)
        assert_array_equal(P, [[1, 1]])

        # 1 and 3 overlapping
        cs1 = [0, 2]
        # 0: [IntInterval('[0, 300]')],
        # 2: [IntInterval('[243, 252]')],
        cs2 = [15, 16]
        #  15: [IntInterval('[253, 300]')],
        #  16: [IntInterval('[253, 258]')]}

        P = self.csm.get_overlap_matrix(cs1, cs2)
        assert_array_equal(P, [[0, 0],
                               [1, 1]])

    # def test_cs2cs_matching_prototypes_and_spatial(self):
    #     self.csm.cs2cs_matching_prototypes_and_spatial()

