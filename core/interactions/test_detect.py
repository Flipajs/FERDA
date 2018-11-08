from __future__ import print_function
import unittest

import numpy as np

from core.interactions.detect import InteractionDetector
from core.project.project import Project
from utils.video_manager import get_auto_video_manager


class InteractionDetectorTestCase(unittest.TestCase):
    def setUp(self):
        project_file = '../projects/2_temp/180810_2359_Cam1_clip_fixed_cardinality'
        # it = Interactions()
        # it._load_project(project_file)
        # it._init_regions()
        # self.project = it._project
        self.project = Project()
        self.project.load(project_file)
        self.vm = get_auto_video_manager(self.project)
        self.gm = self.project.gm
        self.rm = self.project.rm
        self.im = self.project.img_manager
        self.cm = self.project.chm
        self.tracklets_multi = [t for t in self.cm.chunk_gen() if t.is_multi()]
        self.tracklets_two = [t for t in self.tracklets_multi if t.get_cardinality(self.gm) == 2]
        self.tracklets_two.sort(lambda x, y: cmp(len(x), len(y)), reverse=True)  # descending by tracklet length
        self.detector = InteractionDetector('/home/matej/prace/ferda/experiments/180830_1637_single_50')
        # assert len(self.detector.ti.PREDICTED_PROPERTIES) == 3

    def test_detect(self):
        t = np.random.choice(self.tracklets_two)
        regions = list(t.r_gen(self.gm, self.rm))
        r = np.random.choice(regions)
        img = self.im.get_whole_img(r.frame())
        detections = self.detector.detect(img, r.centroid()[::-1])
        assert len(detections) == 6

    def test_single_detect(self):
        t = self.project.chm[3463]
        r = t.get_region(self.project.gm, -1)
        img = self.im.get_whole_img(r.frame() + 1)
        detections = self.detector.detect_single(img, self.detector.region_to_dict(r))


    def get_detections(self, tracklet):
        images = []
        detections = []
        for r in tracklet.r_gen(self.gm, self.rm):
            img = self.im.get_whole_img(r.frame())
            pred = self.detector.detect(img, r.centroid()[::-1])
            for obj_i in range(2):
                pred['{}_major'.format(obj_i)] = 60
                pred['{}_minor'.format(obj_i)] = 15
            images.append(img)
            detections.append(pred)
        return detections, images

    def test_track(self):
        t = np.random.choice(self.tracklets_two)
        detections, images = self.get_detections(t)
        tracks, confidence, costs = self.detector.track(detections)
        tracks['0_major'] = 60
        tracks['1_major'] = 60
        tracks['0_minor'] = 15
        tracks['1_minor'] = 15
        self.detector.write_interaction_movie(images, tracks, 'out/test.mp4')

    def test_solve(self):
        # t = np.random.choice(self.tracklets_two)
        t = self.tracklets_two[13]
        print(t.solve_interaction(self.detector, self.gm, self.rm, self.im))


if __name__ == '__main__':
    unittest.main(defaultTest='InteractionDetectorTestCase.test_single_detect')
