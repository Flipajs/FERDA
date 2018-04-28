import unittest
import numpy as np
import scripts.interactions as interactions
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from os.path import join
import core.region.transformableregion as tr
import matplotlib.pylab as plt

PROJECT_DIR = '/home/matej/prace/ferda/projects/camera1_10-15/'
VIDEO_FILE = '/datagrid/ferda/data/ants_ist/camera_1/camera_1_ss00:10:00_t00:05:00.mp4'


class InteractionsTestCase(unittest.TestCase):
    def setUp(self):
        self.intr = interactions.Interactions()

    def init(self):
        self.intr._load_project(PROJECT_DIR, video_file=VIDEO_FILE)
        self.intr._init_regions()
        from core.bg_model.median_intensity import MedianIntensity
        self.intr._bg = MedianIntensity(self.intr._project)
        self.intr._bg.compute_model()

    def test__synthetize(self):
        self.init()
        n_objects = 2
        single_regions = [item for sublist in self.intr._single.values() for item in sublist]
        regions = np.random.choice(single_regions, n_objects * 1)
        images = [self.intr._video.get_frame(r.frame()) for r in regions]
        masks = [np.zeros(shape=images[0].shape[:2], dtype=images[0].dtype) for _ in regions]
        for m, r in zip(masks, regions):
            r.draw_mask(m)
            # plt.figure()
            # plt.imshow(m)
            # plt.show()

        img_synthetic, mask, centers_xy, main_axis_angles_rad = \
            self.intr._Interactions__synthetize(regions, [np.radians(0)], [np.radians(45)], [5], masks)
        # plt.imshow(mask)
        # plt.show()

    def test_write_synthetized_interactions(self):
        self.intr.write_synthetized_interactions('/home/matej/prace/ferda/projects/camera1_10-15/', 10, 2,
                                                 join('out', 'write_synthetized_interactions.csv'),
                                                 out_hdf5=join('out', 'images.h5'), hdf5_dataset_name='train',
                                                 out_image_dir='out', write_masks=True)


if __name__ == '__main__':
    unittest.main()
