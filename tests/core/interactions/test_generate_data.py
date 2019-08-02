import unittest
import shutil
from os.path import join
import core.interactions.generate_data


class DataGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        self.dg = core.interactions.generate_data.DataGenerator()
        self.project_dir = 'test/project/Sowbug3_cut_300_frames'
        self.gt_filename = 'data/GT/Sowbug3_cut.txt'
        self.out_dir = 'test/out/DataGeneratorTestCase'
        try:
            shutil.rmtree(self.out_dir)
        except:
            pass

    def init(self):
        self.dg._load_project(self.project_dir)
        self.dg._init_regions()
        from core.bg_model.median_intensity import MedianIntensity
        self.dg.bg_model = MedianIntensity(self.dg._project)
        self.dg.bg_model.compute_model()

    def test_write_regions_for_testing(self):
        self.dg.write_regions_for_testing(self.project_dir, 10, self.out_dir)

    def test_write_detections(self):
        self.dg.write_detections(self.project_dir, (0, 10), self.out_dir)

    # def write_annotated_blobs_groundtruth(self):
    #     self.dg.write_annotated_blobs_groundtruth(self.project_dir, )

    # def test_write_segmentation_data(self):
    #     self.dg.write_segmentation_data(self.project_dir, 10, join(self.out_dir, 'test_write_segmentation_data.h5'))

    # def test_write_detection_data(self):
    #     self.dg.write_detection_data(self.project_dir, 2, self.out_dir, image_format='file')
    #     # this fails: gt_filename=self.gt_filename
        # this fails: image_format='hdf5'

    # def test__synthetize(self):
    #     self.init()
    #     n_objects = 2
    #     single_regions = [item for sublist in self.dg._single.values() for item in sublist]
    #     regions = np.random.choice(single_regions, n_objects * 1)
    #     images = [self.dg._video.get_frame(r.frame()) for r in regions]
    #     masks = [np.zeros(shape=images[0].shape[:2], dtype=images[0].dtype) for _ in regions]
    #     for m, r in zip(masks, regions):
    #         r.draw_mask(m)
    #         # plt.figure()
    #         # plt.imshow(m)
    #         # plt.show()
    #
    #     img_synthetic, mask, centers_xy, main_axis_angles_rad = \
    #         self.dg._Interactions__synthetize(regions, [np.radians(0)], [np.radians(45)], [5], masks)
    #     # plt.imshow(mask)
    #     # plt.show()

    # def test_write_synthetized_interactions(self):
    #     self.dg.write_synthetized_interactions('/home/matej/prace/ferda/projects/camera1_10-15/', 10, 2,
    #                                            join('out', 'write_synthetized_interactions.csv'),
    #                                            out_hdf5=join('out', 'images.h5'), hdf5_dataset_name='train',
    #                                            out_image_dir='out', write_masks=True)


if __name__ == '__main__':
    unittest.main()
