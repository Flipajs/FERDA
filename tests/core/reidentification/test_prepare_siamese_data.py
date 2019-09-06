import unittest
from core.reidentification.prepare_siamese_data import generate_reidentification_training_data, DEFAULT_PARAMETERS
from utils.misc import makedirs


class PrepareSiameseDataTestCase(unittest.TestCase):
    def test_generate_reidentification_training_data(self):
        out_dir = 'out/reidentification'
        makedirs(out_dir)
        params = DEFAULT_PARAMETERS.copy()
        params['num_examples'] = 10
        generate_reidentification_training_data('test/project/Sowbug3_cut_300_frames', out_dir, params)


if __name__ == '__main__':
    unittest.main()
