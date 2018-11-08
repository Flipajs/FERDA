from __future__ import unicode_literals
import unittest

import matplotlib.pylab as plt
import pandas as pd

from core.interactions.visualization import plot_interaction


class InteractionResultsTestCase(unittest.TestCase):
    def test_plot_interaction(self):
        pred = pd.DataFrame([{'0_x': 10, '0_y': 20, '0_angle_deg_cw': 20, '0_minor': 3, '0_major': 8}])
        gt = pd.DataFrame([{'0_x': 14, '0_y': 21, '0_angle_deg_cw': 25, '0_minor': 4, '0_major': 10}])
        plot_interaction(1, pred.to_records(), gt.to_records())
        plt.xlim(0, 30)
        plt.ylim(0, 30)
        # plt.show()





