__author__ = 'fnaiser'

import numpy as np


class ClassesStats(object):
    def __init__(self):
        self.area_median = None
        self.major_axis_median = None
        self.margin_median = None

    def compute_stats(self, regions, classes):
        areas = []
        major_axes = []
        margins = []
        for r, c in zip(regions, classes):
            if not c:
                continue

            areas.append(r.area())
            major_axes.append(r.ellipse_major_axis_length() * 2)
            margins.append(r.margin_)

        areas = np.array(areas)
        major_axes = np.array(major_axes)
        margins = np.array(margins)

        self.area_median = np.median(areas)
        self.major_axis_median = np.median(major_axes)
        self.margin_median = np.median(margins)

