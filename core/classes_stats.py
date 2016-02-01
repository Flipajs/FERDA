__author__ = 'fnaiser'

import numpy as np
from core.antlikeness import Antlikeness

class ClassesStats():
    def __init__(self):
        self.area_median = -1
        self.major_axis_median = -1
        self.margin_median = -1
        self.antlikeness_svm = Antlikeness()

    def compute_stats(self, regions, classes):
        self.antlikeness_svm.learn(regions, classes)

        areas = []
        major_axes = []
        margins = []
        for r, c in zip(regions, classes):
            if not c:
                continue

            areas.append(r.area())
            major_axes.append(r.a_ * 2)
            margins.append(r.margin_)

        areas = np.array(areas)
        major_axes = np.array(major_axes)
        margins = np.array(margins)

        self.area_median = np.median(areas)
        self.major_axis_median = np.median(major_axes)
        self.margin_median = np.median(margins)

def dummy_classes_stats():
    from core.antlikeness import DummyAntlikeness

    cs = ClassesStats()
    cs.area_median = 100
    cs.major_axis_median = 5
    cs.margin_median = 5
    cs.antlikeness_svm = DummyAntlikeness()

    return cs
