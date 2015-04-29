__author__ = 'fnaiser'

from sklearn import svm
from utils.drawing.points import get_contour
import numpy as np


class Antlikeness():
    def __init__(self):
        self.use_area_cont_ratio = True
        self.use_min_intensity_percentile = True
        self.use_min_intensity = True
        self.use_margin = True

        self.min_intensity_percentile = 3
        self.use_probability = True
        self.svm_model = svm.SVC(kernel='linear', probability=self.use_probability)

    def learn(self, f_regions, classes, imgs_gray=None):
        X = []
        if imgs_gray:
            for f in f_regions:
                img_gray = imgs_gray[f]
                for r in f_regions[f]:
                    X.append(self.get_x(r, img_gray))
        else:
            self.use_min_intensity_percentile = False
            self.use_min_intensity = True
            for f in f_regions:
                for r in f_regions[f]:
                    X.append(self.get_x(r))

        self.svm_model.fit(X, classes)

    def get_x(self, r, img_gray=None):
        x = []
        if self.use_margin:
            x.append(r.margin_)

        if self.use_area_cont_ratio:
            cl = len(get_contour(r.pts()))
            x.append(cl/r.area()**0.5)

        if self.use_min_intensity_percentile:
            intensities = img_gray[r.pts()[:, 0], r.pts()[:, 1]]
            min_i_percentile = np.percentile(intensities, self.min_intensity_percentile)
            x.append(min_i_percentile)

        if self.use_min_intensity:
            x.append(r.min_intensity_)

        return x

    def get_prob(self, region, img_gray=None):
        x = self.get_x(region, img_gray)

        prob = self.svm_model.predict_proba([x])

        return prob[0]