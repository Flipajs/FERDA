__author__ = 'fnaiser'


import numpy as np
from utils.video_manager import get_auto_video_manager
from utils.img import prepare_for_segmentation
from core.region.mser import get_filtered_regions

class Reduced:
    def __init__(self, region=None):
        if not region:
            self.centroid_ = None
            self.frame_ = -1
        else:
            self.centroid_ = region.centroid()
            self.frame_ = region.frame_

    def centroid(self):
        return np.copy(self.centroid_)

    def reconstruct(self, project):
        vid = get_auto_video_manager(project)
        img = vid.seek_frame(self.frame_)
        img = prepare_for_segmentation(img, project)
        msers = get_filtered_regions(img, project, self.frame_)

        best_match_d = np.inf
        region = None
        for r in msers:
            d = np.linalg.norm(r.centroid() - self.centroid())
            if best_match_d > d:
                best_match_d = d
                region = r

        return region
