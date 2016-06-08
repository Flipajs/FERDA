import numpy as np
from processing import transform_img_
from hist_3d import ColorHist3d
from processing import get_colormarks, match_cms_region, filter_cms
import cv2
from math import ceil
from utils.img import get_safe_selection


class ColormarksModel:
    def __init__(self):
        self.num_bins1 = 16
        self.num_bins2 = 16
        self.num_bins3 = 16

        self.im_space = 'irb'
        self.hist3d = None
        self.num_colors = -1
        self.num_bins_v = None

        self.irgb_img_cache = {}
        self.colors_ = {}

    def compute_model(self, main_img, color_samples):
        self.num_bins_v = np.array([self.num_bins1, self.num_bins2, self.num_bins3], dtype=np.float)

        img_t = transform_img_(main_img, self)

        self.num_colors = len(color_samples)

        self.hist3d = ColorHist3d(img_t.copy(), self.num_colors,
                                  num_bins1=self.num_bins1, num_bins2=self.num_bins2, num_bins3=self.num_bins3,
                                  theta=0.3, epsilon=0.9)

        for (picked_pxs, all_pxs, mean_color), c_id in zip(color_samples, range(1, len(color_samples)+1)):
            self.hist3d.remove_bg(all_pxs)
            self.hist3d.add_color(picked_pxs, c_id)
            self.colors_[c_id] = mean_color

        self.hist3d.assign_labels()

    def get_labelling(self, pos):
        labels = self.hist3d.hist_labels_[pos[:, :, 0], pos[:, :, 1], pos[:, :, 2]]

        return labels

    def get_bounding_box(self, r, project):
        # TODO set this...:
        border_percent = 1.3

        frame = r.frame()
        if frame in self.irgb_img_cache:
            img = self.irgb_img_cache[frame]
        else:
            from utils.video_manager import get_auto_video_manager
            vid_m = get_auto_video_manager(project)

            img = project.img_manager.get_whole_img(frame)

            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform_img_(img, self)

            self.irgb_img_cache[frame] = img

        roi = r.roi()

        height2 = int(ceil((roi.height() * border_percent) / 2.0))
        width2 = int(ceil((roi.width() * border_percent) / 2.0))
        x = r.centroid()[1] - width2
        y = r.centroid()[0] - height2

        bb = get_safe_selection(img, y, x, height2*2, width2*2)

        return bb, np.array([y, x])

    def find_colormarks(self, project, region):
        bb, offset = self.get_bounding_box(region, project)

        cms = get_colormarks(bb, self, min_a=20)

        # print region.id_
        #
        # im_ = np.zeros((bb.shape[0], bb.shape[1]), dtype=np.uint8)
        # for pts, label in cms:
        #     for pt in pts:
        #         im_[pt[0], pt[1]] = 255
        #         # bb[pt[0], pt[1], 0:2] = 0
        #
        #     cv2.imshow('bb', bb)
        #     cv2.imshow('im_', im_)
        #
        #     print len(pts)
        #     cv2.waitKey(0)

        matches = match_cms_region(filter_cms(cms), region, offset)

        # order by size:
        matches = sorted(matches, key=lambda x: -len(x[0]))

        return matches

    def assign_colormarks(self, project, regions):
        for r in regions:
            r.colormarks = self.find_colormarks(project, r)