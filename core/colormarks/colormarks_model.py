import numpy as np
from processing import transform_img_
from hist_3d import ColorHist3d

class ColormarksModel:
    def __init__(self):
        self.num_bins1 = 16
        self.num_bins2 = 16
        self.num_bins3 = 16

        self.im_space = 'irb'
        self.hist3d = None
        self.num_colors = -1

    def compute_model(self, main_img, color_samples):
        self.num_bins_v = np.array([self.num_bins1, self.num_bins2, self.num_bins3], dtype=np.float)

        img_t = transform_img_(main_img, self)

        self.num_colors = len(color_samples)

        self.hist3d = ColorHist3d(img_t.copy(), self.num_colors,
                                  num_bins1=self.num_bins1, num_bins2=self.num_bins2, num_bins3=self.num_bins3,
                                  theta=0.3, epsilon=0.9)

        for (picked_pxs, all_pxs), c_id in zip(color_samples, range(len(color_samples))):
            self.hist3d.remove_bg(all_pxs)
            self.hist3d.add_color(picked_pxs, c_id)

        self.hist3d.assign_labels()

    def get_labelling(self, pos):
        labels = self.hist3d.hist_labels_[pos[:, :, 0], pos[:, :, 1], pos[:, :, 2]]

        return labels