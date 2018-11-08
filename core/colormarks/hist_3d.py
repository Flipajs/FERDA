from __future__ import print_function
import numpy as np

class ColorHist3d():
    def __init__(self, im, num_colors, num_bins1=32, num_bins2=32, num_bins3=32, theta=0.1, epsilon=0.3):
        self.theta = theta
        self.epsilon = epsilon

        # TODO: 2x multiply num of bins
        self.num_bins1 = num_bins1
        self.num_bins2 = num_bins2
        self.num_bins3 = num_bins3

        self.num_bins_v = np.array([self.num_bins1, self.num_bins2, self.num_bins3], dtype=np.float)

        self.num_pxs = im.shape[0] * im.shape[1] * im.shape[2]
        self.num_colors = num_colors
        self.BG = 0

        pos = np.asarray(im / self.num_bins_v, dtype=np.int)

        # num_colors + 1 for background
        self.hist_ = np.zeros((self.num_bins1, self.num_bins2, self.num_bins3, num_colors + 1), dtype=np.int)
        self.hist_[:, :, :, self.BG] += 1

        self.hist_labels_ = np.zeros((self.num_bins1, self.num_bins2, self.num_bins3), dtype=np.int) + self.BG

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                p = pos[i, j]
                self.hist_[p[0], p[1], p[2], self.BG] += 1

    def swap_bg2color(self, pxs, color_id):
        pos = np.asarray(pxs / self.num_bins_v, dtype=np.int)
        for i in range(pxs.shape[0]):
            p = pos[i, :]

            if self.hist_[p[0], p[1], p[2], self.BG] > 1:
                self.hist_[p[0], p[1], p[2], self.BG] -= 1

            self.hist_[p[0], p[1], p[2], color_id] += 1

    def remove_bg(self, pxs):
        pos = np.asarray(pxs / self.num_bins_v, dtype=np.int)
        for i in range(pxs.shape[0]):
            p = pos[i, :]

            if self.hist_[p[0], p[1], p[2], self.BG] > 1:
                self.hist_[p[0], p[1], p[2], self.BG] -= 1

    def add_color(self, pxs, color_id):
        pos = np.asarray(pxs / self.num_bins_v, dtype=np.int)

        for i in range(pxs.shape[0]):
            p = pos[i, :]

            self.hist_[p[0], p[1], p[2], color_id] += 1

    def compute_p_fg(self):
        for i in range(self.num_bins1):
            for j in range(self.num_bins2):
                for k in range(self.num_bins3):
                    num_bg = self.hist_bg_[i, j, k]
                    num_fg = self.hist_fg_[i, j, k]
                    if num_bg + num_fg > 0:
                        self.p_fg_[i, j, k] = num_fg / float(num_bg + num_fg)
                        print(i, j, k, self.p_fg_[i, j, k])

    def get_p_k_x(self, k, x):
        a = self.hist_[x[0], x[1], x[2], k]
        n = np.sum(self.hist_[x[0], x[1], x[2], :])

        return a / float(n)

    def get_p_x_k(self, x, k):
        a = self.hist_[x[0], x[1], x[2], k]
        if a == 0:
            return 0.0

        n = np.sum(self.hist_[:, :, :, k])

        return a / float(n)

    def assign_labels(self):
        # skip bg
        for c_id in range(1, self.num_colors+1):
            sum_ = 0
            good_enough = []

            for i in range(self.num_bins1):
                for j in range(self.num_bins2):
                    for k in range(self.num_bins3):
                        pkx = self.get_p_k_x(c_id, [i, j, k])
                        pxk = self.get_p_x_k([i, j, k], c_id)

                        if pkx > self.theta:
                            good_enough.append((pxk, [i, j, k]))

            good_enough = sorted(good_enough, key=lambda x: -x[0])

            sum_ = 0
            for g in good_enough:
                self.hist_labels_[g[1][0], g[1][1], g[1][2]] = c_id
                sum_ += g[0]

                if sum_ > self.epsilon:
                    break

            print("C_ID DONE: ", c_id, sum_)