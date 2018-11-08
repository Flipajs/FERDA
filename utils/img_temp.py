from __future__ import division
from __future__ import unicode_literals
# this should be moved back to img.py after get_settings() PyQt dependency is removed
from past.utils import old_div
import numpy as np
from gui.gui_utils import get_settings


def get_igbr_normalised(im):
    igbr = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.double)

    igbr[:, :, 0] = np.sum(im, axis=2) + 1
    igbr[:, :, 1] = old_div(im[:, :, 0], igbr[:, :, 0])
    igbr[:, :, 2] = old_div(im[:, :, 1], igbr[:, :, 0])
    igbr[:, :, 3] = old_div(im[:, :, 2], igbr[:, :, 0])

    i_norm = (old_div(1, get_settings('igbr_i_weight', float))) * get_settings('igbr_i_norm', float)
    igbr[:, :, 0] = old_div(igbr[:, :, 0], i_norm)

    return igbr