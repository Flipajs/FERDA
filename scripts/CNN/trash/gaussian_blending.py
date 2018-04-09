import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from core.project.project import Project
from utils.img import get_safe_selection
from math import ceil

wd = '/Users/flipajs/Documents/wd/FERDA/Cam1'

p = Project()
p.load(wd)
MARGIN = 1.25
major_axis = 36
offset = major_axis * MARGIN

# IMPORTANT PARAMS
el_margin = 10
sigma = 10

def apply_ellipse_mask(r, im, sigma=10, ellipse_dilation=10):
    x = np.zeros((im.shape[0], im.shape[1]))

    deg = int(r.theta_ * 57.295)
    # angle of rotation of ellipse in anti-clockwise direction
    cv2.ellipse(x, (x.shape[0] / 2, x.shape[1] / 2),
                (int(ceil(r.a_)) + ellipse_dilation, int(ceil(r.b_)) + ellipse_dilation),
                -deg, 0, 360, 255, -1)

    y = ndimage.filters.gaussian_filter(x, sigma=sigma)
    y /= y.max()

    for i in range(3):
        im[:, :, i] = np.multiply(im[:, :, i].astype(np.float), y)

    return im

f, axs = plt.subplots(5, 10, tight_layout=True)
axs = axs.flatten()

id_offset = 50
for id_ in range(50):
    r = p.rm[id_+id_offset]
    img = p.img_manager.get_whole_img(r.frame())

    y, x = r.centroid()
    im = get_safe_selection(img, y - offset, x - offset, 2 * offset, 2 * offset)

    im = apply_ellipse_mask(r, im, sigma, el_margin)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    axs[id_].imshow(im)
    axs[id_].set_title(str(id_+id_offset))

plt.show()
