from __future__ import print_function
from __future__ import unicode_literals
from builtins import str
import pickle
import numpy as np
from imageio import imread
from utils.img import get_safe_selection
import cv2
from core.project.project import Project
from utils.video_manager import get_auto_video_manager

import matplotlib.pyplot as plt


wd = '/Volumes/Seagate Expansion Drive/HH1_POST'


p = Project()
p.load(wd)


MARGIN = 1.25
major_axis = 36

offset = major_axis * MARGIN

#####
with open(wd + '/softmax_results_map.pkl') as f:
    d = pickle.load(f)

x = []
for val in d.values():
    x.append(val)

x = np.array(x)
plt.hist(x[:, -1])
plt.show()

from tqdm import tqdm

for id, val in tqdm(iter(d.items())):

    if val[-1] < 0.5:
        continue

    r = p.rm[id]

    frame = r.frame()

    t_id = None
    for t in p.chm.tracklets_in_frame(frame):
        if id in t.rid_gen(p.gm):
            t_id = t.id()
        break

    if t_id is None:
        continue

    t = p.chm[t_id]
    if not t.is_single():
        continue


    img = p.img_manager.get_whole_img(r.frame())

    y, x = r.centroid()
    crop = get_safe_selection(img, y - offset, x - offset, 2 * offset, 2 * offset)

    print("saving ", id)
    cv2.imwrite('/Users/flipajs/Documents/dev/ferda/scripts/out4/' + str(id) + '_'+"{:.2f}".format(val[-1])+'.jpg', crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
