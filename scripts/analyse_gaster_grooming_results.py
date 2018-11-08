from __future__ import print_function
from __future__ import unicode_literals
from builtins import range
from core.project.project import Project
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils.img import get_safe_selection
import cv2
from tqdm import tqdm

WD_POST = '/Volumes/Seagate Expansion Drive/HH1_POST'
OUT_DIR = '/Users/flipajs/Documents/dev/ferda/scripts/gaster_grooming_out/imgs'
MARGIN = 1.25
major_axis = 36

offset = major_axis * MARGIN


print("Loading...")
p = Project()
p.load(WD_POST)

with open('/Users/flipajs/Documents/dev/ferda/scripts/gaster_grooming_out/HH1_post_predictions.pkl') as f:
    (predictions, rids) = pickle.load(f)

rids = np.array(rids)
print(predictions.shape)
print(rids.shape)

print(np.sum(predictions == 1))

frames = []

for rid in tqdm(list(rids[predictions == 1])):
    r = p.rm[rid]
    frames.append(r.frame())
    # print r.frame()

    # img = p.img_manager.get_whole_img(r.frame())

    # y, x = r.centroid()
    # crop = get_safe_selection(img, y - offset, x - offset, 2 * offset, 2 * offset)
    # cv2.imwrite(OUT_DIR + '/' + str(r.id()) + '.jpg', crop,
    #             [int(cv2.IMWRITE_JPEG_QUALITY), 95])


frames = sorted(frames)
last_f = -100

gt_frames = []
data= [(21728,59),
(22011,19),
(54200,136),
(66769,149),
(69557,37),
(71859,710),
(76103,142),
(6804,8),
(38345,143),
(49193,179),
(57710,14),
(65794,105),
(13825,194),
(20472,313),
(27794,186),
(34437,141),
(56099,142),
(60981,265),
(71806,190),
(46048,127),
(49368,135)]

for frame, length in data:
    for f in range(frame, frame+length):
        gt_frames.append(f)

# frames = set(frames)

plt.scatter(frames, np.ones((len(list(frames)), 1)), s=5, c='r')
plt.hold(True)
plt.scatter(gt_frames, np.ones((len(list(gt_frames)), 1)), s=1, c='b')
plt.grid()
plt.show()

frames = set((frames))
gt_frames = set((gt_frames))
print("num frames {} num gt{}".format(frames.__len__(), gt_frames.__len__()))
print("intersection: {}".format((frames.intersection(gt_frames)).__len__()))
print("gt-frames: {}".format((gt_frames.difference(frames)).__len__()))
print("frames - gs: {}".format((frames.difference(gt_frames)).__len__()))



length = 0
for f in frames:
    if f != last_f+1:
        print("{} -> {}".format(f-length, f))
        last_f = f
