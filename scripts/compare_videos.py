from __future__ import print_function
import cv2
import time
from utils.video_manager import get_auto_video_manager
from core.project.project import Project

def measure(p, random_frames):
    vm = get_auto_video_manager(p)

    t_in_seq = time.time()
    for i in range(100):
        img = vm.next_frame()

    print("in seq: {}s".format(time.time() - t_in_seq))

    t_seek = time.time()
    for frame in random_frames:
        img = vm.get_frame(frame)

    print("seek : {}s".format(time.time() - t_seek))

p1 = Project()
p1.video_paths = ['/Volumes/Transcend/old_dropbox/FERDA/5Zebrafish_nocover_22min.avi']

p2 = Project()
p2.video_paths = ['/Users/flipajs/Downloads/Cam1_clip_slow.mp4']

import random
random_frames = random.sample(range(4500), 50)

print("Super fast")
measure(p1, random_frames)
# print("Slow")
# measure(p1, random_frames)