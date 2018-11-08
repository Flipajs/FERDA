from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from builtins import range
from past.utils import old_div
import multiprocessing
from utils.video_manager import get_auto_video_manager
from core.project.project import Project
import time
from utils.img import img_saturation_coef
from functools import partial
from core.id_detection.feature_manager import FeatureManager
from core.id_detection.features import get_colornames_hists_saturated
import cv2
import numpy as np
from libs import img_features

def video_read(start=0, len=200):
    print(start, len)
    p = Project()
    p.video_paths = ['/Volumes/Transcend/Dropbox/FERDA/Cam1_clip.avi']

    vm = get_auto_video_manager(p)
    vm.get_frame(start)
    for i in range(len):
        img = vm.next_frame()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        f1 = img_features.colornames_descriptor(img_hsv, pyramid_levels=1)
        # get_colornames_hists_saturated()

if __name__ == '__main__':
    cc = multiprocessing.cpu_count()
    len_ = 100

    t = time.time()

    # important, one can enable it inside process by setting it to -1 but on some systems it breaks
    cv2.setNumThreads(0)

    frames = [f for f in range(0, len_*cc, len_)]
    # should work for python 3
    # multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(cc)
    p = multiprocessing.Process()
    pool.map(partial(video_read, len=len_), frames)

    pool.close()
    pool.join()

    t1 = time.time() - t

    t = time.time()
    video_read(0, cc*len_)
    t2 = time.time() - t
    print("MP time: {:.2f}, SP time: {:.2f}, ratio: {:.2%}".format(t1, t2, old_div(t2,t1)))
