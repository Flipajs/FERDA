import numpy as np
import sys, os, re, random
import h5py
import string
import tqdm
from imageio import imread
from core.project.project import Project
from utils.img import get_safe_selection
from utils.video_manager import get_auto_video_manager
import time
from utils.img import apply_ellipse_mask
from core.graph.region_chunk import RegionChunk

ELLIPSE_DILATION = 10
MASK_SIGMA = 10

OUT_DIR = '/Volumes/Seagate Expansion Drive/CNN_HH1_POST'
WD = '/Volumes/Seagate Expansion Drive/HH1_POST'
BATCH_SIZE = 30000

def save_batch(batch_i, imgs, ids):
    global OUT_DIR
    print "Saving batch: ", batch_i

    imgs = np.array(imgs)
    imgs = imgs.astype('float32')
    imgs /= 255

    batch_s = str(batch_i)
    while len(batch_s) < 3:
        batch_s = "0" + batch_s

    with h5py.File(OUT_DIR + '/test/batch_' + batch_s + '_imgs.h5', 'w') as hf:
        hf.create_dataset("data", data=imgs)

    with h5py.File(OUT_DIR + '/test/batch_' + batch_s + '_ids.h5', 'w') as hf:
        hf.create_dataset("data", data=ids)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        WD = sys.argv[1]
        OUT_DIR = sys.argv[2]
        if len(sys.argv) > 3:
            BATCH_SIZE = string.atoi(sys.argv[3])

    p = Project()
    p.load(WD)

    MARGIN = 1.25
    major_axis = 36

    offset = major_axis * MARGIN

    id = 0

    try:
        os.mkdir(OUT_DIR)
    except:
        pass

    vm = get_auto_video_manager(p)

    print "Exporting test examples: "

    try:
        os.mkdir(OUT_DIR + '/test')
    except:
        pass

    p.img_manager.max_size_mb = 500

    batch_i = 0
    imgs = []
    ids = []

    print p.chm

    t_ = time.time()
    for frame in tqdm.tqdm(xrange(vm.total_frame_count())):
        for t in p.chm.tracklets_in_frame(frame):
            if t.is_single():
                rch = RegionChunk(t, p.gm, p.rm)
                r = rch.region_in_t(frame)

                img = p.img_manager.get_whole_img(r.frame())

                y, x = r.centroid()
                crop = get_safe_selection(img, y - offset, x - offset, 2 * offset, 2 * offset)
                crop = apply_ellipse_mask(r, crop, MASK_SIGMA, ELLIPSE_DILATION)

                imgs.append(crop)
                ids.append(r.id())
                if len(imgs) == BATCH_SIZE:
                    save_batch(batch_i, imgs, ids)

                    imgs = []
                    ids = []

                    batch_i += 1

                    print "TOTAL TIME: ", time.time() - t_

    if len(imgs):
        save_batch(batch_i, imgs, ids)