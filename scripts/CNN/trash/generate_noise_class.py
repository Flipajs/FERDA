import numpy as np
import sys, os, re, random
import h5py
import string
import tqdm
from imageio import imread


# OUT_DIR = '/Users/flipajs/Documents/wd/FERDA/CNN_HH1_3'
OUT_DIR = '/Volumes/Seagate Expansion Drive/CNN_HH1_train'
NUM_ANIMALS = 6

# NUM_EXAMPLES x NUM_A
NUM_EXAMPLES = 1000
TRAIN_TEST_RATIO = 0.1
BATCH_SIZE = 100


if __name__ == '__main__':
    # WD = sys.argv()
    wd = '/Volumes/Seagate Expansion Drive/HH1_PRE_upper_thr_/'
    from core.project.project import Project

    p = Project()
    p.load(wd)

    from utils.img import get_safe_selection
    import cv2

    MARGIN = 1.25
    major_axis = 36

    offset = major_axis * MARGIN

    id = 0

    from utils.video_manager import get_auto_video_manager
    vm = get_auto_video_manager(p)

    noise_id = 6
    try:
        os.mkdir(OUT_DIR + '/'+str(noise_id))
    except Exception as e:
        print e


    for i in tqdm.tqdm(range(200)):
        frame = random.randint(0, vm.total_frame_count()-1)

        for r in p.gm.regions_in_t(frame):
            xd = random.randint(9, 20) * random.sample([-1, 1], 1)[0]
            yd = random.randint(9, 20) * random.sample([-1, 1], 1)[0]

            img = p.img_manager.get_whole_img(frame)

            y, x = r.centroid()
            y += yd
            x += xd
            crop = get_safe_selection(img, y - offset, x - offset, 2 * offset, 2 * offset)
            cv2.imwrite(OUT_DIR + '/' + str(noise_id) + '/' + str(r.id()) + '.jpg', crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
