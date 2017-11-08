import numpy as np
import sys, os, re, random
import h5py
import string
import tqdm
from imageio import imread

# OUT_DIR = '/Users/flipajs/Documents/wd/FERDA/CNN_HH1_3'
OUT_DIR = '/Volumes/Seagate Expansion Drive/HH1_PRE_upper_thr/CNN_HH1_train'
NUM_ANIMALS = 6

# NUM_EXAMPLES x NUM_A
NUM_EXAMPLES = 1000
TRAIN_TEST_RATIO = 0.1

def repre(project):
    repre = {}
    repre['yellow'] = [28664, 38, 40]
    repre['green'] = [1, 32, 18558, 17678, 14138, 6996, 20254, 27530, 32710, 7870, 53, 30187]
    repre['orange'] = [20, 32, 43, 47, 2604, 18914, 25770, 60, 61, 32707]
    repre['red'] = [20187, 22403, 5185, 5477, 26798, 23095, 28403, 23345, 13443, 28932]
    repre['blue'] = [25021, 29489, 21416, 19292, 22633, 21292, 191, 32167, 2532, 11778, 27151]
    repre['purple'] = [6001]

    for key, vals in repre.iteritems():
        s = 0
        for id_ in vals:
            try:
                s += len(project.chm[id_])
            except:
                print id_

        print key, s

    return repre


if __name__ == '__main__':
    wd = '/Volumes/Seagate Expansion Drive/HH1_PRE_upper_thr/HH1_PRE_upper_thr.fproj'
    from core.project.project import Project

    p = Project()
    p.load(wd)

    repre = repre(p)

    split_idx = int((1-TRAIN_TEST_RATIO) * NUM_EXAMPLES)
    print "SPLIT: ", split_idx
    imgs = {}

    from utils.img import get_safe_selection
    import cv2

    MARGIN = 1.25
    major_axis = 36

    offset = major_axis * MARGIN

    id = 0

    try:
        os.mkdir(OUT_DIR)
    except:
        pass

    from utils.video_manager import get_auto_video_manager
    vm = get_auto_video_manager(p)

    if False:
        for key, tracklets in tqdm.tqdm(repre.iteritems()):
            print id, key
            try:
                os.mkdir(OUT_DIR+'/'+str(id))
            except:
                pass

            i = 0
            imgs[key] = []
            for t in tracklets:
                for r_id in p.chm[t].rid_gen(p.gm):
                    r = p.rm[r_id]
                    # img = vm.seek_frame(r.frame())
                    img = p.img_manager.get_whole_img(r.frame())

                    y, x = r.centroid()
                    crop = get_safe_selection(img, y - offset, x - offset, 2 * offset, 2 * offset)
                    cv2.imwrite(OUT_DIR + '/' + str(id) + '/' + str(r.id()) + '.jpg', crop,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    i += 1

                    # if i == NUM_EXAMPLES:
                    #     break

                # if i == NUM_EXAMPLES:
                #     break

            id += 1

    print "Training examplest exported..."
    print "Exporting test examples: "

    try:
        os.mkdir(OUT_DIR + '/test')
    except:
        pass

    for t in tqdm.tqdm(p.chm.chunk_gen(), len(p.chm)):
        if not t.is_single():
            continue

        for r_id in t.rid_gen(p.gm):
            r = p.rm[r_id]
            img = p.img_manager.get_whole_img(r.frame())

            y, x = r.centroid()
            crop = get_safe_selection(img, y - offset, x - offset, 2 * offset, 2 * offset)
            cv2.imwrite(OUT_DIR + '/test/' + str(r.id()) + '.jpg', crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])

