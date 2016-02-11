import cv2
import cPickle as pickle
from core.graph.region_chunk import RegionChunk
from utils.img import get_safe_selection
from math import ceil
from processing import get_colormarks, match_cms_region, filter_cms
import matplotlib.pyplot as plt
import numpy as np
from scripts.irg_hist_demo import ColorHist3d
from scripts.irg_hist_demo import *
from processing import transform_img_


def analyse_chunk(ch, project, cm_model, sample_step):
    r_ch = RegionChunk(ch, project.gm, project.rm)

    ch_cms = {}
    for t in range(r_ch.start_frame(), r_ch.end_frame()+1, sample_step):
        r = r_ch[t - r_ch.start_frame()]

        # TODO: set some reasonable parameters
        bb = get_bounding_box(r, project, cm_model)

        cms = get_colormarks(bb, cm_model)
        for pts, label in cms:
            for pt in pts:
                bb[pt[0], pt[1], 0:2] = 255

        cv2.imshow('bb', bb)
        cv2.imshow('bb_t', transform_img_(bb, cm_model))
        cv2.waitKey(5)

        matches = match_cms_region(filter_cms(cms), r)

        ch_cms[t] = matches

    return ch_cms


def get_bounding_box(r, project, cm_model):
    # TODO set this...:
    border_percent = 1.3

    frame = r.frame()
    img = project.img_manager.get_whole_img(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform_img_(img, cm_model)

    roi = r.roi()

    height2 = int(ceil((roi.height() * border_percent) / 2.0))
    width2 = int(ceil((roi.width() * border_percent) / 2.0))
    x = r.centroid()[1] - width2
    y = r.centroid()[0] - height2

    bb = get_safe_selection(img, y, x, height2*2, width2*2)

    return bb


if __name__ == '__main__':
    from colormarks_model import ColormarksModel
    from core.project.project import Project
    cm_model = ColormarksModel()
    cm_model.im_space = 'irb'

    p = Project()
    p.load('/Users/flipajs/Documents/wd/C210/C210.fproj')

    from utils.img_manager import ImgManager
    p.img_manager = ImgManager(p)

    with open(p.working_directory + '/color_samples.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        color_samples = up.load()
        masks = up.load()


    from utils.video_manager import get_auto_video_manager
    vm = get_auto_video_manager(p)
    frame = 0
    main_img = vm.get_frame(frame)
    main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)


    cm_model.compute_model(main_img, color_samples)

    for cs, _ in color_samples:
        for px in cs:
            pos = np.asarray(px / cm_model.num_bins_v, dtype=np.int)
            print px, cm_model.hist3d.hist_labels_[pos[0], pos[1], pos[2]]

    chunks = []

    for v_id in p.gm.get_all_relevant_vertices():
        ch_id = p.gm.g.vp['chunk_start_id'][p.gm.g.vertex(v_id)]
        if ch_id > 0:
            chunks.append(p.chm[ch_id])

    # # TODO: remove, debug...
    # measurements = analyse_chunk(chunks[7], p, cm_model, 3)

    i = 0
    for ch in chunks:
        print i
        measurements = analyse_chunk(ch, p, cm_model, 3)
        i += 1