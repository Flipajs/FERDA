from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import range
from past.utils import old_div
import cv2
import pickle as pickle
from core.graph.region_chunk import RegionChunk
from utils.img import get_safe_selection
from math import ceil
from .processing import get_colormarks, match_cms_region, filter_cms
import matplotlib.pyplot as plt
import numpy as np
from scripts.irg_hist_demo import ColorHist3d
from scripts.irg_hist_demo import *
from .processing import transform_img_
from utils.img_manager import ImgManager


def analyse_chunk(ch, project, cm_model, sample_step):
    r_ch = RegionChunk(ch, project.gm, project.rm)

    ch_cms = {}
    for t in range(r_ch.start_frame(), r_ch.end_frame()+1, sample_step):
        r = r_ch[t - r_ch.start_frame()]

        # TODO: set some reasonable parameters
        bb, offset = get_bounding_box(r, project, cm_model)

        cms = get_colormarks(bb, cm_model)
        for pts, label in cms:
            for pt in pts:
                bb[pt[0], pt[1], 0:2] = 255

        cv2.imshow('bb', bb)
        cv2.imshow('bb_t', transform_img_(bb, cm_model))
        cv2.waitKey(1)

        matches = match_cms_region(filter_cms(cms), r, offset)

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

    height2 = int(ceil(old_div((roi.height() * border_percent), 2.0)))
    width2 = int(ceil(old_div((roi.width() * border_percent), 2.0)))
    x = r.centroid()[1] - width2
    y = r.centroid()[0] - height2

    bb = get_safe_selection(img, y, x, height2*2, width2*2)

    return bb, np.array([y, x])


def evolve_measurements(measurements):
    votes_ = {}

    for m in measurements.values():
         if m:
             id_ = m[0][1]
             if id_ not in votes_:
                 votes_[id_] = 0

             votes_[id_] += 1

    best_id = None
    best_val = 0
    for id_, num in votes_.items():
        if num > best_val:
            best_val = num
            best_id = id_

    return best_id, old_div(best_val,float(len(measurements)))


def colormarks_init_finished_cb(project, masks):
    from scripts.irg_hist_demo import get_ccs, find_dist_thresholds

    color_samples = []
    for m in masks:
        mask, frame = m['mask'], m['frame']

        if np.sum(mask) == 0:
            continue

        ccs = get_ccs(mask)

        im_orig = project.img_manager.get_whole_img(frame)

        im = cv2.cvtColor(im_orig, cv2.COLOR_BGR2RGB)
        irg_255 = get_irg_255(im)
        sample_pxs, all_pxs, mean_color = find_dist_thresholds(ccs, irg_255.copy(), im_orig)

        color_samples.append((sample_pxs, all_pxs, mean_color))

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    import os
    try:
        os.mkdir(project.working_directory+'/temp')
    except:
        pass

    with open(project.working_directory+'/temp/color_samples_'+timestr+'.pkl', 'wb') as f:
        pickle.dump(color_samples, f)

    return color_samples


if __name__ == '__main__':
    from core.config import config
    from .colormarks_model import ColormarksModel
    from core.project.project import Project
    cm_model = ColormarksModel()
    cm_model.im_space = 'irb'

    p = Project()
    p.load('/Users/flipajs/Documents/wd/GT/Cam1__/cam1.fproj')
    # p.load('/Users/flipajs/Documents/wd/C210/c210.fproj')
    p.img_manager = ImgManager(p, max_size_mb=config['cache']['img_manager_size_MB'])

    if False:
        app = QtGui.QApplication(sys.argv)

        from gui.arena.colormarks_picker import ColormarksPicker
        cp = ColormarksPicker(p, colormarks_init_finished_cb)

        cp.show()
        cp.move(-500, -500)
        cp.showMaximized()
        cp.setFocus()

        app.exec_()
        app.deleteLater()
        sys.exit()

    with open(p.working_directory + '/temp/color_samples_20160223-135328.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        color_samples = up.load()


    from utils.video_manager import get_auto_video_manager
    vm = get_auto_video_manager(p)
    frame = 0
    main_img = vm.get_frame(frame)
    main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)


    cm_model.compute_model(main_img, color_samples)

    for cs, _, _ in color_samples:
        for px in cs:
            pos = np.asarray(old_div(px, cm_model.num_bins_v), dtype=np.int)
            print(px, cm_model.hist3d.hist_labels_[pos[0], pos[1], pos[2]])

    chunks = []

    for v_id in p.gm.get_all_relevant_vertices():
        ch_id = p.gm.g.vp['chunk_start_id'][p.gm.g.vertex(v_id)]
        if ch_id > 0:
            chunks.append(p.chm[ch_id])

    # # TODO: remove, debug...
    # measurements = analyse_chunk(chunks[7], p, cm_model, 3)

    i = 0

    ch_ids = {}
    ch_probs = {}

    for ch in chunks:
        r_ch = RegionChunk(ch, p.gm, p.rm)
        for f in range(ch.start_frame(p.gm), ch.end_frame(p.gm)):
            r = r_ch.region_in_t(f)

            cm = cm_model.find_colormarks(p, r)
