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
from utils.img_manager import ImgManager
from colormarks_model import ColormarksModel
from core.project.project import Project

DEBUG = True

transformed_cache = {}

def analyse_chunk(ch, project, cm_model, sample_step):
    r_ch = RegionChunk(ch, project.gm, project.rm)

    ch_cms = {}
    i = 0
    for t in range(r_ch.start_frame(), r_ch.end_frame()+1, sample_step):
        if DEBUG:
            if t > 100:
                break

        i += 1

        r = r_ch[t - r_ch.start_frame()]

        bb, offset, orig = get_bounding_box(r, project, cm_model)

        cms = get_colormarks(bb, cm_model)

        # colors = [(0,0,0), 'y', (0.7, 0.7, 0.7), 'r', 'g', 'b']
        colors = [(0, 0, 0), (255, 0, 255), (200, 200, 200), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for pts, label in cms:
            print len(pts)
            c = colors[label]
            for pt in pts:
                bb[pt[0], pt[1], :] = c

        # if i == 1:
        plt.ion()
        plt.subplot(1, 3, 1)
        plt.imshow(bb)
        plt.subplot(1, 3, 2)
        plt.imshow(transform_img_(bb, cm_model))
        plt.subplot(1, 3, 3)
        plt.imshow(orig)
        plt.waitforbuttonpress()

        matches = match_cms_region(filter_cms(cms), r, offset)

        ch_cms[t] = matches

    return ch_cms


def get_bounding_box(r, project, cm_model):
    # TODO set this...:
    border_percent = 1.3

    frame = r.frame()
    img_orig = project.img_manager.get_whole_img(frame)
    if frame in transformed_cache:
        img = transformed_cache[r.frame_]
    else:
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img = transform_img_(img, cm_model)
        transformed_cache[frame] = img

    roi = r.roi()

    height2 = int(ceil((roi.height() * border_percent) / 2.0))
    width2 = int(ceil((roi.width() * border_percent) / 2.0))
    x = r.centroid()[1] - width2
    y = r.centroid()[0] - height2

    bb = get_safe_selection(img, y, x, height2*2, width2*2)

    orig = None
    orig = get_safe_selection(img_orig, y, x, height2*2, width2*2)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB )

    return bb, np.array([y, x]), orig


def evolve_measurements(measurements, animal_id, colors=None):
    votes_ = {}

    i = 0
    for m in measurements.itervalues():
        if m:
            id_ = m[0][1]
            if id_ not in votes_:
                votes_[id_] = 0

            votes_[id_] += 1

            if id_ is not None:
                if id_ < 0 or id_ >= len(colors):
                    c = (0, 0, 0)
                else:
                    c = colors[id_]

                plt.scatter(i, animal_id, c=c)
        i += 1



    best_id = None
    best_val = 0
    for id_, num in votes_.iteritems():
        if num > best_val:
            best_val = num
            best_id = id_

    # plt.grid(True)
    # plt.show()
    # plt.ion()
    # plt.waitforbuttonpress()

    return best_id, best_val /float(len(measurements))


def colormarks_init_finished_cb(project, masks):
    from scripts.irg_hist_demo import get_ccs, find_dist_thresholds

    color_samples = []
    for m in masks:
        mask, frame = m['mask'], m['frame']

        if np.sum(mask) == 0:
            continue

        ccs = get_ccs(mask, min_a=30)

        im = project.img_manager.get_whole_img(frame)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        irg_255 = get_irg_255(im)
        sample_pxs, all_pxs = find_dist_thresholds(ccs, irg_255.copy())

        color_samples.append((sample_pxs, all_pxs))

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open(project.working_directory+'/temp/color_samples_'+timestr+'.pkl', 'wb') as f:
        pickle.dump(color_samples, f)


def exp1():
    p = Project()
    p.load('/Users/flipajs/Documents/wd/GT/Cam1/cam1.fproj')
    p.img_manager = ImgManager(p)

    from utils.video_manager import get_auto_video_manager
    vm = get_auto_video_manager(p)
    frame = 0
    main_img = vm.get_frame(frame)
    main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)


    # with open(p.working_directory + '/temp/color_samples_20160216-155612.pkl', 'rb') as f:
    with open(p.working_directory + '/temp/color_samples_20160218-142811.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        color_samples = up.load()

    cm_model = ColormarksModel(32, 32, 32)
    cm_model.im_space = 'irb'
    cm_model.compute_model(main_img, color_samples)

    chunks = []

    for v_id in p.gm.get_all_relevant_vertices():
        ch_id = p.gm.g.vp['chunk_start_id'][p.gm.g.vertex(v_id)]
        if ch_id > 0:
            chunks.append(p.chm[ch_id])

    ch_ids = {}
    ch_probs = {}

    step = 1

    for ch, animal_id in zip(chunks[:6], range(1, 7)):
        ch.animal_id = animal_id

    colors = [(0,0,0), 'y', (0.7, 0.7, 0.7), 'r', 'g', 'b']

    plt.figure(1)

    for ch in chunks[1:6]:
        measurements = analyse_chunk(ch, p, cm_model, step)

        best_id, best_val = evolve_measurements(measurements, ch.animal_id, colors)

        print "best id:", best_id, "val: ", best_val
        ch_ids[ch], ch_probs[ch] = best_id, best_val


    plt.show()






if __name__ == '__main__':
    exp1()


    # cm_model = ColormarksModel(32, 32, 32)
    # cm_model.im_space = 'irb'
    #
    # p = Project()
    # p.load('/Users/flipajs/Documents/wd/GT/Cam1/cam1.fproj')
    # # p.load('/Users/flipajs/Documents/wd/C210/c210.fproj')
    # p.img_manager = ImgManager(p)
    #
    # if False:
    #     app = QtGui.QApplication(sys.argv)
    #
    #     from gui.arena.colormarks_picker import ColormarksPicker
    #     cp = ColormarksPicker(p, colormarks_init_finished_cb)
    #
    #     cp.show()
    #     cp.move(-500, -500)
    #     cp.showMaximized()
    #     cp.setFocus()
    #
    #     app.exec_()
    #     app.deleteLater()
    #     sys.exit()
    #
    #
    # with open(p.working_directory + '/temp/color_samples_20160216-155612.pkl', 'rb') as f:
    #     up = pickle.Unpickler(f)
    #     color_samples = up.load()
    #
    # from utils.video_manager import get_auto_video_manager
    # vm = get_auto_video_manager(p)
    # frame = 0
    # main_img = vm.get_frame(frame)
    # main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    #
    #
    # cm_model.compute_model(main_img, color_samples)
    #
    # # for cs, _ in color_samples:
    # #     for px in cs:
    # #         pos = np.asarray(px / cm_model.num_bins_v, dtype=np.int)
    # #         # print px, cm_model.hist3d.hist_labels_[pos[0], pos[1], pos[2]]
    #
    # chunks = []
    #
    # for v_id in p.gm.get_all_relevant_vertices():
    #     ch_id = p.gm.g.vp['chunk_start_id'][p.gm.g.vertex(v_id)]
    #     if ch_id > 0:
    #         chunks.append(p.chm[ch_id])
    #
    # ch_ids = {}
    # ch_probs = {}
    #
    # step = 1
    #
    # for ch in chunks[5:]:
    #     print ch.id_
    #     measurements = analyse_chunk(ch, p, cm_model, step)
    #
    #     best_id, best_val = evolve_measurements(measurements)
    #     print "best id: %d, val: %f" %(best_id, best_val)
    #     ch_ids[ch], ch_probs[ch] = best_id, best_val
