import cv2
import cPickle as pickle
from core.graph.region_chunk import RegionChunk
from utils.img import get_safe_selection
from math import ceil
from processing import get_colormarks, match_cms_region


def analyse_chunk(ch, project, cm_model, sample_step):
    r_ch = RegionChunk(ch, project.gm, project.rm)

    ch_cms = {}
    for t in range(r_ch.start_frame(), r_ch.end_frame()+1, sample_step):
        r = r_ch[t - r_ch.start_frame()]

        # TODO: set some reasonable parameters
        bb = get_bounding_box(r, project)

        cms = get_colormarks(bb, cm_model)
        for pts, label in cms:
            print label
            for pt in pts:
                if label == 6:
                    bb[pt[0], pt[1], 0] = 255
                else:
                    bb[pt[0], pt[1], 1] = 255

        cv2.imshow('bb', bb)
        cv2.waitKey(25)
        matches = match_cms_region(cms, r)

        ch_cms[t] = matches

    return ch_cms


def get_bounding_box(r, project):
    # TODO set this...:
    border_percent = 1.3

    frame = r.frame()
    img = project.img_manager.get_whole_img(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    main_img = vm.get_frame(500)
    main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)

    for frame, mask in masks:
        im = vm.get_frame(frame)
        # ccs = get_ccs(mask)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # irg_255 = get_irg_255(im)
        # sample_pxs, all_pxs = find_dist_thresholds(ccs, irg_255.copy())

        # color_samples.append((sample_pxs, all_pxs))


    cm_model.compute_model(main_img, color_samples)



    chunks = []

    for v_id in p.gm.get_all_relevant_vertices():
        ch_id = p.gm.g.vp['chunk_start_id'][p.gm.g.vertex(v_id)]
        if ch_id > 0:
            chunks.append(p.chm[ch_id])


    # TODO: remove, debug...
    measurements = analyse_chunk(chunks[7], p, cm_model, 3)

    # i = 0
    # for ch in chunks:
    #     print i
    #     measurements = analyse_chunk(ch, p, cm_model, 3)
    #     i += 1