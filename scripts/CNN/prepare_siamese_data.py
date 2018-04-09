import numpy as np
from utils.img import get_safe_selection
from utils.img import apply_ellipse_mask

def get_region_image(region, vm, offset=45, add_ellipse_mask=True, mask_sigma=10, ellipse_dilation=10):
    img = vm.get_frame(region.frame())
    y, x = region.centroid()
    crop = get_safe_selection(img, y - offset, x - offset, 2 * offset, 2 * offset)
    if add_ellipse_mask:
        crop = apply_ellipse_mask(region, crop, mask_sigma, ellipse_dilation)

    return crop

if __name__ == '__main__':
    NUM_EXAMPLES = 100
    P_WD = '/Users/flipajs/Documents/wd/FERDA/april-paper/Cam1_clip'
    # P_WD = '/Users/flipajs/Documents/wd/FERDA/zebrafish_new'
    from core.project.project import Project

    p = Project()
    p.load(P_WD)

    # go in frame order (optimized video accesss) and pick regions from single-ID tracklets. Compute for them descriptor
    imgs_batch = []
    ids_batch = []
    descriptors = {}

    import cv2

    # TODO: project parameter?
    OFFSET = 45
    ELLIPSE_DILATION = 10
    MASK_SIGMA = 10
    BATCH_SIZE = 500
    APPLY_ELLIPSE = True

    np.set_printoptions(precision=2)
    from tqdm import tqdm

    vm = p.get_video_manager()
    last = None

    from random import randint, choice

    i = 0
    while i < NUM_EXAMPLES:
        frame = randint(0, vm.total_frame_count())

        tracklets = filter(lambda x: x.is_single(), p.chm.tracklets_in_frame(frame))
        if len(tracklets) > 1:
            for ti, trackletA in enumerate(tracklets):
                regionA1 = trackletA.get_random_region(p.gm)
                regionA2 = trackletA.get_random_region(p.gm)

                # TODO: is there a more elegant way?
                while True:
                    trackletB = choice(tracklets)
                    if trackletB != trackletA:
                        break

                regionB = trackletB.get_random_region(p.gm)

                cropA1 = get_region_image(regionA1, vm, offset=OFFSET, add_ellipse_mask=APPLY_ELLIPSE, mask_sigma=MASK_SIGMA, ellipse_dilation=ELLIPSE_DILATION)
                cropA2 = get_region_image(regionA2, vm, offset=OFFSET, add_ellipse_mask=APPLY_ELLIPSE,
                                          mask_sigma=MASK_SIGMA, ellipse_dilation=ELLIPSE_DILATION)
                cropB = get_region_image(regionB, vm, offset=OFFSET, add_ellipse_mask=APPLY_ELLIPSE,
                                          mask_sigma=MASK_SIGMA, ellipse_dilation=ELLIPSE_DILATION)


                cv2.imshow('A1', cropA1)
                cv2.imshow('A2', cropA2)
                cv2.imshow('B', cropB)
                cv2.waitKey(0)

                # imgs_batch.append(crop)
                # ids_batch.append(region.id())
