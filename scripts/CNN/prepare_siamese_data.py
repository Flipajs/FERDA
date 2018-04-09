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

                # imgs += [[im1, im2]]
                # imgs += [[im1, im_negative]]
                # labels += [1, 0]

                # imgs_batch.append(crop)
                # ids_batch.append(region.id())
                i += 1

    # imgs = np.array(imgs)
    # # normalize..
    # imgs = imgs.astype('float32')
    # imgs /= 255
    #
    # split_idx = int(args.test_ratio * len(imgs))
    #
    # imgs_test = np.array(imgs[:split_idx])
    # imgs_train = np.array(imgs[split_idx:])
    # labels_test = np.array(labels[:split_idx])
    # labels_train = np.array(labels[split_idx:])
    #
    # print "imgs TEST: {}, TRAIN: {}".format(imgs_test.shape, imgs_train.shape)
    # print "labels TEST: {}, TRAIN: {}".format(labels_test.shape, labels_train.shape)
    #
    # outdir = args.datadir[0]
    # try:
    #     outdir = args.outdir
    # except:
    #     pass
    #
    # with h5py.File(outdir + '/imgs_train_hard_' + str(args.num_negative) + '.h5', 'w') as hf:
    #     hf.create_dataset("data", data=imgs_train)
    # with h5py.File(outdir + '/imgs_test_hard_' + str(args.num_negative) + '.h5', 'w') as hf:
    #     hf.create_dataset("data", data=imgs_test)
    # with h5py.File(outdir + '/labels_train_hard_' + str(args.num_negative) + '.h5', 'w') as hf:
    #     hf.create_dataset("data", data=labels_train)
    # with h5py.File(outdir + '/labels_test_hard_' + str(args.num_negative) + '.h5', 'w') as hf:
    #     hf.create_dataset("data", data=labels_test)