from core.project.project import Project
from core.graph.region_chunk import RegionChunk
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
from skimage.data import binary_blobs
import skimage

if __name__ == '__main__':
    p = Project()
    p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_')

    ch = p.chm[257]

    rch = RegionChunk(ch, p.gm, p.rm)
    start_vertex = ch.start_vertex(p.gm)

    in_regions = []
    for n in start_vertex.in_neighbours():
        r = p.gm.region(n)
        in_regions.append(r)

    r = rch[0]
    r.frame()

    from utils.video_manager import get_auto_video_manager
    from skimage.morphology import skeletonize_3d
    import cv2

    vm = get_auto_video_manager(p)

    # TODO: idea - label erosion before each nex iteration...
    from scipy.ndimage.morphology import binary_erosion

    whole_labels = None
    for r1 in rch.regions_gen():
        markers = np.zeros((1000, 1000), dtype=np.int32)
        r1_im = np.zeros((1000, 1000), dtype=np.bool)
        r1_im[r1.pts()[:, 0], r1.pts()[:, 1]] = True
        markers[np.logical_not(r1_im)] = -1

        img = vm.get_frame(r1.frame())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i, r in enumerate(in_regions):
            if whole_labels is None:
                im2 = np.zeros((1000, 1000), dtype=np.bool)
                im2[r.pts()[:, 0], r.pts()[:, 1]] = True

                markers[np.logical_and(r1_im, im2)] = i+1
            else:
                l_ = whole_labels==i+1
                l_ = binary_erosion(l_, iterations=5)
                markers[np.logical_and(r1_im, l_)] = i+1

        tl = r1.roi().top_left_corner()
        br = r1.roi().bottom_right_corner()
        gray = gray[tl[0]:br[0], tl[1]:br[1]].copy()
        markers = markers[tl[0]:br[0], tl[1]:br[1]].copy()
        r1_im = r1_im[tl[0]:br[0], tl[1]:br[1]].copy()
        skel = skeletonize_3d(r1_im)

        data=np.asarray(r1_im, dtype=np.uint8)*255
        labels = random_walker(gray, markers, beta=500000, mode='bf')

        whole_labels = np.zeros((1000, 1000), dtype=np.int32)
        whole_labels[tl[0]:br[0], tl[1]:br[1]] = labels.copy()

        # Plot results
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8, 3.2), sharex=True, sharey=True)
        ax1.imshow(gray, cmap='gray', interpolation='nearest')
        ax1.axis('off')
        ax1.set_adjustable('box-forced')
        ax1.set_title('Noisy data')
        ax2.imshow(markers, cmap='hot', interpolation='nearest')
        ax2.axis('off')
        ax2.set_adjustable('box-forced')
        ax2.set_title('Markers')
        ax3.imshow(labels, cmap='hot', interpolation='nearest')
        ax3.axis('off')
        ax3.set_adjustable('box-forced')
        ax3.set_title('Segmentation')
        ax4.imshow(skel)
        ax4.axis('off')
        ax4.set_adjustable('box-forced')
        ax4.set_title('skeleton')


        fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                            right=1)
        plt.show()
        plt.ion()
        plt.waitforbuttonpress()
        plt.close()




