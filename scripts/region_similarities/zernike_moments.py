from __future__ import print_function
import mahotas
import numpy as np
import argparse
import cPickle as pickle
import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import scipy.spatial
from pylab import pcolor, show, colorbar, xticks, yticks
from utils.img import imresize, get_normalised_img, draw_pts, get_cropped_pts
from core.desc.zernike_moments import ZernikeMoments


def process_chunk():
    norm_size = np.array([30, 100])
    desc = ZernikeMoments(21, norm_size)

    with open('/Users/flipajs/Documents/dev/ferda/scripts/datasets/c210-few_chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)

    ch = chunks[1]
    # for ch in chunks:
        # if len(ch) < 10:
        #     continue

    thresh = 0.2

    last_desc = None
    dists = []
    for r in ch:
        moments = desc.describe(r)

        if not (last_desc is None):
            d = np.linalg.norm(moments-last_desc)
            if d > thresh:
                print(r.frame(), d)
                im = draw_pts(get_cropped_pts(r))
                plt.imshow(im)
                plt.show()


            dists.append(d)

        last_desc = moments

    dists = np.array(dists[1:-1])
    print(('ch len: %d, min: %.3f max: %.3f med: %.3f') %(len(ch), np.min(dists), np.max(dists), np.median(dists)))


if __name__ == '__main__':
    # process_chunk()
    if True:
        norm_size = np.array([30, 100])

        desc = ZernikeMoments(51, norm_size)

        with open('/Users/flipajs/Documents/dev/ferda/scripts/datasets/c210-few_regions.pkl', 'rb') as f:
            regions = pickle.load(f)


        regions = sorted(regions, key=lambda x: -x.area())

        prev_desc = None
        i = 0

        im_num = 80

        descriptions = []
        images = []
        for i in range(im_num):
            r = regions[i]
            normed_im = get_normalised_img(r, norm_size)

            moments = desc.describe(r, normalize=False)
            if not prev_desc is None:
                print("distance to previous: ", np.linalg.norm(prev_desc-moments))

            descriptions.append(moments)
            images.append(normed_im)
            prev_desc = moments
            print(moments)

            # plt.imshow(normed_im)
            # plt.show()

            i += 1
            if i == im_num:
                break

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid
        import numpy as np

        fig = plt.figure(1, (25., 15.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(4, 20),
                         axes_pad=0.05,  # pad between axes in inch.
                         )

        for i in range(im_num):
            grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

        plt.figure(2)

        descriptions = np.array(descriptions)
        dists = scipy.spatial.distance.cdist(descriptions, descriptions)

        pcolor(dists)
        # yticks(arange(0.5,10.5),range(0,10))
        # xticks(arange(0.5,10.5),range(0,10))
        plt.show()
