import mahotas
from utils.img import get_normalised_img
import scipy.ndimage
import matplotlib.pyplot as plt

class ZernikeMoments:
    def __init__(self, radius, im_normalised_size):
        self.radius = radius
        self.norm_size = im_normalised_size

    def describe(self, region):
        # return the Zernike moments for the image
        normed_im = get_normalised_img(region, self.norm_size, blur_sigma=0.2)

        # plt.figure()
        # plt.imshow(normed_im)
        # plt.figure()
        # plt.imshow(normed_im)
        # plt.show()


        return mahotas.features.zernike_moments(normed_im, self.radius)