import mahotas

from utils.img import get_normalised_img, draw_pts, get_cropped_pts


class ZernikeMoments:
    def __init__(self, radius, im_normalised_size):
        self.radius = radius
        self.norm_size = im_normalised_size

    def describe(self, region, normalize = True):
        # return the Zernike moments for the image
        if normalize:
            im = get_normalised_img(region, self.norm_size, blur_sigma=0.2)
        else:
            pts_ = get_cropped_pts(region)
            im = draw_pts(pts_)


        # plt.figure()
        # plt.imshow(normed_im)
        # plt.figure()
        # plt.imshow(normed_im)
        # plt.show()


        return mahotas.features.zernike_moments(im, self.radius)