import matplotlib.pylab as plt
import numpy as np
import cv2

# montage of images


class Montage(object):
    """
    example:

    plt.figure()
    montage = Montage((2500, 1600), (2, 3))
    plt.imshow(montage.montage(list_of_images))
    plt.grid(False)
    plt.tight_layout(0)
    plt.axis('off')
    """
    def __init__(self, montage_size, nm=None):
        """
        Initialize image montage.

        :param montage_size: montage size in pixels (w, h)
        :type montage_size: tuple
        :param nm: number of images horizontally and vertically (n, m)
        :type nm: tuple
        """
        self.montage_size = np.array(montage_size)
        if nm is None:
            self.nm = None
            self.cell_size = None
        else:
            self.nm = np.array(nm, dtype=int)
            self.cell_size = self.montage_size / self.nm
        self.shape = None
        self.sizes = None

    def __adjust_image_size__(self, img_size):
        """
        Compute a new size for an image to fit into the montage.
        """
        ratio = img_size[0] / float(img_size[1])
        sizes_ratio = self.cell_size.astype(float) / img_size
        if sizes_ratio[0] < sizes_ratio[1]:
            # horizontally tight
            dst_size = [self.cell_size[0], self.cell_size[0] / ratio]
        else:
            # vertically tight
            dst_size = [self.cell_size[1] * ratio, self.cell_size[1]]
        return np.round(dst_size).astype(int)

    # def __get_image_shapes__(self, images, max_width):
    #     width = 0
    #     shapes = []
    #     i = 0
    #     while width + images[i].shape[1] <= max_width:
    #         shapes.append(images[i].shape)
    #         width += images[i].shape[1]
    #         i += 1
    #     return shapes
    #
    # def __get_max_height__(self, images, widths):
    #     width = 0
    #     heights = []
    #     i = 0
    #     while width + images[i].shape[1] <= max_width:
    #         shapes.append(images[i].shape)
    #         width += images[i].shape[1]
    #         i += 1
    #     return shapes
    #
    # def __montage_auto__(self, images):
    #     shapes_first_line = self.__get_image_shapes__(images, self.montage_size[1])
    #     widths = [shape[1] for shape in shapes_first_line]
    #
    #     width = 0
    #     i = 0
    #     while width + images[i].shape[1] <= self.montage_size[1]:
    #         widths.append(images[i].shape[1])
    #         width += images[i].shape[1]
    #         i += 1
    #
    #     height = 0
    #     heights = []
    #     i = 0
    #     while height + images[i].shape[0] <= self.montage_size[0]:
    #         while width + images[i].shape[1] <= self.montage_size[1]:
    #             widths.append(images[i].shape[1])
    #             width += images[i].shape[1]
    #             i += 1
    #         widths.append(images[i].shape[1])
    #         height += images[i].shape[1]
    #         i += 1

    def montage(self, images):
        """
        Make a montage out of images.

        :param images: list of images, max n*m
        :type images: list of array-like images
        :return: image montage
        :rtype: array-like
        """
        if not self.sizes:
            self.sizes = [self.__adjust_image_size__(img.shape[:2][::-1]) for img in images]
        if not self.shape:
            if images[0].ndim == 2:
                self.shape = tuple(self.montage_size[::-1])
            else:
                self.shape = tuple(self.montage_size[::-1]) + (3,)

        images_resized = [cv2.resize(f, (tuple(np.round(s).astype(int)))) for f, s in zip(images, self.sizes)]
        out = np.zeros(self.shape, dtype=np.uint8)
        for i in xrange(len(images)):
            x = (i % self.nm[0]) * int(self.cell_size[0])
            y = (i / self.nm[0]) * int(self.cell_size[1])
            imgw, imgh = self.sizes[i]
            out[y: y + imgh, x: x + imgw] = images_resized[i]
        return out


def save_figure_as_image(file_name, fig, size_px=None):
    """ Save a Matplotlib figure as an image without borders or frames.
       Args:
            file_name (str): String that ends with .png etc.

            fig (Matplotlib figure instance): figure you want to save as the image

        :param size_px: dimensions of the figure in pixels (width, height)
        :type size_px: tuple of ints

    TODO: loosely followed http://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels
    """
    if size_px is not None:
        display_dpi = 96
        fig.set_dpi(display_dpi)
        fig.set_size_inches([size_px[0] / display_dpi, size_px[1] / display_dpi])
        dpi = display_dpi
    else:
        dpi = None

    fig.patch.set_alpha(0)
    ax = fig.gca()
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    # if size_px is not None:
    #     plt.xlim(0, size_px[0])
    #     plt.ylim(size_px[1], 0)
    # else:
    #     fig_size = fig.get_size_inches()
    #     w_in, h_in = fig_size[0], fig_size[1]
    #     plt.xlim(0, w_in)
    #     plt.ylim(h_in, 0)

    fig.savefig(file_name, transparent=True, bbox_inches='tight', pad_inches=0) # , dpi=dpi)