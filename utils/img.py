
import math
import cv2
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
from utils.roi import get_roi


def get_safe_selection(img, y, x, height, width, fill_color=(255, 255, 255), return_offset=False):
    y = int(y)
    x = int(x)
    height = int(height)
    width = int(width)

    border = max(max(-y, -x), 0)

    channels = 1
    if len(img.shape) > 2:
        channels = img.shape[2]

    if len(fill_color) != channels:
        fill_color = 255

    h_ = img.shape[0] - (height + y)
    w_ = img.shape[1] - (width + x)

    border = max(border, max(max(-h_, -w_), 0))

    # fast detection of case where selection is inside image
    borders_needed = y < 0 or x < 0 or y + height >= img.shape[0] or x + width >= img.shape[1]
    if border > 0 and borders_needed:
        img_ = np.zeros((img.shape[0] + 2 * border, img.shape[1] + 2 * border, channels), dtype=img.dtype)
        img_ += np.asarray(fill_color, dtype=img.dtype)
        img_[border:-border, border:-border] = img
        # crop = np.ones((height, width, channels), dtype=img.dtype)
        # crop *= np.asarray(fill_color, dtype=img.dtype)

        y += border
        x += border
        crop = np.copy(img_[y:y + height, x:x + width, :])
    else:
        crop = img[y:y + height, x:x + width, :].copy()

    if return_offset:
        return crop, np.array([y, x])

    return crop


def safe_crop(img, xy, crop_size_px, return_src_range=False):
    """
    Crop image safely around a position, even if the output lays outside the input image.

    (TODO: similar to get_safe_selection)
    Note: uses numpy rounding, see ndarray.round.


    :param img: input image
    :param xy: center of the output image
    :param crop_size_px: output image size (rectangle side length)
    :return: img_crop: output image
             delta_xy: correction delta for coordinates in the output image; e.g. img_crop_pos = img_pos - delta_xy
    """
    def crop_range(x, src_size, dst_size):
        """
        Find source and destination ranges to safely copy dst_size array from source array centered in x position.

        :param x: center position
                  corner case: when x is not modulo 0.5 or dst_size is even (the returned integral range can't be
                  symmetrical) the returned ranges include the first element rather then the last element
        :param src_size: size of source array
        :param dst_size: size of destination array
        :return: src_range_clipped, dst_range_clipped - ranges (suitable for slice(*src_range_clipped))
        """
        src_range = np.floor(np.array((x - dst_size / 2, x + dst_size / 2)) + 0.5).astype(int)  # range end is excluded
        src_range_clipped = np.clip(src_range, 0, src_size)
        dst_range = np.array((0, dst_size))  # range end is excluded
        dst_range_clipped = dst_range - (src_range - src_range_clipped)
        return src_range_clipped, dst_range_clipped

    img_crop = np.zeros(((crop_size_px, crop_size_px) + img.shape[2:]), dtype=np.uint8)
    x_range_src, x_range_dst = crop_range(xy[0], img.shape[1], crop_size_px)
    y_range_src, y_range_dst = crop_range(xy[1], img.shape[0], crop_size_px)

    img_crop[slice(*y_range_dst), slice(*x_range_dst)] = \
        img[slice(*y_range_src), slice(*x_range_src)]
    delta_xy = np.array((x_range_src[0] - x_range_dst[0], y_range_src[0] - y_range_dst[0]))
    if return_src_range:
        return img_crop, delta_xy, (x_range_src, y_range_src)
    else:
        return img_crop, delta_xy


def get_img_around_pts(img, pts, margin=0):
    roi = get_roi(pts)

    width = roi.width()
    height = roi.height()

    m_ = max(width, height)
    margin = m_ * margin

    y_ = roi.y() - margin
    x_ = roi.x() - margin
    height_ = height + 2 * margin
    width_ = width + 2 * margin

    crop = get_safe_selection(img, y_, x_, height_, width_, fill_color=(0, 0, 0))
    return crop, np.array([y_, x_])


def avg_circle_area_color(im, y, x, radius):
    """
    computes average color in circle area given by pos and radius
    :param im:
    :param pos:
    :param radius:
    :return:
    """

    c = np.zeros((1, 3), dtype=np.double)
    num_px = 0
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            d = ((w - radius) ** 2 + (h - radius) ** 2) ** 0.5
            if d <= radius:
                num_px += 1
                c += im[y - radius + h, x - radius + w, :]

    print(num_px)
    c /= num_px

    return [c[0, 0], c[0, 1], c[0, 2]]


def prepare_for_visualisation(img, project):
    from skimage.transform import rescale

    if project.other_parameters.img_subsample_factor > 1.0:
        img = np.asarray(rescale(img, 1/project.other_parameters.img_subsample_factor) * 255, dtype=np.uint8)

    return img


def prepare_for_segmentation(img, project, grayscale_speedup=True):
    from scipy.ndimage import gaussian_filter
    from skimage.transform import rescale
    if project.bg_model:
        img = project.bg_model.bg_subtraction(img)

    if grayscale_speedup:
        try:
            if project.other_parameters.use_only_red_channel:
                img = img[:,:,2].copy()
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(e)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if project.arena_model is not None:
        img = project.arena_model.mask_image(img)

    if project.mser_parameters.gaussian_kernel_std > 0:
        img = gaussian_filter(img, sigma=project.mser_parameters.gaussian_kernel_std)

    if project.other_parameters.img_subsample_factor > 1.0:
        img = np.asarray(rescale(img, 1/project.other_parameters.img_subsample_factor) * 255, dtype=np.uint8)

    return img


def get_cropped_pts(region, return_roi=True, only_contour=True):
    roi_ = region.roi()
    if only_contour:
        pts = region.contour() - roi_.top_left_corner()
    else:
        pts = region.pts() - roi_.top_left_corner()

    if return_roi:
        return pts, roi_

    return pts

def imresize(im,sz):
    """  Resize an image array using PIL. """
    from PIL import Image

    pil_im = Image.fromarray(np.uint8(im))

    return np.array(pil_im.resize(sz))


def draw_pts(pts_):
    h_ = np.max(pts_[:, 0]) + 1
    w_ = np.max(pts_[:, 1]) + 1

    im = np.zeros((h_, w_), dtype=np.bool)
    im[pts_[:, 0], pts_[:, 1]] = True

    return im


def get_normalised_img(region, norm_size, blur_sigma=0):
    pts_ = get_cropped_pts(region)

    im = draw_pts(pts_)

    # plt.figure()
    # plt.imshow(im)

    if blur_sigma > 0:
        im = np.asarray(np.round(gaussian_filter(im, sigma=blur_sigma)), dtype=np.bool)

    # plt.figure()
    # plt.imshow(im)
    # plt.show()

    im_normed = imresize(im, norm_size)
    im_out = np.zeros(im_normed.shape, dtype=np.uint8)

    # fill holes
    (cnts, _) = cv2.findContours(im_normed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    cv2.drawContours(im_out, [cnts], -1, 255, -1)

    return im_out


def replace_everything_but_pts(img, pts, fill_color=[0, 0, 0]):
    # TODO: 3 channels vs 1 channel (:, :, 1) and (:, :)
    if len(img.shape) == 2 or img.shape[3] == 1:
        if len(fill_color) > 1:
            fill_color = fill_color[0]

    new_img = np.zeros(img.shape, dtype=img.dtype)
    new_img[:, :] = fill_color

    new_img[pts[:, 0], pts[:, 1]] = img[pts[:, 0], pts[:, 1]]

    return new_img


class DistinguishableColors():
    def __init__(self, N, step=5, cmap='hsv'):
        self.step = step
        self.N = math.ceil(N/step) * step
        color_norm = colors.Normalize(vmin=0, vmax=self.N-1)
        self.scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)

    def get_color(self, index):
        i = self.step * (index % self.step) + index / self.step #   index/self.step + index % self.step
        return self.scalar_map.to_rgba(i)

def get_cmap(N, step):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''

    N = math.ceil(N/step) * step
    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='nipy_spectral')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color


def rotate_img(img, theta, center=None):
    s_ = max(img.shape[0], img.shape[1])

    im_ = np.zeros((s_, s_, img.shape[2]), dtype=img.dtype)
    h_ = (s_ - img.shape[0]) / 2
    w_ = (s_ - img.shape[1]) / 2

    im_[h_:h_+img.shape[0], w_:w_+img.shape[1], :] = img

    if isinstance(center, np.ndarray):
        center = (center[0], center[1])
    elif center is None:
        center = (im_.shape[0] / 2, im_.shape[1] / 2)

    rot_mat = cv2.getRotationMatrix2D(center, -np.rad2deg(theta), 1.0)  # TODO: 2018-10-31 is this correct? getRotationMatrix2D and Region have both ccw angle orientation
    return cv2.warpAffine(im_, rot_mat, (s_, s_))


def centered_crop(img, new_h, new_w):
    new_h = int(new_h)
    new_w = int(new_w)

    h_ = img.shape[0]
    w_ = img.shape[1]

    y_ = (h_ - new_h) / 2
    x_ = (w_ - new_w) / 2

    if y_ < 0 or x_ < 0:
        Warning('cropped area cannot be bigger then original image!')
        return img

    return img[y_:y_+new_h, x_:x_+new_w, :].copy()

def get_bounding_box(r, project, relative_border=1.3, absolute_border=-1, img=None):
    from math import ceil

    if img is None:
        frame = r.frame()
        img = project.img_manager.get_whole_img(frame)

    roi = r.roi()

    height2 = int(ceil((roi.height() * relative_border) / 2.0))
    width2 = int(ceil((roi.width() * relative_border) / 2.0))

    if absolute_border > -1:
        height2 = absolute_border
        width2 = absolute_border

    x = r.centroid()[1] - width2
    y = r.centroid()[0] - height2

    bb = get_safe_selection(img, y, x, height2*2, width2*2)

    return bb, np.array([y, x])

def endpoint_rot(bb_img, pt, theta, centroid):
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta), math.cos(theta)]]
    )

    pt_ = pt-centroid
    pt = np.dot(rot, pt_.reshape(2, 1))
    new_pt = [int(round(pt[0][0] + bb_img.shape[0]/2)), int(round(pt[1][0] + bb_img.shape[1]/2))]

    return new_pt


def alter_img_saturation_intensity(img, sat_alpha=1.0, int_alpha=1.0):
    intensity_matrix = np.diag([int_alpha] * 3)
    rwgt = 0.3086
    gwgt = 0.6094
    bwgt = 0.0820

    saturation_matrix = np.full([3, 3], 1.0 - sat_alpha)
    saturation_matrix[0, :] *= rwgt
    saturation_matrix[1, :] *= gwgt
    saturation_matrix[2, :] *= bwgt


    for x in range(3): saturation_matrix[x, x] += sat_alpha

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = img[i, j].dot(saturation_matrix).dot(intensity_matrix)

    return img


def alter_img_intensity(img, alpha=1.0):
    intensity_matrix = np.diag([alpha] * 3)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = img[i, j].dot(intensity_matrix)
    return img


def alter_img_saturation(img, alpha=1.0):
    """To alter saturation, pixel components must move towards or away from the pixel's luminance value.
    By using a black-and-white image as the degenerate version, saturation can be decreased using interpolation,
    and increased using extrapolation. This avoids computationally more expensive conversions to and from HSV space.
    Repeated update in an interactive application is especially fast,
    since the luminance of each pixel need not be recomputed.
    Negative alpha preserves luminance but inverts the hue of the input image"""
    rwgt = 0.3086
    gwgt = 0.6094
    bwgt = 0.0820

    saturation_matrix = np.full([3, 3], 1.0 - alpha)
    saturation_matrix[0, :] *= rwgt
    saturation_matrix[1, :] *= gwgt
    saturation_matrix[2, :] *= bwgt
    for x in range(3): saturation_matrix[x, x] += alpha

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = img[i, j].dot(saturation_matrix)

    return img


def img_saturation_coef(img, saturation_coef=2.0, intensity_coef=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv = np.asarray(img_hsv, dtype=float)

    img_hsv[:,:,0] *= intensity_coef
    img_hsv[:,:,1] *= saturation_coef
    img_hsv[:,:,2] *= saturation_coef

    img = np.asarray(np.clip(img_hsv - img_hsv.min(), 0, 255), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return img


def apply_ellipse_mask(r, im, sigma=10, ellipse_dilation=10):
    from scipy import ndimage
    from math import ceil

    x = np.zeros((im.shape[0], im.shape[1]))

    deg = int(r.theta_ * 57.295)
    # angle of rotation of ellipse in anti-clockwise direction
    cv2.ellipse(x, (int(x.shape[0] / 2), int(x.shape[1] / 2)),
                (int(ceil(r.a_)) + ellipse_dilation, int(ceil(r.b_)) + ellipse_dilation),
                -deg, 0, 360, 255, -1)

    y = ndimage.filters.gaussian_filter(x, sigma=sigma)
    y /= y.max()

    for i in range(3):
        im[:, :, i] = np.multiply(im[:, :, i].astype(np.float), y)

    return im


def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])

    author: mohiksan, https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator/32857432#32857432
    """
    # define local variables for readability


    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer

