from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import cv2
import numpy as np
from skimage.transform import pyramid_gaussian
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb
import time


class SegmentationHelper(object):
    def __init__(self, image, num=2, scale=2):
        self.pyramid = None
        self.images = None
        self.image = None
        self.edges = None
        self.shiftx = None
        self.shifty = None
        self.avg = None
        self.maxs = None
        self.mins = None
        self.diff = None
        self.bg = None
        self.gr = None
        self.rb = None
        self.h = None
        self.w = None
        self.num = num
        self.scale = 2**0.5

        self.rfc_n_jobs = 1
        self.rfc_n_estimators = 10

        self.use_reduced_feature_set = True
        self.num_features = 6

        # these arrays contain learning data from previous frames
        # after confirming selection on a frame, temporary data is copied here
        self.X = []  # X is a list of tuples, each tuple contains N properties of a pixel, eg (r, g, b)
        self.y = []  # y is a list of class ids, and contains class data for tuples in X respectively

        # these arrays are used to store learn data from current frame
        # they are recreated from scratch with every action
        # after confirming selection, it's copied to permanent storage (self.X and self.y)
        self.Xtmp = []
        self.ytmp = []

        self.rfc = None
        self.unused = [] # list of features that are currently ignored
        self.set_image(image)

    def set_image(self, image):
        """
        Sets a new image, computes all features on the new images and prepares to use it in human_iloop_classification.
        To preserve RFC training data from previous images, use "helper.update_xy()"
        :param image: new image
        :return: None
        """

        # TODO: remove this...
        self.print_times = False
        #
        # try:
        #     self.rfc.n_jobs = 1

        tt = time.time()
        t = time.time()

        self.image = image  # original image
        self.h, self.w, c = self.image.shape

        # images are stored in lists with len corresponding to pyramid height
        # index 0 contains data obtained from largest image, all other indices contain data from scaled images
        #     but are expanded again to match original image size
        # self.pyramid = self.make_pyramid()  # source image in all scales
        self.pyramid = pyramid(self.image, scale=self.scale, num=self.num)  # source image in all scales

        if self.print_times:
            print("set_image pyramid time: {:.4f}".format(time.time() - t))

        t = time.time()
        self.images = self.get_images()  # original images from pyramid, but expanded (result looks blurry)
        if self.print_times:
            print("set_image rescale time: {:.4f}".format(time.time() - t))

        t = time.time()
        # TODO: FASTER SHIFT USING COPY
        # self.shiftx = np.zeros_like(imageY)
        # shift_up[0:-1, :] = image[1:, :].copy()

        self.shiftx = self.get_shift(shift_x=2, shift_y=0)  # diff from shifted images, rescaled
        self.shifty = self.get_shift(shift_x=0, shift_y=2)
        if self.print_times:
            print("set_image shift time: {:.4f}".format(time.time() - t))

        if not self.use_reduced_feature_set:
            self.bg = self.get_cdiff(0, 1)  # channel difs, rescaled
            self.gr = self.get_cdiff(1, 2)
            self.rb = self.get_cdiff(2, 0)

            self.avg = self.get_avg()  # average value on each pixel, rescaled

            self.edges = self.get_edges()  # canny edge detector, rescaled to largest image

            # self.maxs, self.mins, self.diff = self.get_dif()

        t = time.time()
        self.diff = self.get_dif()
        if self.print_times:
            print("set_image diff time: {:.4f}".format(time.time() - t))

        if self.print_times:
            print("set_image time: {:.4f}".format(time.time() - tt))

    def get_data(self, i, j, X, y, classification):
        """
        Appends pixel data to given lists
        :param i: I coordinate
        :param j: J coordinate
        :param X: input list with feature tuples
        :param y: input list with classifications
        :param classification: class of the given pixel
        :return:
        """

        x = []
        if not self.use_reduced_feature_set:
            for k in range(0, self.num):
                b, g, r = self.images[k][i][j]
                sx = self.shiftx[k][i][j]
                sy = self.shifty[k][i][j]
                a = self.avg[k][i][j]
                c = self.bg[k][i][j]
                d = self.gr[k][i][j]
                e = self.rb[k][i][j]
                f = self.maxs[k][i][j]
                h = self.mins[k][i][j]
                k = self.diff[k][i][j]
                x.extend([b, g, r, a, sx, sy, c, d, e, f, h, k])
        else:
            for k in range(0, self.num):
                b, g, r = self.images[k][i][j]
                sx = self.shiftx[k][i][j]
                sy = self.shifty[k][i][j]
                # a = self.avg[k][i][j]
                # c = self.bg[k][i][j]
                # d = self.gr[k][i][j]
                # e = self.rb[k][i][j]
                # f = self.maxs[k][i][j]
                # h = self.mins[k][i][j]
                k = self.diff[k][i][j]
                x.extend([r, sx, sy, k])

        X.append(x)
        y.append(classification)

    def train(self, background, foreground):
        """
        Creates a classificator using previous frame data and bg/fg examples. Computes probabilities for current frame.
        :param background: np mask for examples in background class
        :param foreground: np mask for examples in foreground class
        :return: 0.0 - 1.0 probability mask
        """
        from sklearn.ensemble import RandomForestClassifier

        # prepare learning data
        # X contains tuples of data for each evaluated unit-pixel (R, G, B, edge?)
        # y contains classifications for all pixels respectively
        self.Xtmp = []
        self.ytmp = []

        # loop all nonzero pixels from foreground (ants) and background and add them to testing data
        start = time.time()
        nzero = np.nonzero(background[0])
        if len(nzero[0]) == 0 and 0 not in self.y:
            return None
        for i, j in zip(nzero[0], nzero[1]):
            self.get_data(i, j, self.Xtmp, self.ytmp, 0)  # 0 for background

        nzero = np.nonzero(foreground[0])
        if len(nzero[0]) == 0 and 1 not in self.y:
            return None
        for i, j in zip(nzero[0], nzero[1]):
            self.get_data(i, j, self.Xtmp, self.ytmp, 1)  # 1 for foreground

        if self.print_times:
            print("Retrieving data takes %f" % (time.time() - start))

        # create the classifier
        self.rfc = RandomForestClassifier(n_estimators=self.rfc_n_estimators, n_jobs=self.rfc_n_jobs)

        # to train on all data (current and all previous frames), join the arrays together
        # class variables are not affected here
        X = list(self.Xtmp)
        X.extend(self.X)
        y = list(self.ytmp)
        y.extend(self.y)

        # train the classifier
        start = time.time()
        self.rfc.fit(X, y)
        if self.print_times:
            print("RFC fitting takes     %f" % (time.time() - start))

        # find unused features and remove them
        # create new classifier with less features, it will be faster
        start = time.time()
        self.unused = find_unused_features(self.rfc)
        self.rfc = get_filtered_rfc(self.unused, X, y, self.rfc_n_estimators, self.rfc_n_jobs)
        if self.print_times:
            print("RFC filtering takes   %f. Using %d out of %d features." % \
                  (time.time() - start, len(self.rfc.feature_importances_), len(X[0])))

    def predict(self):
        if self.rfc is None:
            return

        t = time.time()
        # get all feature data from image and create one large array

        try:
            layers = self.get_features()
        except ValueError:
            print("error occured, logging...")
            print("pyramid size: ", len(self.pyramid))
            for i in range(len(self.pyramid)):
                print("i, shape: ", self.pyramid[0].shape)

            print("")
            print("images size: ", len(self.images))
            for i in range(len(self.images)):
                print("i, shape: ", self.images[0].shape)

            cv2.imwrite(self.image, "img_dump.png")

        for id_ in sorted(self.unused[0], key=lambda x: -x):
            layers.pop(id_)

        for i in range(0, self.num):
            data = np.dstack((layers))

        # reshape the image so it contains 12*n-tuples, each descripting a features of a single pixel
        #     on all layers in the pyramid
        h, w, c = self.image.shape
        data.shape = ((h * w, len(layers)))

        # print self.unused, self.rfc.feature_importances_

        # remove features that were found unnecessary
        # filtered = get_filtered_model(self.unused, data)

        if self.print_times:
            print("data preparation time: ", time.time() - t)

        # predict result on current image data
        start = time.time()
        mask1 = self.rfc.predict_proba(data)
        if self.print_times:
            print("RFC predict takes     %f" % (time.time() - start))
            print(data.shape)

        # reshape mask to be a grid, not a list
        mask1 = mask1[:, 1]
        mask1.shape = ((h, w))

        return mask1

    def train_raw_(self, X, y):
        """ Create the RFC classifier from raw X and y data
        """
        from sklearn.ensemble import RandomForestClassifier

        # create the classifier
        self.rfc = RandomForestClassifier(n_estimators=self.rfc_n_estimators, n_jobs=self.rfc_n_jobs)

        # train the classifier
        start = time.time()
        self.rfc.fit(X, y)
        if self.print_times:
            print("RFC fitting takes     %f" % (time.time() - start))

        # find unused features and remove them
        # create new classifier with less features, it will be faster
        start = time.time()
        self.unused = find_unused_features(self.rfc)
        self.rfc = get_filtered_rfc(self.unused, X, y)
        if self.print_times:
            print("RFC filtering takes   %f. Using %d out of %d features." % \
                  (time.time() - start, len(self.rfc.feature_importances_), len(X[0])))

    def get_features(self):
        """
        Gets all 12 features computed on the image in one array, ready for rfc.predict
        Lowest (largest) level of the pyramid is first, then there are all features from the second layer and so on
        :return: (h*w, 12*n) shaped array
        """
        result = []
        h, w, c = self.pyramid[0].shape

        if not self.use_reduced_feature_set:
            for i in range(0, self.num):
                result.append(self.images[i][:, :, 2].reshape((h * w, 1)))
                result.append(self.images[i][:, :, 1].reshape((h * w, 1)))
                result.append(self.images[i][:, :, 0].reshape((h * w, 1)))
                result.append(self.avg[i].reshape((h * w, 1)))
                result.append(self.shiftx[i].reshape((h * w, 1)))
                result.append(self.shifty[i].reshape((h * w, 1)))
                result.append(self.bg[i].reshape((h * w, 1)))
                result.append(self.gr[i].reshape((h * w, 1)))
                result.append(self.rb[i].reshape((h * w, 1)))
                result.append(self.maxs[i].reshape((h * w, 1)))
                result.append(self.mins[i].reshape((h * w, 1)))
                result.append(self.diff[i].reshape((h * w, 1)))
        else:
            for i in range(0, self.num):
                result.append(self.images[i][:, :, 2].reshape((h * w, 1)))
                # result.append(self.images[i][:, :, 1].reshape((h * w, 1)))
                # result.append(self.images[i][:, :, 0].reshape((h * w, 1)))
                # result.append(self.avg[i].reshape((h * w, 1)))
                result.append(self.shiftx[i].reshape((h * w, 1)))
                result.append(self.shifty[i].reshape((h * w, 1)))
                # result.append(self.bg[i].reshape((h * w, 1)))
                # result.append(self.gr[i].reshape((h * w, 1)))
                # result.append(self.rb[i].reshape((h * w, 1)))
                # result.append(self.maxs[i].reshape((h * w, 1)))
                # result.append(self.mins[i].reshape((h * w, 1)))
                result.append(self.diff[i].reshape((h * w, 1)))

        return result

    def update_xy(self):
        """
        Saves current X and y data to permanent storage, this should be done once all data is correct and frame is
        going to be switched soon.
        :return: None
        """
        # append temporary data to X and y
        self.X.extend(self.Xtmp)
        self.y.extend(self.ytmp)

    def get_rfc(self):
        """
        Returns current trained RFC
        :return: RFC
        """
        return self.rfc

    def make_pyramid(self):
        """
        Creates an image pyramid from current image
        :return: pyramid
        """
        result = []
        layers = self.num
        for (i, resized) in enumerate(pyramid_gaussian(self.image, downscale=self.scale)):
            if layers <= 0:  # stop when all layers are ready
                break
            layers -= 1

            # images are created as RGB 0.0 - 1.0 floats and need to be converted to 0-255 np.uint8
            foo = resized * 255
            bar = foo.astype(np.uint8)
            result.append(bar)
        return result

    def get_images(self):
        """
        Creates a set of RGB images from the pyramid, and rescales them to their original
        Therefore, higher layers have worst quality and bigger "pixels"
        :return: rescaled images from pyramid
        """
        result = []
        for i in range(0, len(self.pyramid)):
            result.append(self.get_scaled(self.pyramid[i], i))
        return result

    def get_blur(self, blur=33):
        """
        Creates blurred images. They are kept in their original pyramid shape (not rescaled)
        :param blur: blur kernel (33 by default)
        :return: blurred pyramid
        """
        result = []
        for i in range(0, len(self.pyramid)):
            result.append(cv2.GaussianBlur(self.pyramid[i], (blur, blur), 0))
        return result

    def get_edges(self, param_1=0, param_2=37):
        """
        Finds edges on pyramid images
        :param param_1: Canny filter first parameter
        :param param_2: Canny filter second parameter
        :return: list of images with edges - rescaled to original shape
        """

        blur_image = self.get_blur()   # blurred images pyramid (different scales!)
        result = []
        # find edges on the blurred image
        for i in range(0, len(self.pyramid)):
            canny = cv2.Canny(blur_image[i], param_1, param_2)
            result.append(self.get_scaled(canny, i))
        return result

    def get_shift(self, shift_x=2, shift_y=2):
        """
        Shifts all images in the pyramid.
        :param shift_x: 2 by default
        :param shift_y: 2 by default
        :return: Shifted imaged, rescaled to original dimensions
        """
        result = []
        for i in range(0, len(self.pyramid)):
            shifted = get_shift_im(self.pyramid[i], shift_x=shift_x, shift_y=shift_y)
            result.append(self.get_scaled(shifted, i))
        return result

    def get_avg(self):
        """
        Gets average values on each pixel (and it's 4 neighbours) in every image of the pyramid
        :return: pyramid average pixel value images, rescaled to original shape
        """
        result = []
        for i in range(0, len(self.pyramid)):
            shift_up = get_shift_im(self.pyramid[i], shift_x=-1, shift_y=0)
            shift_down = get_shift_im(self.pyramid[i], shift_x=1, shift_y=0)
            shift_left = get_shift_im(self.pyramid[i], shift_x=0, shift_y=-1)
            shift_right = get_shift_im(self.pyramid[i], shift_x=0, shift_y=1)
            img = cv2.cvtColor(self.pyramid[i], cv2.COLOR_BGR2GRAY)
            img_sum = shift_up + shift_down + shift_left + shift_right + img
            result.append(self.get_scaled(old_div(img_sum, 5), i))
        return result

    def get_dif(self):
        """
        Gets miscellaneous features for each image
        :return: 3 pyramids, each with one feature (max, min, diff)
        """
        result1 = []
        # result2 = []
        # result3 = []
        for i in range(0, len(self.pyramid)):
            t = time.time()
            image = cv2.cvtColor(self.pyramid[i], cv2.COLOR_BGR2GRAY)
            # shift_up = get_shift_im(self.pyramid[i], shift_x=-1, shift_y=0)
            # shift_down = get_shift_im(self.pyramid[i], shift_x=1, shift_y=0)
            # shift_left = get_shift_im(self.pyramid[i], shift_x=0, shift_y=-1)
            # shift_right = get_shift_im(self.pyramid[i], shift_x=0, shift_y=1)

            shift_up = np.zeros_like(image)
            shift_up[0:-1, :] = image[1:, :].copy()

            shift_down = np.zeros_like(image)
            shift_down[1:, :] = image[0:-1, :].copy()
            shift_left = np.zeros_like(image)
            shift_left[:, 0:-1] = image[:, 1:].copy()
            shift_right = np.zeros_like(image)
            shift_right[:, 1:] = image[:, 0:-1].copy()

            dif_up = np.asarray(image, dtype=np.int32) - np.asarray(shift_up, dtype=np.int32)
            dif_down = np.asarray(image, dtype=np.int32) - np.asarray(shift_down, dtype=np.int32)
            dif_left = np.asarray(image, dtype=np.int32) - np.asarray(shift_left, dtype=np.int32)
            dif_right = np.asarray(image, dtype=np.int32) - np.asarray(shift_right, dtype=np.int32)

            difs = np.dstack((dif_up, dif_down, dif_left, dif_right))
            maxs = np.amax(difs, axis=2)
            mins = np.amin(difs, axis=2)

            diff = np.asarray(maxs, dtype=np.int32) - np.asarray(mins, dtype=np.int32)
            # result1.append(self.get_scaled(maxs, i))
            # result2.append(self.get_scaled(mins, i))
            result1.append(self.get_scaled(diff, i))

        # return result1, result2, result3
        return result1

    def get_cdiff(self, c1, c2):
        """
        Gets difference between color channels in 3 channel images
        :param c1: first channel id
        :param c2: second channel id
        :return: pyramid of dif images, scaled to original shape
        """
        result = []
        for i in range(0, len(self.pyramid)):
            im = np.asarray(self.pyramid[i][:, :, c1], dtype=np.int32)\
                 - np.asarray(self.pyramid[i][:, :, c2], dtype=np.int32)
            result.append(self.get_scaled(im, i))
        return result

    def get_scaled(self, im, i):
        """
        Scales the image to its original size
        :param im: small image
        :param i: pyramid layer
        :return: scaled image
        """
        s = self.scale ** i
        if len(im.shape) == 2:
            # all images must have 3 dimensions to be scaled successfully
            w, h = im.shape
            im.shape = ((w, h, 1))
        im = np.asarray(im, dtype=np.uint8)
        return cv2.resize(im, (self.w, self.h))


def get_shift_im(im, shift_x=2, shift_y=2):
    """
    Shifts the one given image in the given direction
    :param im: source image
    :param shift_x: X-axis pixel shift (2 by default)
    :param shift_y: y-axis pixel shift (2 by default)
    :return: gray scale shifted image
    """
    w, h, c = im.shape

    # create shift matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    # apply shift matrix
    img2 = cv2.warpAffine(im, M, (h, w))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return img2


def find_unused_features(rfc, threshold=0.01):
    """
    Find indexes of insignificant features
    :param rfc: trained random forest classifier
    :param threshold: features with importance lower than threshold will be found useless
    :return: array indices of unused features
    """
    return np.where(rfc.feature_importances_ < threshold)


def get_filtered_model(zeros, data):
    """
    Removes unused features from data
    :param zeros: List of indices with to-be-deleted features
    :param data: all features data
    :return: new data array without some features
    """
    return np.delete(data, zeros, axis=1)


def get_filtered_rfc(zeros, X, y, n_estimators, n_jobs):
    """
    Trains a new RFC with less features
    :param zeros: features to be removed from rfc
    :param X: training data - features
    :param y: training data - classifications
    :return: new faster RFC with less features
    """
    newX = []
    for tup in X:
        newtup = np.delete(tup, zeros)
        newX.append(newtup)

    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
    return rfc.fit(newX, y)


def get_lbp(image, method="uniform"):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    w = width = radius - 1

    # get lbg
    lbp = local_binary_pattern(gray_image, n_points, radius, method)

    # get edge_labels
    edge_labels = list(range(n_points // 2 - w, n_points // 2 + w + 1))
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    i_14 = n_points // 4            # 1/4th of the histogram
    i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
    corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                     list(range(i_34 - w, i_34 + w + 1)))

    mask = np.logical_or.reduce([lbp == each for each in edge_labels])

    # import matplotlib.pyplot as plt
    # plt.imshow(mask)
    # plt.show()

    # mask = np.logical_or.reduce([lbp == each for each in flat_labels])

    # plt.imshow(mask)
    # plt.show()

    # mask = np.logical_or.reduce([lbp == each for each in corner_labels])

    # plt.imshow(mask)
    # plt.show()
    return lbp

    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)

# import the necessary packages
def pyramid(image, scale=1.5, minSize=(30, 30), num=-1):
    # yield the original image

    results = [image]
    i = 0
    # keep looping over the pyramid
    while True and (num < 0 or i < num):
        i += 1
        # compute the new dimensions of the image and resize it
        w = int(old_div(image.shape[1], scale))
        h = int(old_div(image.shape[0], scale))
        image = cv2.resize(image, (w, h))

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        results.append(image)

    return results


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # image = cv2.imread("/home/dita/img_67.png")
    image = cv2.imread("/home/dita/lbp_test.png")
    np.set_printoptions(threshold=np.inf)
    print(old_div(image[:, :, 0], 255))
    methods = ["nri_uniform", "default", "ror", "uniform", "var"]

    lbp = get_lbp(image)
    print(lbp)
    cv2.imwrite("/home/dita/lbp_test_out.png", lbp*(old_div(255,lbp.max())))

    print(lbp.shape)

    plt.imshow(lbp > 23)
    plt.show()
