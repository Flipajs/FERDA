import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage.transform import pyramid_gaussian
import scipy.ndimage


class Helper:
    def __init__(self, image, num=4):
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
        self.num = num

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
        self.set_image(image)

    def set_image(self, image):
        self.image = image
        self.pyramid = get_pyramid(self.image, layers=self.num)
        self.images = get_images(self.pyramid)
        blur_image = get_blur(self.pyramid)

        # find edges on the blurred image
        self.edges = get_edges(blur_image)

        # original img - shifted img
        self.shiftx = get_shift(self.pyramid, shift_x=2, shift_y=0)
        self.shifty = get_shift(self.pyramid, shift_x=0, shift_y=2)

        self.avg = get_avg(self.pyramid)

        self.maxs, self.mins, self.diff = get_dif(self.pyramid)

        # channel difs
        self.bg = get_cdiff(self.pyramid, 0, 1)
        self.gr = get_cdiff(self.pyramid, 1, 2)
        self.rb = get_cdiff(self.pyramid, 2, 0)
        pass

    def get_data(self, i, j, X, y, classification):
        x = []
        for k in range(0, self.num):
            b, g, r = self.images[k][i][j] # TODO out of bounds here
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
        X.append(x)
        y.append(classification)

    def done(self, background, foreground, rfc=None):
        if rfc is None:
            # prepare learning data
            # X contains tuples of data for each evaluated unit-pixel (R, G, B, edge?)
            # y contains classifications for all pixels respectively
            self.Xtmp = []
            self.ytmp = []

            # loop all nonzero pixels from foreground (ants) and background and add them to testing data
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

            # create the classifier
            self.rfc = RandomForestClassifier(class_weight='balanced')

            # to train on all data (current and all previous frames), join the arrays together
            # class variables are not affected here
            X = list(self.Xtmp)
            X.extend(self.X)
            y = list(self.ytmp)
            y.extend(self.y)
            self.rfc.fit(X, y)
        else:
            self.rfc = rfc

        h, w, c = self.image.shape

        layers = self.get_layers()
        for i in range(0, self.num):
            data = np.dstack((layers))

        # reshape the image so it contains 4-tuples, each descripting a single pixel
        data.shape = ((h * w, 12*self.num))

        # prepare a mask and predict result for data (current image)
        # mask1 = np.zeros((h*w, c))
        mask1 = self.rfc.predict_proba(data)

        # reshape mask to be a grid, not a list
        mask1 = mask1[:, 1]
        mask1.shape = ((h, w))

        return mask1

    def get_layers(self):
        result = []
        h, w, c = self.pyramid[0].shape
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
        return result

    def update_xy(self):
        # append temporary data to X and y
        self.X.extend(self.Xtmp)
        self.y.extend(self.ytmp)

    def get_rfc(self):
        return self.rfc


def get_pyramid(image, scale=2, layers=4):
    result = []
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale=scale)):
        if layers <= 0:
            break
        layers -= 1

        foo = resized * 255
        bar = foo.astype(np.uint8)
        result.append(bar)
        pass

    return result


def get_images(pyramid):
    result = []
    for i in range(0, len(pyramid)):
        result.append(get_scaled(pyramid[i], i))
    return result


def get_blur(pyramid):
    result = []
    # create a blurred image
    blur = 33
    for i in range(0, len(pyramid)):
        result.append(cv2.GaussianBlur(pyramid[i], (blur, blur), 0))
    return result


def get_shift_im(im, blur_kernel=3, blur_sigma=0.3, shift_x=2, shift_y=2):
    w, h, c = im.shape

    # prepare first image (original), make it grayscale and blurred
    img1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img1, (blur_kernel, blur_kernel), blur_sigma)

    # create shift matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    # apply shift matrix
    img2 = cv2.warpAffine(im, M, (w, h))

    # prepare second image (translated), make it grayscale and blurred
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.GaussianBlur(img2, (blur_kernel, blur_kernel), blur_sigma)

    # get dif image
    return np.abs(np.asarray(img1, dtype=np.int32) - np.asarray(img2, dtype=np.int32))


def get_edges(pyramid):
    result = []
    # find edges on the blurred image
    a = 0
    b = 37
    for i in range(0, len(pyramid)):
        canny = cv2.Canny(pyramid[i], a, b)
        result.append(get_scaled(canny, i))
    return result


def get_shift(pyramid, blur_kernel=3, blur_sigma=0.3, shift_x=2, shift_y=2):
    result = []
    for i in range(0, len(pyramid)):
        shifted = get_shift_im(pyramid[i], blur_kernel=blur_kernel, blur_sigma=blur_sigma,
                               shift_x=shift_x, shift_y=shift_y)
        result.append(get_scaled(shifted, i))
    return result


def get_avg(pyramid):
    result = []
    for i in range(0, len(pyramid)):
        shift_up = get_shift_im(pyramid[i], shift_x=-1, shift_y=0)
        shift_down = get_shift_im(pyramid[i], shift_x=1, shift_y=0)
        shift_left = get_shift_im(pyramid[i], shift_x=0, shift_y=-1)
        shift_right = get_shift_im(pyramid[i], shift_x=0, shift_y=1)
        img = cv2.cvtColor(pyramid[i], cv2.COLOR_BGR2GRAY)
        img_sum = shift_up + shift_down + shift_left + shift_right + img
        result.append(get_scaled(img_sum / 5, i))
    return result


def get_dif(pyramid):
    result1 = []
    result2 = []
    result3 = []
    for i in range(0, len(pyramid)):
        shift_up = get_shift_im(pyramid[i], shift_x=-1, shift_y=0)
        shift_down = get_shift_im(pyramid[i], shift_x=1, shift_y=0)
        shift_left = get_shift_im(pyramid[i], shift_x=0, shift_y=-1)
        shift_right = get_shift_im(pyramid[i], shift_x=0, shift_y=1)
        image = cv2.cvtColor(pyramid[i], cv2.COLOR_BGR2GRAY)

        dif_up = np.asarray(image, dtype=np.int32) - np.asarray(shift_up, dtype=np.int32)
        dif_down = np.asarray(image, dtype=np.int32) - np.asarray(shift_down, dtype=np.int32)
        dif_left = np.asarray(image, dtype=np.int32) - np.asarray(shift_left, dtype=np.int32)
        dif_right = np.asarray(image, dtype=np.int32) - np.asarray(shift_right, dtype=np.int32)

        difs = np.dstack((dif_up, dif_down, dif_left, dif_right))
        maxs = np.amax(difs, axis=2)
        mins = np.amin(difs, axis=2)

        diff = np.asarray(maxs, dtype=np.int32) - np.asarray(mins, dtype=np.int32)
        result1.append(get_scaled(maxs, i))
        result2.append(get_scaled(mins, i))
        result3.append(get_scaled(diff, i))

    return result1, result2, result3


def get_cdiff(pyramid, c1, c2):
    result = []
    for i in range(0, len(pyramid)):
        im = np.asarray(pyramid[i][:,:,c1], dtype=np.int32) - np.asarray(pyramid[i][:,:,c2], dtype=np.int32)
        result.append(get_scaled(im, i))
    return result


def get_scaled(im, i):
    scale = 2**i
    if len(im.shape) == 2:
        w, h = im.shape
        im.shape = ((w, h, 1))
    return np.asarray(scipy.ndimage.zoom(im, (scale, scale, 1), order=0), dtype=np.uint8)


if __name__ == "__main__":
    image = cv2.imread("/home/dita/img_67.png")
    # scale_test(image)