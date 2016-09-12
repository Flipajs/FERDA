import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class Helper:

    def __init__(self, image):
        self.image = image

        # create a blurred image
        blur = 33
        a = 0
        b = 37
        blur_image = cv2.GaussianBlur(self.image, (blur, blur), 0)

        # find edges on the blurred image
        self.edges = cv2.Canny(blur_image, a, b)

        # original img - shifted img
        self.shiftx = get_shift(self.image, shift_x=2, shift_y=0)
        self.shifty = get_shift(self.image, shift_x=0, shift_y=2)

        self.avg = get_avg(self.image)

        self.maxs, self.mins, self.diff = get_dif(self.image)

        # channel difs
        self.bg = np.asarray(self.image[:,:,0], dtype=np.int32) - np.asarray(self.image[:,:,1], dtype=np.int32)
        self.gr = np.asarray(self.image[:,:,1], dtype=np.int32) - np.asarray(self.image[:,:,2], dtype=np.int32)
        self.rb = np.asarray(self.image[:,:,2], dtype=np.int32) - np.asarray(self.image[:,:,0], dtype=np.int32)

    def get_data(self, i, j, X, y, classification):
        b, g, r = self.image[i][j]
        sx = self.shiftx[i][j]
        sy = self.shifty[i][j]
        a = self.avg[i][j]
        c = self.bg[i][j]
        d = self.gr[i][j]
        e = self.rb[i][j]
        f = self.maxs[i][j]
        h = self.mins[i][j]
        k = self.diff[i][j]
        X.append((b, g, r, a, sx, sy, c, d, e, f, h, k))
        y.append(classification)

    def done(self, background, foreground):
        # prepare learning data
        # X contains tuples of data for each evaluated unit-pixel (R, G, B, edge?)
        # y contains classifications for all pixels respectively
        X = []
        y = []

        # loop all nonzero pixels from foreground (ants) and background and add them to testing data
        nzero = np.nonzero(background[0])
        if len(nzero[0]) == 0:
            return None
        for i, j in zip(nzero[0], nzero[1]):
            self.get_data(i, j, X, y, 0)

        nzero = np.nonzero(foreground[0])
        if len(nzero[0]) == 0:
            return None
        for i, j in zip(nzero[0], nzero[1]):
            self.get_data(i, j, X, y, 1)

        # create the classifier
        rfc = RandomForestClassifier(class_weight='balanced')
        rfc.fit(X, y)

        h, w, c = self.image.shape

        data = np.dstack((self.image[:, :, 2].reshape((h * w, 1)),
                          self.image[:, :, 1].reshape((h * w, 1)),
                          self.image[:, :, 0].reshape((h * w, 1)),
                          self.avg.reshape((h * w, 1)),
                          self.shiftx.reshape((h * w, 1)),
                          self.shifty.reshape((h * w, 1)),
                          self.bg.reshape((h * w, 1)),
                          self.gr.reshape((h * w, 1)),
                          self.rb.reshape((h * w, 1)),
                          self.maxs.reshape((h * w, 1)),
                          self.mins.reshape((h * w, 1)),
                          self.diff.reshape((h * w, 1))))

        # reshape the image so it contains 4-tuples, each descripting a single pixel
        data.shape = ((h * w, 12))

        # prepare a mask and predict result for data (current image)
        # mask1 = np.zeros((h*w, c))
        mask1 = rfc.predict_proba(data)

        # reshape mask to be a grid, not a list
        mask1 = mask1[:, 1]
        mask1.shape = ((h, w))

        # create a rgba image from mask
        r = np.zeros((h, w), dtype=np.uint8)
        g = np.asarray(mask1 * 255, dtype=np.uint8)
        b = np.zeros((h, w), dtype=np.uint8)
        a = np.full((h, w), 100, dtype=np.uint8)
        rgb = np.dstack((b, g, r, a))

        return rgb


def get_shift(image, w=-1, h=-1, blur_kernel=3, blur_sigma=0.3, shift_x=2, shift_y=2):
    if w < 0 or h < 0:
        w, h, c = image.shape

    # prepare first image (original), make it grayscale and blurre
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img1, (blur_kernel, blur_kernel), blur_sigma)

    # create shift matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    # apply shift matrix
    img2 = cv2.warpAffine(image, M, (w, h))

    # prepare second image (tranlated), make it grayscale and blurred
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.GaussianBlur(img2, (blur_kernel, blur_kernel), blur_sigma)

    # get dif image
    dif = np.abs(np.asarray(img1, dtype=np.int32) - np.asarray(img2, dtype=np.int32))
    return dif


def get_avg(image):
    shift_up = get_shift(image, shift_x=-1, shift_y=0)
    shift_down = get_shift(image, shift_x=1, shift_y=0)
    shift_left = get_shift(image, shift_x=0, shift_y=-1)
    shift_right = get_shift(image, shift_x=0, shift_y=1)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_sum = shift_up + shift_down + shift_left + shift_right + img
    avg = img_sum / 5
    return avg


def get_dif(image):
    shift_up = get_shift(image, shift_x=-1, shift_y=0)
    shift_down = get_shift(image, shift_x=1, shift_y=0)
    shift_left = get_shift(image, shift_x=0, shift_y=-1)
    shift_right = get_shift(image, shift_x=0, shift_y=1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dif_up = np.asarray(image, dtype=np.int32) - np.asarray(shift_up, dtype=np.int32)
    dif_down = np.asarray(image, dtype=np.int32) - np.asarray(shift_down, dtype=np.int32)
    dif_left = np.asarray(image, dtype=np.int32) - np.asarray(shift_left, dtype=np.int32)
    dif_right = np.asarray(image, dtype=np.int32) - np.asarray(shift_right, dtype=np.int32)

    difs = np.dstack((dif_up, dif_down, dif_left, dif_right))
    maxs = np.amax(difs, axis=2)
    mins = np.amin(difs, axis=2)

    diff = np.asarray(maxs, dtype=np.int32) - np.asarray(mins, dtype=np.int32)
    return maxs, mins, diff