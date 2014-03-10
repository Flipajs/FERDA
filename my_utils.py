__author__ = 'flip'

import math
import cv2


def is_inside_ellipse(el, point):
    rx = el.size.width / 2
    ry = el.size.height / 2

    f = math.sqrt(rx * rx - ry * ry)
    x1 = point.x - el.center.x - f
    x2 = point.y - el.center.y
    x3 = point.x - el.center.x + f
    x4 = point.y - el.center.y

    return math.sqrt(x1 * x1 + x2 * x2) + math.sqrt(x3 * x3 + x4 * x4) < 2 * rx


class Point():
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        p = Point(0, 0)
        p.x = self.x + other.x
        p.y = self.y + other.y
        return p

    def __sub__(self, other):
        p = Point(0, 0)
        p.x = self.x - other.x
        p.y = self.y - other.y
        return p

    def int_tuple(self):
        t = (int(self.x), int(self.y))
        return t


class Size():
    width = 0
    height = 0

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


class RotatedRect():
    center = Point(0, 0)
    size = Size(0, 0)
    angle = 0

    def __init__(self, center, size, angle):
        self.center = center
        self.size = size
        self.angle = angle


def mser_main_axis_rate(sxy, sxx, syy):
    la = (sxx + syy) / 2
    lb = math.sqrt(4 * sxy * sxy + (sxx - syy) * (sxx - syy)) / 2

    lambda1 = math.sqrt(la + lb)
    lambda2 = math.sqrt(la - lb)

    return lambda1 / lambda2, lambda1, lambda2


def e_distance(p1, p2):
    p = p1 - p2
    return math.sqrt(p.x * p.x + p.y * p.y)


def mser_theta(sxy, sxx, syy):
    theta = 0.5*math.atan2(2*sxy, (sxx - syy))
    #it must be reversed around X because in image is top left corner [0, 0] and it is not very intuitive
    theta = -theta
    if theta < 0:
        theta += math.pi

    return theta
    #return 0.5*math.atan(2 * sxy / (sxx - syy))

def imshow(title, img, imshow_decreasing_factor = 1):
    img = cv2.resize(img, (int(img.shape[1]*imshow_decreasing_factor),
                           int(img.shape[0]*imshow_decreasing_factor)))

    cv2.imshow(title, img)
