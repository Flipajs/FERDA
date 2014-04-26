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


def mser_main_axis_ratio(sxy, sxx, syy):
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

def best_margin(regions, r_ids):
    best_margin = -1
    best_margin_id = -1
    for r_id in r_ids:
        if regions[r_id]['margin'] > best_margin:
            best_margin = regions[r_id]['margin']
            best_margin_id = r_id

    return best_margin, best_margin_id


def count_head_tail(area, a, b):
    axis_ratio = a / float(b)
    b_ = math.sqrt(area / (axis_ratio * math.pi))
    a_ = b_ * axis_ratio

    return a_, b_


def get_circle_line_intersection(params, c, p):
    m = Point(params.arena.center.x, params.arena.center.y) #circle middle
    r = params.arena.size.width / 2 #circle radius
    x = Point(0, 0)

    a_ = c.x**2 - 2*p.x*c.x + p.x**2
    a_ += c.y**2 - 2*p.y*c.y + p.y**2

    b_ = 2*c.x*p.x - 2*c.x*m.x - 2*p.x**2 + 2*p.x*m.x
    b_ += 2*c.y*p.y - 2*c.y*m.y - 2*p.y**2 + 2*p.y*m.y

    c_ = p.x**2 - 2*p.x*m.x + m.x**2
    c_ += p.y**2 - 2*p.y*m.y + m.y**2
    c_ -= r**2


    if a_ == 0 or b_**2 - 4*a_*c_ < 0:
        return p

    d_ = math.sqrt(b_**2 - 4*a_*c_)
    alpha1 = (-b_ + d_) / (2*a_)
    alpha2 = (-b_ - d_) / (2*a_)

    alpha = 0
    if 0 <= alpha1 <= 1:
        alpha = alpha1
    elif 0 <= alpha2 <= 1:
        alpha = alpha2

    x.x = alpha * c.x + (1 - alpha) * p.x
    x.y = alpha * c.y + (1 - alpha) * p.y

    return x