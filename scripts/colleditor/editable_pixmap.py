__author__ = 'filip@naiser.cz'

from PyQt4.QtGui import *
import numpy as np
import math
from utils import geometry


class EditablePixmap:
    #class for editable binary qt pix_map
    def __init__(self, pts, scene, img_width, img_height, color=QColor(0, 255, 255, 90).rgba(), centroid=None):
        self.original_pts = list(pts)
        self.pts = list(pts)
        self.color = color
        self.pixmap = None  # QPixmap
        self.scene = scene  # QtGui.QGraphicsScene

        self.img_width = img_width
        self.img_height = img_height

        self.update_pixmap()
        self.centroid = centroid
        if not self.centroid:
            self.centroid = geometry.count_centroid(self.pts)

    def add_points(self, new_pts):
        for pt in new_pts:
            if pt not in self.pts:
                self.pts.append([pt[0], pt[1]])

        self.update_pixmap()

    def remove_points(self, new_pts):
        for pt in new_pts[:]:
            try:
                self.pts.remove([pt[0], pt[1]])
            except ValueError:
                pass

        self.update_pixmap()

    def update_pixmap(self):
        if self.pixmap is not None:
            self.scene.removeItem(self.pixmap)
            self.pixmap = None

        pixmap = self.pixmap_from_pts(self.pts)

        self.pixmap = self.scene.addPixmap(pixmap)

    def pixmap_from_pts(self, pts):
        img_q = self.draw_pts(pts)
        pix_map = QPixmap.fromImage(img_q)

        return pix_map

    def draw_pts(self, pts):
        img_q = QImage(self.img_width, self.img_height, QImage.Format_ARGB32)
        img_q.fill(QColor(0, 0, 0, 0).rgba())

        for pt in pts:
            img_q.setPixel(pt[0], pt[1], self.color)

        return img_q

    def translate(self, x, y):
        self.pts = [[el[0]+x, el[1]+y] for el in self.pts]
        self.centroid[0] += x
        self.centroid[1] += y
        self.update_pixmap()

    def rotate(self, theta_radians, method='back_projection'):
        self.pts = geometry.rotate(
            self.pts, theta_radians, self.centroid, method
        )

        self.update_pixmap()


def nearest_free_neighbour(pts, pt):
    """Searches eight neighbours of pt in pts, picks
    nearest free (missing in pts) pt if exists, else
    returns None
    """
    x = int(pt[0])
    y = int(pt[1])
    free = []
    for x_i in range(x - 1, x + 1 + 1):  # range(0, 2) -> [0, 1]
        for y_i in range(y - 1, y + 1 + 1):
            if [x_i, y_i] not in pts:
                free.append([[x_i, y_i], math.hypot(x-x_i, y-y_i)])

    best_pt = None
    best_d = 10

    for f in free:
        if f[1] < best_d:
            best_pt = f[0]

    return best_pt