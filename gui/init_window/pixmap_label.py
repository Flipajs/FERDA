__author__ = 'flipajs'

from PyQt4 import QtGui, QtCore
import ImageQt
import visualize
import ant
import cv2
from numpy import *


try:
    cv_line_type = cv2.LINE_AA
except:
    cv_line_type = cv2.CV_AA


class PixmapLabel(QtGui.QLabel):
    def __init__(self, window_p, img, region, r_id, update_graphics_view, ant_assignment):
        super(PixmapLabel, self).__init__()

        self.window_p = window_p
        self.img = img
        self.region = region
        self.r_id = r_id

        self.default_c = (230, 230, 230)
        self.ant_number = 0

        self.init_image()
        self.update_graphics_view = update_graphics_view
        self.ant_assignment = ant_assignment

    def init_image(self):
        pix_map = self.get_pixmap(self.default_c)

        self.setScaledContents(True)

        self.update_pixmap(pix_map)

    def update_pixmap(self, pix_map):
        self.setFixedWidth(pix_map.width())
        self.setFixedHeight(pix_map.height())
        self.setPixmap(pix_map)

    def get_pixmap(self, c=None):
        r = self.region

        cell_size = 70
        border = cell_size

        img_ = zeros((shape(self.img)[0] + 2 * border, shape(self.img)[1] + 2 * border, 3), dtype=uint8)
        img_[border:-border, border:-border] = self.img.copy()

        if c:
            visualize.draw_region(img_[border:-border, border:-border], r, c, contour=False)

        img_small = img_[border + r["cy"] - cell_size / 2:border + r["cy"] + cell_size / 2, border + r["cx"] - cell_size / 2:border + r["cx"] + cell_size / 2].copy()

        if self.ant_number > 0:
            cv2.putText(img_small, str(self.ant_number), (3, 12), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 1, cv_line_type)

        img_q = ImageQt.QImage(img_small.data, img_small.shape[1], img_small.shape[0], img_small.shape[1]*3, 13)
        pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

        return pix_map

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:

            i = self.window_p.actual_focus
            if i < 0:
                i = 0
                while i < len(self.ant_assignment):
                    if not self.ant_assignment[i]:
                        break
                    i += 1

            if i >= len(self.ant_assignment):
                return

            self.ant_assignment[i].append(self.r_id)

            self.ant_number += 1
            pix_map = self.get_pixmap(ant.get_color(i, len(self.ant_assignment)))

            button = self.window_p.object_init_layout.itemAt(i).widget()
            button_val = int(button.text())
            if button_val > -1:
                j = 0
                for r_id in self.window_p.chosen_regions_indexes:
                    if r_id == button_val:
                        break

                    j += 1

                self.ant_assignment[i].remove(button_val)

                r, c, _, _ = self.window_p.init_mser_grid.getItemPosition(j)
                prev_pix_map = self.window_p.init_mser_grid.itemAtPosition(r, c).widget()

                prev_pix_map_ = prev_pix_map.get_pixmap(self.default_c)
                prev_pix_map.update_pixmap(prev_pix_map_)
                prev_pix_map.ant_number -= 1

            self.window_p.object_init_layout.itemAt(i).widget().setText(str(self.r_id))
            self.update_pixmap(pix_map)

            self.window_p.assign_ant(i, self.r_id)


        elif e.button() == QtCore.Qt.RightButton:
            for i in reversed(range(len(self.ant_assignment))):
                if self.r_id in self.ant_assignment[i]:
                    self.ant_assignment[i].remove(self.r_id)
                    self.window_p.object_init_layout.itemAt(i).widget().setText(str(-1))
                    self.window_p.assign_ant(i, -1)
                    break

            if self.ant_number > 0:
                self.ant_number -= 1


            c = self.default_c
            for i in reversed(range(len(self.ant_assignment))):
                if self.r_id in self.ant_assignment[i]:
                    c = ant.get_color(i, len(self.ant_assignment))
                    break

            pix_map = self.get_pixmap(c)
            self.update_pixmap(pix_map)



        self.update_graphics_view()

    def add_ant(self):
        i = self.window_p.actual_focus
        if i < 0:
            i = 0
            while i < len(self.ant_assignment):
                if not self.ant_assignment[i]:
                    break
                i += 1

        if i >= len(self.ant_assignment):
            return

        self.ant_assignment[i].append(self.r_id)

        self.ant_number += 1

        pix_map = self.get_pixmap(ant.get_color(i, len(self.ant_assignment)))

        # button = self.window_p.object_init_layout.itemAt(i).widget()
        # button_val = int(button.text())
        # if button_val > -1:
        #     j = 0
        #     for r_id in self.window_p.chosen_regions_indexes:
        #         if r_id == button_val:
        #             break
        #
        #         j += 1
        #
        #     self.ant_assignment[i].remove(button_val)
        #
        #     r, c, _, _ = self.window_p.init_mser_grid.getItemPosition(j)
        #     prev_pix_map = self.window_p.init_mser_grid.itemAtPosition(r, c).widget()
        #
        #     prev_pix_map_ = prev_pix_map.get_pixmap(self.default_c)
        #     prev_pix_map.update_pixmap(prev_pix_map_)
        #     prev_pix_map.ant_number -= 1

        self.window_p.object_init_layout.itemAt(i).widget().setText(str(self.r_id))
        self.update_pixmap(pix_map)

        self.window_p.assign_ant(i, self.r_id)