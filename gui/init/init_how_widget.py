__author__ = 'fnaiser'

from PyQt4 import QtGui
from gui.img_grid.img_grid_widget import ImgGridWidget
from gui.img_grid.img_grid_dialog import ImgGridDialog
from utils.video_manager import get_auto_video_manager
from core.region.mser import get_msers_
from PIL import ImageQt
from utils.drawing.points import draw_points_crop, get_contour
import numpy as np
from skimage.transform import resize
from gui.gui_utils import SelectableQLabel
from matplotlib.mlab import normpdf
from numpy.linalg import norm
from core import colormark
from gui.view.assignment_widget import AssignmentWidget
import cv2

class InitHowWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project):
        super(InitHowWidget, self).__init__()

        self.finish_callback = finish_callback
        self.project = project

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.b = QtGui.QPushButton('reshape')
        self.b.clicked.connect(lambda: self.image_grid_widget.reshape(10))
        self.vbox.addWidget(self.b)

        self.give_me_selected_b = QtGui.QPushButton('selected ?')
        self.give_me_selected_b.clicked.connect(self.give_me_selected)
        self.vbox.addWidget(self.give_me_selected_b)

        vid = get_auto_video_manager(project.video_paths)

        img = vid.move2_next()
        img = self.fill_colormarks(img)

        self.items = []
        self.init_ants(img)

        self.assignment_widget = AssignmentWidget(project.animals, img=img)
        self.vbox.addWidget(self.assignment_widget)




        d = ImgGridDialog(self, self.items)
        d.confirmed.connect(self.grid_dialog_selection_confirmed)
        d.show()

        # cv2.imshow('filled', img)

        # for i in range(1):
        #     img = vid.move2_next()
        #     img_ = project.bg_model.bg_subtraction(img)
        #
        #     msers = get_msers_(img_)
        #
        #     for j in range(len(msers)):
        #         item = self.get_img_qlabel(msers[j].pts(), img, id, height, width)
        #
        #         self.image_grid_widget.add_item(item)
        #
        # print [r.major_axis_ for r in msers]
        # print [r.minor_axis_ for r in msers]
        #
        # print [r.a_ for r in msers]
        # print [r.b_ for r in msers]

    def give_me_selected(self):
        print self.image_grid_widget.get_selected()

    def dist_score(self, animal, region):
        # half the radius
        std = norm(animal.init_pos_head_ - animal.init_pos_center_) * 0.5

        d = norm(animal.init_pos_center_ - region.centroid())

        max_val = normpdf(0, 0, std)
        s = normpdf(d, 0, std) / max_val

        return s

    def axis_length_score(self, animal, region):
        mean = norm(animal.init_pos_head_ - animal.init_pos_center_)
        std = mean * 0.25

        major_axis = region.a_

        max_val = normpdf(mean, mean, std)
        s = normpdf(major_axis, mean, std) / max_val

        return s


    def margin_score(self, region):
        #TODO REMOVE CONSTANT!
        return min(region.margin() / 30.0, 1)

    def fill_colormarks(self, img):
        img_ = np.copy(img)

        for a_id in range(len(self.project.animals)):
            animal = self.project.animals[a_id]
            c = colormark.get_colormark(img_, animal.color_, animal.init_pos_center_, 200)

            p = np.asarray(c.pts_, dtype=np.int32)
            img[p[:,0], p[:,1], :] = (0, 0, 0)

        return img

    def init_ants(self, img):
        img_ = self.project.bg_model.bg_subtraction(img)

        msers = get_msers_(img_)

        margin_score = np.array([self.margin_score(r) for r in msers])

        # for a_id in range(len(self.project.animals)):
        for a_id in range(1):
            a = self.project.animals[a_id]

            d_score = np.array([self.dist_score(a, r) for r in msers])
            axis_score = np.array([self.axis_length_score(a, r) for r in msers])

            # s = margin_score * d_score * axis_score
            s = margin_score * axis_score
            # s = d_score * margin_score
            order = np.argsort(-s)

            for i in range(len(msers)):
                item = self.get_img_qlabel(msers[order[i]].pts(), img, a_id)
                self.items.append(item)
            #
            # self.image_grid_widget.add_item(item)


    def get_img_qlabel(self, pts, img, id, height=100, width=100):
        cont = get_contour(pts)
        crop = draw_points_crop(img, cont, (0, 0, 255, 0.5), square=True)

        img_q = ImageQt.QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, 13)
        pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

        item = SelectableQLabel(id=id)

        item.setScaledContents(True)
        item.setFixedSize(height, width)
        item.setPixmap(pix_map)

        return item

    def grid_dialog_selection_confirmed(self, ids):
        print "SELECTED ID: ", ids