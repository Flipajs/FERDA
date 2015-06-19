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
from gui.img_grid.img_grid_widget import ImgGridWidget
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
from core.classes_stats import ClassesStats
from core.settings import Settings as S_
from skimage.transform import rescale
import scipy


class InitHowWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project):
        super(InitHowWidget, self).__init__()

        self.finish_callback = finish_callback
        self.project = project

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        vid = get_auto_video_manager(project.video_paths)

        img = vid.move2_next()
        # img = self.fill_colormarks(img)

        self.items = []

        # self.init_ants(img)

        self.assignment_widget = AssignmentWidget(project.animals, img=img)
        self.hlayout = QtGui.QHBoxLayout()
        self.vbox.addLayout(self.hlayout)
        self.hlayout.addWidget(self.assignment_widget)

        elem_width = 70
        self.img_grid = ImgGridWidget()
        self.img_grid.reshape(10, elem_width)
        self.hlayout.addWidget(self.img_grid)

        self.finish_how = QtGui.QPushButton('confirm selection and finish initialization')
        self.finish_how.clicked.connect(self.finish)
        self.vbox.addWidget(self.finish_how)

        self.regions = []

        r_id = 0
        for i in range(3):
            img = vid.move2_next()

            if project.bg_model:
                img_ = project.bg_model.bg_subtraction(img)
            else:
                img_ = img.copy()

            if project.arena_model:
                img_ = project.arena_model.mask_image(img_)

            if S_.mser.gaussian_kernel_std > 0:
                img_ = scipy.ndimage.gaussian_filter(img_, sigma=S_.mser.gaussian_kernel_std)

            if S_.mser.img_subsample_factor > 1.0:
                img_ = np.asarray(rescale(img_, 1/S_.mser.img_subsample_factor) * 255, dtype=np.uint8)
                img = np.asarray(rescale(img, 1/S_.mser.img_subsample_factor) * 255, dtype=np.uint8)

            msers = get_msers_(img_)
            groups = get_region_groups(msers)
            ids = margin_filter(msers, groups)

            for j in ids:
                item = self.get_img_qlabel(msers[j].pts(), img, r_id, elem_width, elem_width)
                r_id += 1
                self.regions.append(msers[j])

                self.img_grid.add_item(item)

        self.classes = [0 for i in range(len(self.regions))]

        self.class_stats = ClassesStats()

    def finish(self):
        selected = self.img_grid.get_selected()

        for i in selected:
            self.classes[i] = 1

        self.class_stats.compute_stats(self.regions, self.classes)

        self.finish_callback('init_how_finished', [self.class_stats])

    def give_me_selected(self):
        print self.img_grid.get_selected()

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