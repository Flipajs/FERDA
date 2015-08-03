import sys

__author__ = 'filip@naiser.cz'

from PyQt4 import QtGui, QtCore
from gui.img_grid.img_grid_widget import ImgGridWidget
from gui.gui_utils import get_image_label
from utils.video_manager import get_auto_video_manager
from core.project import Project
from core.region.mser import get_msers_
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
from utils.drawing.points import draw_points_crop, draw_points, get_contour
from PIL import ImageQt
from gui.gui_utils import SelectableQLabel
from skimage.transform import rescale
import numpy as np
from core.settings import Settings as S_
import scipy
from utils.img import prepare_for_segmentation
import time


class SetMSERs(QtGui.QWidget):
    def __init__(self, project):
        super(SetMSERs, self).__init__()
        self.project = project

        self.setLayout(QtGui.QVBoxLayout())

        self.w_ = QtGui.QWidget()
        self.w_.setLayout(QtGui.QVBoxLayout())
        self.scroll_ = QtGui.QScrollArea()
        self.scroll_.setWidgetResizable(True)
        self.scroll_.setWidget(self.w_)
        self.layout().addWidget(self.scroll_)

        self.top_row = QtGui.QHBoxLayout()
        self.bottom_row = QtGui.QFormLayout()
        self.scroll_.setLayout(QtGui.QVBoxLayout())

        self.w_.layout().addLayout(self.bottom_row)
        self.w_.layout().addLayout(self.top_row)

        self.vid = get_auto_video_manager(project.video_paths)
        # self.im = self.vid.move2_next()
        self.im = self.vid.seek_frame(659)

        im = self.im
        if self.project.bg_model:
            im = self.project.bg_model.bg_subtraction(im)

        if self.project.arena_model:
            im = self.project.arena_model.mask_image(im)

        self.im = im

        self.img_preview = get_image_label(self.im)
        self.top_row.addWidget(self.img_preview)
        self.img_grid = ImgGridWidget()
        self.top_row.addWidget(self.img_grid)

        self.mser_max_area = QtGui.QDoubleSpinBox()
        self.mser_max_area.setMinimum(0.0001)
        self.mser_max_area.setSingleStep(0.0001)
        self.mser_max_area.setMaximum(1.0)
        self.mser_max_area.setDecimals(6)
        self.mser_max_area.setValue(S_.mser.max_area)
        self.mser_max_area.valueChanged.connect(self.val_changed)
        self.bottom_row.addRow('MSER Max relative area', self.mser_max_area)

        self.mser_min_area = QtGui.QSpinBox()
        self.mser_min_area.setMinimum(0)
        self.mser_min_area.setMaximum(1000)
        self.mser_min_area.setValue(S_.mser.min_area)
        self.mser_min_area.valueChanged.connect(self.val_changed)
        self.bottom_row.addRow('MSER Min area', self.mser_min_area)

        self.mser_min_margin = QtGui.QSpinBox()
        self.mser_min_margin.setMinimum(3)
        self.mser_min_margin.setMaximum(100)
        self.mser_min_margin.setValue(S_.mser.min_margin)
        self.mser_min_margin.valueChanged.connect(self.val_changed)
        self.bottom_row.addRow('MSER Min margin', self.mser_min_margin)

        self.mser_img_subsample = QtGui.QDoubleSpinBox()
        self.mser_img_subsample.setMinimum(1.0)
        self.mser_img_subsample.setMaximum(12.0)
        self.mser_img_subsample.setSingleStep(0.1)
        self.mser_img_subsample.setValue(S_.mser.img_subsample_factor)
        self.mser_img_subsample.valueChanged.connect(self.val_changed)
        self.bottom_row.addRow('MSER image subsample factor', self.mser_img_subsample)

        self.blur_kernel_size = QtGui.QDoubleSpinBox()
        self.blur_kernel_size.setMinimum(0.0)
        self.blur_kernel_size.setMaximum(5.0)
        self.blur_kernel_size.setSingleStep(0.1)
        self.blur_kernel_size.setValue(S_.mser.gaussian_kernel_std)
        self.blur_kernel_size.valueChanged.connect(self.val_changed)
        self.bottom_row.addRow('Gblur kernel size', self.blur_kernel_size)

        self.update()
        self.show()

    def update(self):
        img_ = self.im.copy()

        start = time.time()
        img_ = prepare_for_segmentation(img_, self.project, grayscale_speedup=False)

        # if S_.mser.gaussian_kernel_std > 0:
        #     img_ = scipy.ndimage.gaussian_filter(img_, sigma=S_.mser.gaussian_kernel_std)
        #
        # if S_.mser.img_subsample_factor > 1.0:
        #     img_ = np.asarray(rescale(img_, 1/S_.mser.img_subsample_factor) * 255, dtype=np.uint8)

        m = get_msers_(img_)
        groups = get_region_groups(m)
        ids = margin_filter(m, groups)
        # TODO:
        # min_area = self.project.stats.area_median * 0.2
        min_area = 30
        ids = area_filter(m, ids, min_area)
        ids = children_filter(m, ids)

        self.img_grid.setParent(None)
        self.img_grid = ImgGridWidget()
        self.img_grid.element_width = 100
        self.img_grid.cols = 5
        self.top_row.addWidget(self.img_grid)

        for id in ids:
            r = m[id]

            if self.project.stats:
                prob = self.project.stats.antlikeness_svm.get_prob(r)
                if prob[1] < S_.solver.antlikeness_threshold * 0.5:
                    continue

            cont = get_contour(r.pts())
            crop = draw_points_crop(img_, cont, (0, 255, 0, 0.9), square=True)
            draw_points(img_, cont, (0, 255, 0, 0.9))

            img_q = ImageQt.QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, 13)
            pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

            item = SelectableQLabel(id=id)

            item.setScaledContents(True)
            item.setFixedSize(100, 100)
            item.setPixmap(pix_map)

            self.img_grid.add_item(item)

        self.img_preview.setParent(None)
        self.img_preview = get_image_label(img_)
        self.top_row.insertWidget(0, self.img_preview)

    def val_changed(self):
        S_.mser.img_subsample_factor = self.mser_img_subsample.value()
        S_.mser.min_area = self.mser_min_area.value()
        S_.mser.max_area = self.mser_max_area.value()
        S_.mser.min_margin = self.mser_min_margin.value()
        S_.mser.gaussian_kernel_std = self.blur_kernel_size.value()

        self.update()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    proj = Project()
    proj.load('/Users/flipajs/Documents/wd/eight/eight.fproj')
    # proj.video_paths = '/home/flipajs/Downloads/Camera 1_biglense1.avi'
    # proj.video_paths = '/media/flipajs/Seagate Expansion Drive/TestSet/cuts/c6.avi'
    # proj.video_paths = '/media/flipajs/Seagate Expansion Drive/TestSet/cuts/c1.avi'
    # proj.video_paths = '/media/flipajs/Seagate Expansion Drive/TestSet/cuts/c2.avi'

    ex = SetMSERs(proj)
    ex.raise_()
    ex.showMaximized()
    ex.activateWindow()

    app.exec_()
    app.deleteLater()
    sys.exit()
