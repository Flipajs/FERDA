import sys

__author__ = 'filip@naiser.cz'

from PyQt4 import QtGui
from gui.img_grid.img_grid_widget import ImgGridWidget
from gui.gui_utils import get_image_label
from utils.video_manager import get_auto_video_manager
from core.project.project import Project
from core.region.mser import get_msers_
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
from utils.drawing.points import draw_points_crop, draw_points, get_contour
from PIL import ImageQt
from gui.gui_utils import SelectableQLabel
from core.settings import Settings as S_
from utils.img import prepare_for_segmentation
import time
import numpy as np


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

        self.vid = get_auto_video_manager(project)
        self.im = self.vid.next_frame()
        # self.im = self.vid.seek_frame(659)

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
        self.mser_max_area.setValue(project.mser_parameters.max_area)
        self.mser_max_area.valueChanged.connect(self.val_changed)
        self.bottom_row.addRow('MSER Max relative area', self.mser_max_area)

        self.mser_min_area = QtGui.QSpinBox()
        self.mser_min_area.setMinimum(0)
        self.mser_min_area.setMaximum(1000)
        self.mser_min_area.setValue(project.mser_parameters.min_area)
        self.mser_min_area.valueChanged.connect(self.val_changed)
        self.bottom_row.addRow('MSER Min area', self.mser_min_area)

        self.mser_min_margin = QtGui.QSpinBox()
        self.mser_min_margin.setMinimum(3)
        self.mser_min_margin.setMaximum(100)
        self.mser_min_margin.setValue(project.mser_parameters.min_margin)
        self.mser_min_margin.valueChanged.connect(self.val_changed)
        self.bottom_row.addRow('MSER Min margin', self.mser_min_margin)

        self.mser_img_subsample = QtGui.QDoubleSpinBox()
        self.mser_img_subsample.setMinimum(1.0)
        self.mser_img_subsample.setMaximum(12.0)
        self.mser_img_subsample.setSingleStep(0.1)
        self.mser_img_subsample.setValue(project.other_parameters.img_subsample_factor)
        self.mser_img_subsample.valueChanged.connect(self.val_changed)
        self.bottom_row.addRow('MSER image subsample factor', self.mser_img_subsample)

        self.blur_kernel_size = QtGui.QDoubleSpinBox()
        self.blur_kernel_size.setMinimum(0.0)
        self.blur_kernel_size.setMaximum(5.0)
        self.blur_kernel_size.setSingleStep(0.1)
        self.blur_kernel_size.setValue(project.mser_parameters.gaussian_kernel_std)
        self.blur_kernel_size.valueChanged.connect(self.val_changed)
        self.bottom_row.addRow('Gblur kernel size', self.blur_kernel_size)

        self.use_only_red_ch = QtGui.QCheckBox()
        self.use_only_red_ch.stateChanged.connect(self.val_changed)
        self.bottom_row.addRow('use only red channel in img', self.use_only_red_ch)

        self.intensity_threshold = QtGui.QSpinBox()
        self.intensity_threshold.setMinimum(0)
        self.intensity_threshold.setMaximum(256)
        self.intensity_threshold.setSingleStep(1)
        self.intensity_threshold.setValue(256)
        self.intensity_threshold.valueChanged.connect(self.val_changed)
        self.bottom_row.addRow('intensity threshold (ignore pixels above)', self.intensity_threshold)

        self.min_area_relative = QtGui.QDoubleSpinBox()
        self.min_area_relative.setMinimum(0.0)
        self.min_area_relative.setMaximum(1.0)
        self.min_area_relative.setValue(0.2)
        self.min_area_relative.setSingleStep(0.02)
        self.min_area_relative.valueChanged.connect(self.val_changed)
        self.bottom_row.addRow('min_area = (median of selected regions) * ', self.min_area_relative)

        self.random_frame = QtGui.QPushButton('random frame')
        self.random_frame.clicked.connect(self.choose_random_frame)
        self.bottom_row.addRow('', self.random_frame)

        self.update()
        self.show()

    def choose_random_frame(self):
        im = self.vid.random_frame()

        if self.project.bg_model:
            im = self.project.bg_model.bg_subtraction(im)

        if self.project.arena_model:
            im = self.project.arena_model.mask_image(im)

        self.im = im

        self.update()

    def update(self):
        img_ = self.im.copy()

        img_ = prepare_for_segmentation(img_, self.project, grayscale_speedup=True)

        img_vis = np.zeros((img_.shape[0], img_.shape[1], 3), dtype=np.uint8)
        img_vis[:,:,0] = img_
        img_vis[:,:,1] = img_
        img_vis[:,:,2] = img_

        import time
        s = time.time()
        m = get_msers_(img_, self.project)
        print "mser takes: ", time.time() - s

        groups = get_region_groups(m)
        ids = margin_filter(m, groups)
        # TODO:
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
                if prob[1] < self.project.solver_parameters.antlikeness_threshold * 0.5:
                    continue

            cont = get_contour(r.pts())
            crop = draw_points_crop(img_vis, cont, (0, 255, 0, 0.9), square=True)
            draw_points(img_vis, cont, (0, 255, 0, 0.9))

            img_q = ImageQt.QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, 13)
            pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

            item = SelectableQLabel(id=id)

            item.setScaledContents(True)
            item.setFixedSize(100, 100)
            item.setPixmap(pix_map)

            self.img_grid.add_item(item)

        self.img_preview.setParent(None)
        self.img_preview = get_image_label(img_vis)
        self.top_row.insertWidget(0, self.img_preview)

    def val_changed(self):
        self.project.other_parameters.img_subsample_factor = self.mser_img_subsample.value()
        self.project.mser_parameters.min_area = self.mser_min_area.value()
        self.project.mser_parameters.max_area = self.mser_max_area.value()
        self.project.mser_parameters.min_margin = self.mser_min_margin.value()
        self.project.mser_parameters.gaussian_kernel_std = self.blur_kernel_size.value()
        self.project.other_parameters.use_only_red_channel = self.use_only_red_ch.isChecked()
        self.project.mser_parameters.intensity_threshold = self.intensity_threshold.value()
        self.project.mser_parameters.min_area_relative = self.min_area_relative.value()

        self.update()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    proj = Project()

    proj.load('/Users/flipajs/Documents/wd/eight/eight.fproj')
    proj.arena_model = None
    proj.bg_model = None
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
