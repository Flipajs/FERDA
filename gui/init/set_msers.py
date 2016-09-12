import sys

__author__ = 'filip@naiser.cz'

from PyQt4 import QtGui, QtCore
from gui.img_grid.img_grid_widget import ImgGridWidget
from utils.video_manager import get_auto_video_manager
from core.project.project import Project
from core.region.mser import ferda_filtered_msers
from utils.drawing.points import draw_points_crop, get_contour, draw_points_binary
from gui.segmentation.painter import Painter, rgba2qimage, array2qimage
from helper import Helper
from PIL import ImageQt
from gui.gui_utils import SelectableQLabel
from utils.img import prepare_for_segmentation
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2


class SetMSERs(QtGui.QWidget):
    def __init__(self, project, mser_color=(255, 128, 0, 200), prob_color=(0, 255, 0, 200), foreground_color=(0, 255, 0, 255), background_color=(255, 0, 238, 255)):
        super(SetMSERs, self).__init__()

        self.project = project
        self.vid = get_auto_video_manager(project)
        self.im = self.vid.next_frame()
        self.w, self.h, c = self.im.shape
        # self.im = self.vid.seek_frame(659)

        im = self.im
        if self.project.bg_model:
            im = self.project.bg_model.bg_subtraction(im)

        if self.project.arena_model:
            im = self.project.arena_model.mask_image(im)

        self.im = im
        self.use_segmentation_ = False
        self.segmentation = None

        self.color_mser = mser_color
        self.color_prob = prob_color
        self.color_foreground = foreground_color
        self.color_background = background_color

        self.pen_size = 5

        self.painter = Painter(self.im)
        self.helper = Helper(self.im)
        self.painter.add_color_("GREEN", self.color_foreground)
        self.img_grid = None

        self.setLayout(QtGui.QHBoxLayout())

        # Left panel with options and paint tools
        self.left_panel = QtGui.QWidget()
        self.left_panel.setLayout(QtGui.QVBoxLayout())
        self.left_panel.setMaximumWidth(300)
        self.left_panel.setMinimumWidth(300)

        self.form_panel = QtGui.QFormLayout()
        self.left_panel.layout().addLayout(self.form_panel)

        # Right panel with image grid
        self.right_panel = QtGui.QWidget()
        self.right_panel.setLayout(QtGui.QVBoxLayout())

        self.prepare_widgets()
        self.configure_form_panel()
        self.configure_paint_panel()

        # Complete the gui
        self.layout().addWidget(self.left_panel)
        self.layout().addWidget(self.painter)
        self.layout().addWidget(self.right_panel)

        self.painter.update_callback = self.paint_changed

        self.update()
        self.show()

    def choose_random_frame(self):
        if self.frame_number.value() == -1:
            im = self.vid.random_frame()
        else:
            im = self.vid.get_frame(self.frame_number.value())

        if self.project.bg_model:
            im = self.project.bg_model.bg_subtraction(im)

        if self.project.arena_model:
            im = self.project.arena_model.mask_image(im)

        self.im = im

        self.update()

    def update(self):
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        if self.use_segmentation_ and not self.segmentation is None:
            img_ = np.asarray((-self.segmentation*255)+255, dtype=np.uint8)
        else:
            img_ = prepare_for_segmentation(self.im.copy(), self.project, grayscale_speedup=True)

        s = time.time()
        msers = ferda_filtered_msers(img_, self.project)
        print "mser takes: ", time.time() - s
        binary = np.zeros((self.im.shape[0], self.im.shape[1]))

        if self.img_grid:
            self.img_grid.setParent(None)
        self.img_grid = ImgGridWidget(cols=3, element_width=100)
        self.right_panel.layout().addWidget(self.img_grid)

        self.fill_new_grid(msers, self.im.copy(), binary)
        im = np.asarray(binary[..., None]*self.color_mser, dtype=np.uint8)
        qim = array2qimage(im)

        self.painter.set_overlay2(qim)
        self.painter.set_overlay2_visible(self.check_mser.isChecked())
        QtGui.QApplication.restoreOverrideCursor()

    def val_changed(self):
        self.project.other_parameters.img_subsample_factor = self.mser_img_subsample.value()
        self.project.mser_parameters.min_area = self.mser_min_area.value()
        self.project.mser_parameters.max_area = self.mser_max_area.value()
        self.project.mser_parameters.min_margin = self.mser_min_margin.value()
        self.project.mser_parameters.gaussian_kernel_std = self.blur_kernel_size.value()
        self.project.other_parameters.use_only_red_channel = self.use_only_red_ch.isChecked()
        self.use_segmentation_ = self.use_segmentation.isChecked()
        self.project.mser_parameters.intensity_threshold = self.intensity_threshold.value()
        self.project.mser_parameters.min_area_relative = self.min_area_relative.value()
        self.project.mser_parameters.region_min_intensity = self.region_min_intensity.value()
        self.project.mser_parameters.use_children_filter = self.use_children_filter.isChecked()

        self.update()

    def paint_changed(self):
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        result = self.painter.get_result()
        background = result["PINK"]
        foreground = result["GREEN"]
        self.segmentation = self.helper.done(background, foreground)
        if not self.segmentation is None:
            im = np.asarray(self.segmentation[..., None]*self.color_prob, dtype=np.uint8)
            qim = array2qimage(im)
            self.painter.set_overlay(qim)
        else:
            self.painter.set_overlay(None)
        QtGui.QApplication.restoreOverrideCursor()
        self.update()

    def fill_new_grid(self, msers, img_vis, binary):
        for r, r_id in zip(msers, range(len(msers))):
            if self.project.stats:
                prob = self.project.stats.antlikeness_svm.get_prob(r)
                if prob[1] < self.project.solver_parameters.antlikeness_threshold:
                    continue

            cont = get_contour(r.pts())
            crop = draw_points_crop(img_vis, cont, (0, 255, 0, 0.9), square=True)
            binary = draw_points_binary(binary, cont)
            img_q = ImageQt.QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, 13)
            pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

            item = SelectableQLabel(id=r_id)

            item.setScaledContents(True)
            item.setFixedSize(100, 100)
            item.setPixmap(pix_map)

            self.img_grid.add_item(item)

    def pink(self):
        self.cur_color = "PINK"
        self.cur_eraser = False
        self.set_color()

    def green(self):
        self.cur_color = "GREEN"
        self.cur_eraser = False
        self.set_color()

    def set_color(self):
        if self.cur_eraser:
            self.painter.set_pen_color(None)
            self.color_buttons["eraser"].setChecked(True)
        else:
            self.painter.set_pen_color(self.cur_color)
            for color, btn in self.color_buttons.iteritems():
                if color == self.cur_color.lower():
                    btn.setChecked(True)
                else:
                    btn.setChecked(False)

    def set_eraser(self):
        if self.cur_eraser:
            self.cur_eraser = False
        else:
            self.cur_eraser = True
        self.set_color()

    def checkbox(self):
        self.painter.set_image_visible(self.check_bg.isChecked())
        self.painter.set_overlay_visible(self.check_prob.isChecked())
        self.painter.set_masks_visible(self.check_paint.isChecked())
        self.painter.set_overlay2_visible(self.check_mser.isChecked())

    def prepare_widgets(self):
        self.use_children_filter = QtGui.QCheckBox()
        self.button_group = QtGui.QButtonGroup()
        self.use_only_red_ch = QtGui.QCheckBox()
        self.use_full_image = QtGui.QCheckBox()
        self.use_segmentation = QtGui.QCheckBox()
        self.mser_max_area = QtGui.QDoubleSpinBox()
        self.mser_min_area = QtGui.QSpinBox()
        self.mser_min_margin = QtGui.QSpinBox()
        self.mser_img_subsample = QtGui.QDoubleSpinBox()
        self.blur_kernel_size = QtGui.QDoubleSpinBox()
        self.intensity_threshold = QtGui.QSpinBox()
        self.min_area_relative = QtGui.QDoubleSpinBox()
        self.region_min_intensity = QtGui.QSpinBox()
        self.check_bg = QtGui.QCheckBox("Background image")
        self.check_prob = QtGui.QCheckBox("Probability mask")
        self.check_paint = QtGui.QCheckBox("Paint data")
        self.check_mser = QtGui.QCheckBox("MSER view")

    def configure_form_panel(self):
        self.mser_max_area.setMinimum(0.0001)
        self.mser_max_area.setSingleStep(0.0001)
        self.mser_max_area.setMaximum(1.0)
        self.mser_max_area.setDecimals(6)
        self.mser_max_area.setValue(self.project.mser_parameters.max_area)
        self.mser_max_area.valueChanged.connect(self.val_changed)
        self.form_panel.addRow('MSER Max relative area', self.mser_max_area)

        self.mser_min_area.setMinimum(0)
        self.mser_min_area.setMaximum(1000)
        self.mser_min_area.setValue(self.project.mser_parameters.min_area)
        self.mser_min_area.valueChanged.connect(self.val_changed)
        self.form_panel.addRow('MSER Min area', self.mser_min_area)

        self.mser_min_margin.setMinimum(3)
        self.mser_min_margin.setMaximum(100)
        self.mser_min_margin.setValue(self.project.mser_parameters.min_margin)
        self.mser_min_margin.valueChanged.connect(self.val_changed)
        self.form_panel.addRow('MSER Min margin', self.mser_min_margin)

        self.mser_img_subsample.setMinimum(1.0)
        self.mser_img_subsample.setMaximum(12.0)
        self.mser_img_subsample.setSingleStep(0.1)
        self.mser_img_subsample.setValue(self.project.other_parameters.img_subsample_factor)
        self.mser_img_subsample.valueChanged.connect(self.val_changed)
        self.form_panel.addRow('MSER image subsample factor', self.mser_img_subsample)

        self.blur_kernel_size.setMinimum(0.0)
        self.blur_kernel_size.setMaximum(5.0)
        self.blur_kernel_size.setSingleStep(0.1)
        self.blur_kernel_size.setValue(self.project.mser_parameters.gaussian_kernel_std)
        self.blur_kernel_size.valueChanged.connect(self.val_changed)
        self.form_panel.addRow('Gblur kernel size', self.blur_kernel_size)

        self.intensity_threshold.setMinimum(0)
        self.intensity_threshold.setMaximum(256)
        self.intensity_threshold.setSingleStep(1)
        self.intensity_threshold.setValue(256)
        self.intensity_threshold.valueChanged.connect(self.val_changed)
        self.form_panel.addRow('intensity threshold (ignore pixels above)', self.intensity_threshold)

        self.min_area_relative.setMinimum(0.0)
        self.min_area_relative.setMaximum(1.0)
        self.min_area_relative.setValue(0.2)
        self.min_area_relative.setSingleStep(0.02)
        self.min_area_relative.valueChanged.connect(self.val_changed)
        self.form_panel.addRow('min_area = (median of selected regions) * ', self.min_area_relative)

        self.region_min_intensity.setMaximum(256)
        self.region_min_intensity.setValue(56)
        self.region_min_intensity.setMinimum(0)
        self.region_min_intensity.setSingleStep(1)
        self.region_min_intensity.valueChanged.connect(self.val_changed)
        self.form_panel.addRow('region min intensity', self.region_min_intensity)
        # this line is necessary to avoid possible bugs in the future
        self.project.mser_parameters.region_min_intensity = self.region_min_intensity.value()

        """
        self.frame_number = QtGui.QSpinBox()
        self.frame_number.setMinimum(-1)
        self.frame_number.setValue(-1)
        self.frame_number.setMaximum(10000000)
        self.form_panel.addRow('frame (-1 = random)', self.frame_number)
        self.random_frame = QtGui.QPushButton('go 2 frame')
        self.random_frame.clicked.connect(self.choose_random_frame)
        self.form_panel.addRow('', self.random_frame)
        """

        self.use_children_filter.stateChanged.connect(self.val_changed)
        self.use_children_filter.setChecked(self.project.mser_parameters.use_children_filter)
        self.form_panel.addRow('use children filter', self.use_children_filter)

        self.use_only_red_ch.stateChanged.connect(self.val_changed)
        self.form_panel.addRow('use only red channel in img', self.use_only_red_ch)
        self.button_group.addButton(self.use_only_red_ch)

        self.use_full_image.stateChanged.connect(self.val_changed)
        self.form_panel.addRow('full image', self.use_full_image)
        self.button_group.addButton(self.use_full_image)

        self.use_segmentation.stateChanged.connect(self.val_changed)
        self.form_panel.addRow('segmentation', self.use_segmentation)
        self.button_group.addButton(self.use_segmentation)
        self.use_segmentation.setChecked(True)

    def configure_paint_panel(self):

        # PEN SIZE slider
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.setGeometry(30, 40, 50, 30)
        self.slider.setRange(2, 30)
        self.slider.setTickInterval(1)
        self.slider.setValue(self.pen_size)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider.valueChanged[int].connect(self.painter.set_pen_size)
        self.slider.setVisible(True)
        self.left_panel.layout().addWidget(self.slider)

        # color switcher widget
        color_widget = QtGui.QWidget()
        color_widget.setLayout(QtGui.QHBoxLayout())

        self.color_buttons = {}
        pink_button = QtGui.QPushButton("Pink")
        pink_button.setCheckable(True)
        pink_button.setChecked(True)
        pink_button.clicked.connect(self.pink)
        color_widget.layout().addWidget(pink_button)
        self.color_buttons["pink"] = pink_button

        green_button = QtGui.QPushButton("Green")
        green_button.setCheckable(True)
        green_button.clicked.connect(self.green)
        color_widget.layout().addWidget(green_button)
        self.color_buttons["green"] = green_button

        eraser_button = QtGui.QPushButton("Eraser")
        eraser_button.setCheckable(True)
        eraser_button.clicked.connect(self.set_eraser)
        color_widget.layout().addWidget(eraser_button)
        self.left_panel.layout().addWidget(color_widget)
        self.color_buttons["eraser"] = eraser_button

        self.check_bg.setChecked(True)
        self.check_bg.toggled.connect(self.checkbox)
        self.left_panel.layout().addWidget(self.check_bg)

        self.check_prob.setChecked(True)
        self.check_prob.toggled.connect(self.checkbox)
        self.left_panel.layout().addWidget(self.check_prob)

        self.check_paint.setChecked(True)
        self.check_paint.toggled.connect(self.checkbox)
        self.left_panel.layout().addWidget(self.check_paint)

        self.check_mser.setChecked(True)
        self.check_mser.toggled.connect(self.checkbox)
        self.left_panel.layout().addWidget(self.check_mser)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    proj = Project()

    # proj.load("/home/dita/Programovani/FERDA Projects/cam1_test/cam1_test.fproj")
    # proj.load('/Users/flipajs/Documents/wd/GT/C210_5000/C210.fproj')
    proj.load('/Users/flipajs/Documents/wd/GT/Cam1 copy/cam1.fproj')
    # proj.video_paths = ['/Users/flipajs/Documents/wd/GT/C210_5000/C210.fproj']
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

    ex.mser_min_margin.setValue(proj.mser_parameters.min_margin)
    ex.mser_min_area.setValue(proj.mser_parameters.min_area)
    ex.mser_max_area.setValue(proj.mser_parameters.max_area)

    app.exec_()
    app.deleteLater()
    sys.exit()
