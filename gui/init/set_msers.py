import sys

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

__author__ = 'filip@naiser.cz', 'dita'


class SetMSERs(QtGui.QWidget):
    def __init__(self, project, mser_color=(255, 128, 0, 200), prob_color=(0, 255, 0, 200),
                 foreground_color=(0, 255, 0, 255), background_color=(255, 0, 238, 255)):
        """
        Interactive tool to improve msers search using segmentation.
        :param project: Ferda project
        :param mser_color: 0-255 rgba color to visualise mser borders (orange by default)
        :param prob_color: 0-255 rgba color to visualise segmentation probability mask (green by default)
        :param foreground_color: 0-255 rgba color to visualise foreground when painting (pink by default)
        :param background_color: 0-255 rgba color to visualise background when painting (green by default)
        """
        super(SetMSERs, self).__init__()

        self.project = project
        self.vid = get_auto_video_manager(project)

        self.im = self.vid.next_frame()
        im = self.im
        if self.project.bg_model:
            im = self.project.bg_model.bg_subtraction(im)

        if self.project.arena_model:
            im = self.project.arena_model.mask_image(im)
        self.im = im
        self.w, self.h, c = self.im.shape

        self.use_segmentation_ = False
        self.segmentation = None

        # Setup colors
        self.color_mser = mser_color
        self.color_prob = prob_color
        self.color_foreground = foreground_color
        self.color_background = background_color

        # Setup painting tools
        self.pen_size = 5
        self.cur_color = "background"
        self.cur_eraser = False

        # Setup painter
        r, g, b, a = self.color_background
        self.painter = Painter(self.im, paint_name="background", paint_r=r, paint_g=g, paint_b=b, paint_a=a)
        self.painter.add_color_("foreground", self.color_foreground)

        # Setup segmentation helper
        self.helper = Helper(self.im)

        # Prepare img_grid variable
        self.img_grid = None

        self.setLayout(QtGui.QHBoxLayout())

        # Left panel with options and paint tools
        self.left_panel = QtGui.QWidget()
        self.left_panel.setLayout(QtGui.QVBoxLayout())

        # Left panel must be scrollable on smaller screens
        left_scroll = QtGui.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(self.left_panel)
        left_scroll.setMaximumWidth(300)
        left_scroll.setMinimumWidth(300)

        self.form_panel = QtGui.QFormLayout()
        self.left_panel.layout().addLayout(self.form_panel)

        # Right panel with image grid
        self.right_panel = QtGui.QWidget()
        self.right_panel.setLayout(QtGui.QVBoxLayout())

        # Setup gui elements
        self.prepare_widgets()
        self.configure_form_panel()
        self.configure_paint_panel()

        # Complete the gui
        self.layout().addWidget(left_scroll)  # self.layout().addWidget(self.left_panel)
        self.layout().addWidget(self.painter)
        self.layout().addWidget(self.right_panel)

        # Set a callback in painter when paint event occurs
        self.painter.update_callback = self.update_all

        self.update_all()
        self.show()

    def update_all(self):
        """
        Computes new probability map and msers, also updates GUI. Can be called after parameter/image change as a short
        for calling update_paint() and update_mser()
        :return: None
        """
        # paint must be updated first, because segmentation results are used in msers
        self.update_paint()
        self.update_mser()

    def update_mser(self):
        """
        Finds new MSERS and updates all related gui elements (grid and painter mser overlay). This must be called
        every time a parameter or source image is changed.
        :return: None
        """
        # start loading cursor animation
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))

        # get working image
        # TODO: update project settings to contain "use_segmentation_image"
        if self.use_segmentation_ and self.segmentation is not None:
            img_ = np.asarray((-self.segmentation*255)+255, dtype=np.uint8)
        else:
            img_ = prepare_for_segmentation(self.im.copy(), self.project, grayscale_speedup=True)

        # get msers
        s = time.time()
        msers = ferda_filtered_msers(img_, self.project)
        print "mser takes: ", time.time() - s

        # prepare empty array - mser borders will be painted there and it will be visualised as painter's overlay
        mser_vis = np.zeros((self.im.shape[0], self.im.shape[1]))

        # delete old image grid if it exists and prepare a new one
        if self.img_grid:
            self.img_grid.setParent(None)
        self.img_grid = ImgGridWidget(cols=3, element_width=100)
        self.right_panel.layout().addWidget(self.img_grid)

        # fill new grid with msers visualisations on current image, this also fills `mser_vis`
        self.fill_new_grid(msers, self.im.copy(), mser_vis)

        # convert `mser_vis` to 4 channel image and show it as overlay
        im = np.asarray(mser_vis[..., None]*self.color_mser, dtype=np.uint8)
        qim = array2qimage(im)
        self.painter.set_overlay2(qim)
        self.painter.set_overlay2_visible(self.check_mser.isChecked())

        # restore cursor look to default
        QtGui.QApplication.restoreOverrideCursor()

    def update_paint(self):
        """
        Computes new probability map and updates view in painter. This must be called every time the source image
        or any of paint masks is changed.
        :return: None
        """

        # start cursor animation
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))

        # get painted data masks from painter
        result = self.painter.get_result()
        background = result["background"]
        foreground = result["foreground"]

        # obtain segmentation image from helper
        self.segmentation = self.helper.done(background, foreground)

        # show result as overlay
        if self.segmentation is not None:
            im = np.asarray(self.segmentation[..., None]*self.color_prob, dtype=np.uint8)
            qim = array2qimage(im)
            self.painter.set_overlay(qim)
        else:  # or hide it if input data was insufficient to create a result
            self.painter.set_overlay(None)

        # stop cursor animation
        QtGui.QApplication.restoreOverrideCursor()

    def fill_new_grid(self, msers, img_vis, mser_vis):
        """
        Loop through all regions. Add their individual cropped images to image grid and draw them in a binary array.
        :param msers: MSERs
        :param img_vis: RGBA background image to make crops from
        :param mser_vis: one channel image, which will be modified to contain mser contours
        :return: None
        """
        for r, r_id in zip(msers, range(len(msers))):
            if self.project.stats:
                prob = self.project.stats.antlikeness_svm.get_prob(r)
                if prob[1] < self.project.solver_parameters.antlikeness_threshold:
                    continue

            # get region contours
            cont = get_contour(r.pts())

            # visualise it on a small cropped image
            # format color to be BGRA, convert alpha channel from 0-255 to 0.0-1.0 scale
            crop = draw_points_crop(img_vis, cont, (self.color_mser[2], self.color_mser[1], self.color_mser[0],
                                                    self.color_mser[3]/float(255)), square=True)
            # create qimage from crop
            img_q = ImageQt.QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, 13)
            pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

            # add crop to img grid
            item = SelectableQLabel(id=r_id)
            item.setScaledContents(True)
            item.setFixedSize(100, 100)
            item.setPixmap(pix_map)
            self.img_grid.add_item(item)

            # visualise it in the binary image
            mser_vis = draw_points_binary(mser_vis, cont)

    def show_next_frame(self):
        """
        Show current settings on next frame
        :return: None
        """
        image = self.vid.next_frame()
        self.set_image(image)

    def show_random_frame(self):
        """
        Show current settings on random frame
        :return: None
        """
        if self.frame_number.value() == -1:
            im = self.vid.random_frame()
        else:
            im = self.vid.get_frame(self.frame_number.value())

        if self.project.bg_model:
            im = self.project.bg_model.bg_subtraction(im)

        if self.project.arena_model:
            im = self.project.arena_model.mask_image(im)

        self.set_image(im)

    def set_image(self, image):
        """
        Change current display image and adjust everything.
        :param image: new image to display
        :return: None
        """
        # update self image
        self.im = image
        # set new background image in painter
        self.painter.set_image(self.im)
        # delete old paint marks from painter and reset mask data
        self.painter.reset_masks()
        # delete all undo's, as they are from old frames and will never be needed again
        self.painter.backup = []
        # save current xy data in helper
        self.helper.update_xy()
        # update helper's image
        self.helper.set_image(self.im)
        self.update_all()

    def set_color_bg(self):
        self.cur_color = "background"
        self.cur_eraser = False
        self.set_color()

    def set_color_fg(self):
        self.cur_color = "foreground"
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

        # only mser-related parameters were changed, no need to update everything
        self.update_mser()

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
        self.button_next = QtGui.QPushButton("Next frame")

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
        button_bg = QtGui.QPushButton("background")
        button_bg.setCheckable(True)
        button_bg.setChecked(True)
        button_bg.clicked.connect(self.set_color_bg)
        color_widget.layout().addWidget(button_bg)
        self.color_buttons["background"] = button_bg

        button_fg = QtGui.QPushButton("foreground")
        button_fg.setCheckable(True)
        button_fg.clicked.connect(self.set_color_fg)
        color_widget.layout().addWidget(button_fg)
        self.color_buttons["foreground"] = button_fg

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

        self.button_next.clicked.connect(self.show_next_frame)
        self.left_panel.layout().addWidget(self.button_next)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    proj = Project()

    proj.load("/home/dita/Programovani/FERDA Projects/cam1_test/cam1_test.fproj")
    # proj.load('/Users/flipajs/Documents/wd/GT/C210_5000/C210.fproj')
    # proj.video_paths = ['/Users/flipajs/Documents/wd/GT/C210_5000/C210.fproj']
    proj.arena_model = None
    proj.bg_model = None

    # proj.video_paths = '/home/flipajs/Downloads/Camera 1_biglense1.avi'
    # proj.video_paths = '/media/flipajs/Seagate Expansion Drive/TestSet/cuts/c6.avi'
    # proj.video_paths = '/media/flipajs/Seagate Expansion Drive/TestSet/cuts/c1.avi'
    # proj.video_paths = '/media/flipajs/Seagate Expansion Drive/TestSet/cuts/c2.avi'

    print "Done loading"

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
