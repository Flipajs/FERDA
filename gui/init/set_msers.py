import cPickle as pickle
import sys
import time

import numpy as np
from PIL import ImageQt
from PyQt4 import QtGui, QtCore
import cv2

from core.segmentation_helper import SegmentationHelper
from core.project.project import Project
from core.region.mser import get_filtered_msers
from gui.gui_utils import SelectableQLabel
from gui.img_grid.img_grid_widget import ImgGridWidget
from gui.segmentation.painter import Painter
from gui.segmentation.painter import array2qimage
from utils.drawing.points import draw_points_crop, get_contour, draw_points_binary
from utils.img import prepare_for_segmentation
from utils.video_manager import get_auto_video_manager
from core.region.mser import get_filtered_msers

__author__ = 'filip@naiser.cz', 'dita'


class SetMSERs(QtGui.QWidget):
    def __init__(self, project, finish_callback=None, mser_color=(255, 128, 0, 200), prob_color=(0, 255, 0, 200),
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

        self.grid_element_width = 100

        self.project = project
        self.video = get_auto_video_manager(project)

        self.im = self.video.next_frame()
        im = self.im
        if self.project.bg_model:
            im = self.project.bg_model.bg_subtraction(im)

        if self.project.arena_model:
            im = self.project.arena_model.mask_image(im)
        self.im = im
        self.w, self.h, c = self.im.shape

        self.use_segmentation_ = True
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
        self.helper = SegmentationHelper(self.im)

        # Prepare img_grid variable
        self.img_grid = None

        # max area helper
        self.old_max_area_helper = None
        self.old_max_area_helper_text = None

        self.setLayout(QtGui.QVBoxLayout())  # all + continue button
        self.left_panel_and_preview = QtGui.QWidget()
        self.left_panel_and_preview.setLayout(QtGui.QHBoxLayout())

        # Left panel with options and paint tools
        self.left_panel = QtGui.QWidget()
        self.left_panel.setLayout(QtGui.QVBoxLayout())

        # Left panel must be scrollable on smaller screens
        left_scroll = QtGui.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(self.left_panel)
        left_scroll.setMaximumWidth(450)
        left_scroll.setMinimumWidth(300)

        self.form_panel = QtGui.QFormLayout()

        self.gb_general = QtGui.QGroupBox('general')
        self.gb_general.setLayout(QtGui.QFormLayout())
        self.left_panel.layout().addWidget(self.gb_general)

        self.gb_mser_related = QtGui.QGroupBox('MSER settings')
        self.gb_mser_related.setLayout(QtGui.QFormLayout())
        self.left_panel.layout().addWidget(self.gb_mser_related)

        # TODO: describe all parameters...
        self.gb_region_filtering = QtGui.QGroupBox('region filtering')
        self.gb_region_filtering.setLayout(QtGui.QFormLayout())
        self.left_panel.layout().addWidget(self.gb_region_filtering)

        self.gb_video_controls = QtGui.QGroupBox('video controls')
        self.gb_video_controls.setLayout(QtGui.QVBoxLayout())
        self.left_panel.layout().addWidget(self.gb_video_controls)

        self.gb_pixel_classifier = QtGui.QGroupBox('pixel classifier')
        self.gb_pixel_classifier.setLayout(QtGui.QFormLayout())
        self.gb_pixel_classifier.setCheckable(True)
        self.gb_pixel_classifier.setChecked(False)
        self.left_panel.layout().addWidget(self.gb_pixel_classifier)

        self.left_panel.layout().addLayout(self.form_panel)

        self.num_animals_sb = QtGui.QSpinBox()
        self.num_animals_sb.setValue(6)
        self.num_animals_sb.setMinimum(1)
        self.num_animals_sb.setMaximum(999)
        self.num_animals_sb.setToolTip('Number of tracked objects')

        self.gb_general.layout().addRow('<b>#animals:</b> ', self.num_animals_sb)

        # Right panel with image grid
        self.right_panel = QtGui.QWidget()
        self.right_panel.setLayout(QtGui.QVBoxLayout())

        # Setup gui elements
        self.prepare_widgets()
        self.configure_form_panel()
        self.configure_paint_panel()

        # Complete the gui
        self.left_panel_and_preview.layout().addWidget(left_scroll)  # self.layout().addWidget(self.left_panel)
        self.left_panel_and_preview.layout().addWidget(self.painter)
        self.layout().addWidget(self.left_panel_and_preview)
        if finish_callback:
            self.finish_button = QtGui.QPushButton('Continue')
            self.finish_button.clicked.connect(finish_callback)
            self.layout().addWidget(self.finish_button)

        # Set a callback in painter when paint event occurs
        self.painter.update_callback = self.update_all

        self.update_all()
        self.update_frame_number()
        self.show()

    def update_all(self):
        """
        Computes new probability map and msers, also updates GUI. Can be called after parameter/image change as a short
        for calling update_paint() and update_mser()
        :return: None
        """
        # paint must be updated first, because segmentation results are used in msers

        b = self.gb_pixel_classifier.isChecked()
        self.painter.disable_drawing = not b
        self.painter.set_masks_visible(self.check_paint and b)

        self.update_paint()
        self.update_img()

        self.update_mser()

    def draw_max_area_helper(self):
        if self.old_max_area_helper is not None:
            self.painter.scene.removeItem(self.old_max_area_helper)
            self.painter.scene.removeItem(self.old_max_area_helper_text)

        import math
        from gui.img_controls import markers
        radius = math.ceil((self.project.mser_parameters.max_area/np.pi)**0.5)

        c = QtGui.QColor(167, 255, 36)
        it = markers.CenterMarker(0, 0, 2*radius, c, 0, None)
        it.setFlag(QtGui.QGraphicsItem.ItemIsMovable, False)
        it.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)
        it.setOpacity(0.2)

        text = QtGui.QGraphicsTextItem()
        text.setPos(radius/3, radius/3)
        text.setPlainText("max area helper")

        self.painter.scene.addItem(text)

        self.old_max_area_helper = it
        self.old_max_area_helper_text = text
        self.painter.scene.addItem(it)

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
        from core.region.mser import get_msers_img
        # msers = get_msers_(img_, self.project, 0, prefiltered=False)

        msers = get_filtered_msers(img_, self.project, 0)

        # print "mser takes: ", time.time() - s

        # prepare empty array - mser borders will be painted there and it will be visualised as painter's overlay
        mser_vis = np.zeros((self.im.shape[0], self.im.shape[1]))

        # delete old image grid if it exists and prepare a new one
        if self.img_grid:
            self.img_grid.setParent(None)
        self.img_grid = ImgGridWidget(cols=1, element_width=self.grid_element_width)
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

        self.draw_max_area_helper()

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
        self.helper.train(background, foreground)

        # stop cursor animation
        QtGui.QApplication.restoreOverrideCursor()

    def update_img(self):
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))

        if self.use_segmentation_:
            t = time.time()
            self.segmentation = self.helper.predict()
            # print "prediction takes: {:.4f}".format(time.time() - t)

            # show result as overlay
            if self.segmentation is not None:
                im = np.asarray(self.segmentation[..., None]*self.color_prob, dtype=np.uint8)
                qim = array2qimage(im)
                self.painter.set_overlay(qim)
            else:  # or hide it if input data was insufficient to create a result
                self.painter.set_overlay(None)
        else:
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

        from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter

        groups = get_region_groups(msers)
        ids = margin_filter(msers, groups)

        for r, r_id in zip(msers, range(len(msers))):
            # get region contours
            cont = get_contour(r.pts())

            fill_pts = None
            if r.area() > self.project.mser_parameters.min_area:
                fill_pts = r.pts()[:self.project.mser_parameters.min_area]

            # visualise it on a small cropped image
            # format color to be BGRA, convert alpha channel from 0-255 to 0.0-1.0 scale

            if r_id in ids:
                crop = draw_points_crop(img_vis, cont, (255, 0, 255,
                                                        self.color_mser[3]/float(255)), square=True, fill_pts=fill_pts)
            else:
                crop = draw_points_crop(img_vis, cont, (self.color_mser[2], self.color_mser[1], self.color_mser[0],
                                                        self.color_mser[3] / float(255)), square=True,
                                        fill_pts=fill_pts)

            if crop.shape[1] < 100:
                temp = np.zeros((crop.shape[0], 100, 3), dtype=np.uint8)
                temp[:, :crop.shape[1], :] = crop

                crop = temp

            import cv2

            cv2.putText(crop, str(r.min_intensity_)+' '+str(r.area())+' '+str(r.label_), (10, 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (255, 255, 255))

            # create qimage from crop_
            img_q = ImageQt.QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, 13)
            pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

            # add crop_ to img grid
            item = SelectableQLabel(id=r_id)
            item.setScaledContents(True)
            item.setFixedSize(self.grid_element_width, self.grid_element_width)
            item.setPixmap(pix_map)

            self.img_grid.add_item(item)

            # visualise it in the binary image
            mser_vis = draw_points_binary(mser_vis, cont)

    def show_next_frame(self):
        """
        Show current settings on next frame
        :return: None
        """
        image = self.video.next_frame()
        self.update_frame_number()
        self.set_image(image)

    def show_prev_frame(self):
        """
        Show current settings on next frame
        :return: None
        """
        image = self.video.previous_frame()
        self.update_frame_number()
        self.set_image(image)

    def go_to_frame(self):
        try:
            frame = int(self.frame_input.text())
        except ValueError:
            frame = int(self.frame_input.text().split('/')[0])

        image = self.video.get_frame(frame)
        self.set_image(image)
        self.update_frame_number()

    def show_random_frame(self):
        """
        Show current settings on random frame
        :return: None
        """
        im, _ = self.video.random_frame()

        if self.project.bg_model:
            im = self.project.bg_model.bg_subtraction(im)

        if self.project.arena_model:
            im = self.project.arena_model.mask_image(im)

        self.update_frame_number()
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
        self.update_img()
        self.update_mser()
        # self.update_all()

    def done(self):
        with open(self.project.working_directory+'/segmentation_model.pkl', 'wb') as f:
            pickle.dump(self.helper, f, -1)

        self.project.segmentation_model = self.helper

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
        prev_use_s = self.use_segmentation_

        self.project.other_parameters.img_subsample_factor = self.mser_img_subsample.value()
        self.project.mser_parameters.min_area = self.mser_min_area.value()
        self.project.mser_parameters.max_area = self.mser_max_area.value()
        self.project.mser_parameters.min_margin = self.mser_min_margin.value()
        self.project.mser_parameters.use_min_margin_filter = self.use_margin_filter.isChecked()
        self.project.mser_parameters.gaussian_kernel_std = self.blur_kernel_size.value()
        self.project.other_parameters.use_only_red_channel = self.use_only_red_ch.isChecked()
        self.use_segmentation_ = self.gb_pixel_classifier.isChecked()
        self.project.mser_parameters.intensity_threshold = self.intensity_threshold.value()
        self.project.mser_parameters.region_min_intensity = self.region_min_intensity.value()
        self.project.mser_parameters.use_children_filter = self.use_children_filter.isChecked()
        self.project.mser_parameters.use_intensity_percentile_threshold = self.use_intensity_percentile_threshold.isChecked()
        self.project.mser_parameters.intensity_percentile = self.intensity_percentile.value()
        self.project.mser_parameters.area_roi_ratio_threshold = self.area_roi_ratio_threshold.value()

        if prev_use_s == self.use_segmentation_:
            # only mser-related parameters were changed, no need to update everything
            self.update_mser()
        else:
            self.update_all()

    def prepare_widgets(self):
        self.use_intensity_percentile_threshold = QtGui.QCheckBox()
        self.use_intensity_percentile_threshold.setToolTip('Instead of region filtering based on minimum intensity, the n-th percentile will be computed, which is slightly slower but more robust thus relevant.')
        self.intensity_percentile = QtGui.QSpinBox()
        self.intensity_percentile.setMinimum(1)
        self.intensity_percentile.setMaximum(100)
        self.intensity_percentile.setValue(10)

        self.area_roi_ratio_threshold = QtGui.QDoubleSpinBox()
        self.area_roi_ratio_threshold.setMinimum(0)
        self.area_roi_ratio_threshold.setMaximum(1.0)
        self.area_roi_ratio_threshold.setValue(0)

        self.max_dist_object_length = QtGui.QDoubleSpinBox()
        self.max_dist_object_length.setMinimum(0.01)
        self.max_dist_object_length.setMaximum(100)
        self.max_dist_object_length.setValue(2.0)

        self.major_axis_median = QtGui.QSpinBox()
        self.major_axis_median.setMinimum(0)
        self.major_axis_median.setMaximum(1000)
        self.major_axis_median.setValue(20)

        self.use_margin_filter = QtGui.QCheckBox()
        self.use_margin_filter.setChecked(True)

        self.use_children_filter = QtGui.QCheckBox()
        self.button_group = QtGui.QButtonGroup()
        self.use_only_red_ch = QtGui.QCheckBox()
        self.use_full_image = QtGui.QCheckBox()
        self.mser_max_area = QtGui.QSpinBox()
        self.mser_min_area = QtGui.QSpinBox()
        self.mser_min_margin = QtGui.QSpinBox()
        self.mser_img_subsample = QtGui.QDoubleSpinBox()
        self.blur_kernel_size = QtGui.QDoubleSpinBox()
        self.intensity_threshold = QtGui.QSpinBox()
        self.region_min_intensity = QtGui.QSpinBox()
        self.check_bg = QtGui.QCheckBox("Background image")
        self.check_prob = QtGui.QCheckBox("Probability mask")
        self.check_paint = QtGui.QCheckBox("Paint data")
        self.check_mser = QtGui.QCheckBox("MSER view")
        self.button_next = QtGui.QPushButton("Next frame")
        self.button_prev = QtGui.QPushButton("Previous frame")
        self.button_rand = QtGui.QPushButton("Random frame")
        from gui.init.crop_video_widget import SelectAllLineEdit
        self.frame_input = SelectAllLineEdit()
        self.show_frame_b = QtGui.QPushButton('go to frame')
        self.show_frame_b.clicked.connect(self.go_to_frame)


        self.button_refresh = QtGui.QPushButton("refresh (new randomized training)")
        self.button_reset = QtGui.QPushButton("restart (delete labels)")

        self.use_roi_prediction_optimisation_ch = QtGui.QCheckBox('')
        self.prediction_optimisation_border_spin = QtGui.QSpinBox()
        self.full_segmentation_refresh_in_spin = QtGui.QSpinBox()

        # self.button_done = QtGui.QPushButton("Done")

    def update_frame_number(self):
        self.frame_input.setText(str(int(self.video.frame_number())) + '/' + str(self.video.total_frame_count()-1))

    def configure_form_panel(self):
        self.mser_max_area.setMinimum(100)
        self.mser_max_area.setSingleStep(1)
        self.mser_max_area.setMaximum(int(1e6))
        # self.mser_max_area.setValue(self.project.mser_parameters.max_area)
        self.mser_max_area.setValue(50000)

        self.gb_mser_related.layout().addRow('MSER Max area', self.mser_max_area)

        self.mser_min_area.setMinimum(0)
        self.mser_min_area.setMaximum(100000)
        self.mser_min_area.setValue(self.project.mser_parameters.min_area)

        self.gb_mser_related.layout().addRow('MSER Min area', self.mser_min_area)

        self.mser_min_margin.setMinimum(1)
        self.mser_min_margin.setMaximum(100)
        self.mser_min_margin.setValue(self.project.mser_parameters.min_margin)

        self.gb_mser_related.layout().addRow('MSER Min margin', self.mser_min_margin)

        self.mser_img_subsample.setMinimum(1.0)
        self.mser_img_subsample.setMaximum(12.0)
        self.mser_img_subsample.setSingleStep(0.1)
        self.mser_img_subsample.setValue(self.project.other_parameters.img_subsample_factor)

        self.blur_kernel_size.setMinimum(0.0)
        self.blur_kernel_size.setMaximum(5.0)
        self.blur_kernel_size.setSingleStep(0.1)
        self.blur_kernel_size.setValue(self.project.mser_parameters.gaussian_kernel_std)

        self.intensity_threshold.setMinimum(0)
        self.intensity_threshold.setMaximum(256)
        self.intensity_threshold.setSingleStep(1)
        self.intensity_threshold.setValue(256)

        self.gb_mser_related.layout().addRow('ignore pixels >= ', self.intensity_threshold)

        self.region_min_intensity.setMaximum(256)
        self.region_min_intensity.setValue(56)
        self.region_min_intensity.setMinimum(0)
        self.region_min_intensity.setSingleStep(1)

        self.gb_mser_related.layout().addRow('suppress bright region\n(all pixels intensity above threshold)', self.region_min_intensity)
        # this line is necessary to avoid possible bugs in the future
        self.project.mser_parameters.region_min_intensity = self.region_min_intensity.value()

        self.gb_region_filtering.layout().addRow('use percentile for bright regions suppression', self.use_intensity_percentile_threshold)
        self.gb_region_filtering.layout().addRow('percentile: ', self.intensity_percentile)

        self.gb_mser_related.layout().addRow('<i>use only red channel</i>', self.use_only_red_ch)
        self.gb_mser_related.layout().addRow('<i>Gauss. blur kernel size</i>', self.blur_kernel_size)
        self.gb_mser_related.layout().addRow('<i>MSER image subsample factor</i>', self.mser_img_subsample)

        self.use_intensity_percentile_threshold.setChecked(self.project.mser_parameters.use_children_filter)
        self.use_intensity_percentile_threshold.stateChanged.connect(
            lambda x: self.intensity_percentile.setDisabled(
                not self.use_intensity_percentile_threshold.isChecked()
            )
        )

        self.gb_region_filtering.layout().addRow('use margin filter', self.use_margin_filter)

        self.use_children_filter.setChecked(self.project.mser_parameters.use_children_filter)
        self.gb_region_filtering.layout().addRow('use children filter', self.use_children_filter)

        self.gb_region_filtering.layout().addRow('(area / roi ratio) > ', self.area_roi_ratio_threshold)

        # self.form_panel.addRow('work on intensity only', self.use_full_image)
        # self.use_full_image.setChecked(True)

        self.prediction_optimisation_border_spin.setMinimum(0)
        self.prediction_optimisation_border_spin.setMaximum(10000)
        self.prediction_optimisation_border_spin.setValue(25)

        self.full_segmentation_refresh_in_spin.setMinimum(0)
        self.full_segmentation_refresh_in_spin.setMaximum(10000)
        self.full_segmentation_refresh_in_spin.setValue(25)

        self.gb_region_filtering.layout().addRow('major axis median', self.major_axis_median)
        self.gb_region_filtering.layout().addRow('max dist = this x major axis median', self.max_dist_object_length)

        # self.form_panel.addRow('max distance [px]', self.max_dist_object_length)
        self.gb_pixel_classifier.layout().addRow('use ROI prediction optimisation', self.use_roi_prediction_optimisation_ch)
        self.gb_pixel_classifier.layout().addRow('prediction ROI border', self.prediction_optimisation_border_spin)
        self.gb_pixel_classifier.layout().addRow('full segmentation every n-th frame', self.full_segmentation_refresh_in_spin)

        self.mser_max_area.valueChanged.connect(self.val_changed)
        self.mser_min_area.valueChanged.connect(self.val_changed)
        self.mser_min_margin.valueChanged.connect(self.val_changed)
        self.use_margin_filter.stateChanged.connect(self.val_changed)
        self.mser_img_subsample.valueChanged.connect(self.val_changed)
        self.blur_kernel_size.valueChanged.connect(self.val_changed)
        self.intensity_threshold.valueChanged.connect(self.val_changed)
        self.region_min_intensity.valueChanged.connect(self.val_changed)
        self.use_intensity_percentile_threshold.stateChanged.connect(self.val_changed)
        self.intensity_percentile.valueChanged.connect(self.val_changed)
        self.area_roi_ratio_threshold.valueChanged.connect(self.val_changed)
        self.use_children_filter.stateChanged.connect(self.val_changed)
        self.use_only_red_ch.stateChanged.connect(self.val_changed)
        self.use_full_image.stateChanged.connect(self.val_changed)
        self.gb_pixel_classifier.clicked.connect(self.val_changed)

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
        self.gb_pixel_classifier.layout().addRow(self.slider)

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
        self.gb_pixel_classifier.layout().addRow(color_widget)
        self.color_buttons["eraser"] = eraser_button

        self.check_bg.setChecked(True)
        self.check_bg.toggled.connect(self.checkbox)
        self.gb_pixel_classifier.layout().addRow(self.check_bg)

        self.check_prob.setChecked(True)
        self.check_prob.toggled.connect(self.checkbox)
        self.gb_pixel_classifier.layout().addRow(self.check_prob)

        self.check_paint.setChecked(True)
        self.check_paint.toggled.connect(self.checkbox)
        self.gb_pixel_classifier.layout().addRow(self.check_paint)

        self.check_mser.setChecked(True)
        self.check_mser.toggled.connect(self.checkbox)
        self.gb_pixel_classifier.layout().addRow(self.check_mser)

        self.button_prev.clicked.connect(self.show_prev_frame)
        self.gb_video_controls.layout().addWidget(self.button_prev)

        self.button_next.clicked.connect(self.show_next_frame)
        self.gb_video_controls.layout().addWidget(self.button_next)

        self.button_rand.clicked.connect(self.show_random_frame)
        self.gb_video_controls.layout().addWidget(self.button_rand)

        self.gb_video_controls.layout().addWidget(self.frame_input)
        self.gb_video_controls.layout().addWidget(self.show_frame_b)

        self.button_refresh.clicked.connect(self.update_all)
        self.gb_pixel_classifier.layout().addRow(self.button_refresh)

        self.button_reset.clicked.connect(self.reset_classifier)
        self.gb_pixel_classifier.layout().addRow(self.button_reset)

        # self.button_done.clicked.connect(self.done)
        # self.left_panel.layout().addWidget(self.button_done)

    def reset_classifier(self):
        print "reseting pixel classifier..."

        self.painter.reset_masks()
        self.helper.rfc = None

        self.update_all()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    proj = Project()

    # proj.load('/Users/flipajs/Documents/wd/FERDA/Cam1_rfs')
    proj.load('/Users/flipajs/Documents/wd/FERDA/Cam1')
    # proj.video_paths = ['/Users/flipajs/Documents/wd/GT/C210_5000/C210.fproj']
    proj.arena_model = None
    proj.bg_model = None

    # proj.video_crop_model = {'y1': 110, 'y2': 950, 'x1': 70, 'x2': 910}

    # proj.video_paths = '/Users/flipajs/Desktop/S9T95min.avi'
    # proj.video_paths = '/Volumes/Transcend/Dropbox/FERDA/F3C51min.avi'

    # proj.video_paths = '/media/flipajs/Seagate Expansion Drive/TestSet/cuts/c6.avi'
    # proj.video_paths = '/media/flipajs/Seagate Expansion Drive/TestSet/cuts/c1.avi'
    # proj.video_paths = '/media/flipajs/Seagate Expansion Drive/TestSet/cuts/c2.avi'

    ex = SetMSERs(proj)

    ex.raise_()
    ex.showMaximized()
    ex.activateWindow()

    ex.mser_min_margin.setValue(proj.mser_parameters.min_margin)
    ex.mser_min_area.setValue(proj.mser_parameters.min_area)
    # ex.mser_max_area.setValue(proj.mser_parameters.max_area)

    # im = cv2.imread('/Users/flipajs/Downloads/trhliny/4/DSC_0327.JPG')
    # im = cv2.imread('/Users/flipajs/Downloads/trhliny/5/DSC_0348.JPG')
    # im = cv2.imread('/Users/flipajs/Downloads/trhliny/1/DSC_0297.JPG')
    # ex.set_image(im)

    app.exec_()
    app.deleteLater()
    sys.exit()
