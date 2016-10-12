import cPickle as pickle
import sys
import time

import cv2
import os

import numpy as np
from PIL import ImageQt
from PyQt4 import QtGui, QtCore

from core.segmentation_helper import SegmentationHelper
from core.project.project import Project
from core.region.mser import ferda_filtered_msers
from gui.gui_utils import SelectableQLabel
from gui.img_grid.img_grid_widget import ImgGridWidget
from gui.segmentation.painter import Painter, array2qimage
from utils.drawing.points import draw_points_crop, get_contour, draw_points_binary
from utils.img import prepare_for_segmentation
from utils.video_manager import get_auto_video_manager

__author__ = 'filip@naiser.cz', 'dita'


def gen_filtered_listdir(path):
    for f in os.listdir(path):
        if f in ['.DS_Store', 'Thumbs.db']:
            continue

        yield f


class SetMSERs(QtGui.QWidget):
    def __init__(self, wd, gt_dir, prob_color=(0, 255, 0, 200),
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

        self.project = None
        self.vid = None

        self.PER_LVL = 36

        try:
            os.mkdir(gt_dir)
        except OSError:
            pass

        self.wd = wd
        self.gt_dir = gt_dir

        self.im_paths = [wd+'/'+p for p in gen_filtered_listdir(wd)]
        self.im_i = 0

        self.im = None
        self.load_im()

        self.w, self.h, c = self.im.shape

        self.use_segmentation_ = False
        self.segmentation = None

        # Setup colors
        self.color_prob = prob_color
        self.color_foreground = foreground_color
        self.color_background = background_color

        # Setup painting tools
        self.pen_size = 5
        self.cur_color = "background"
        self.cur_eraser = False

        self.update_rfc = False

        # Setup painter
        r, g, b, a = self.color_background
        self.painter = Painter(self.im, paint_name="background", paint_r=r, paint_g=g, paint_b=b, paint_a=a)
        self.painter.add_color_("foreground", self.color_foreground)

        # Setup segmentation helper
        self.helper = SegmentationHelper(num=2)
        self.helper.set_image(self.im, self.get_angle(), self.get_lvl())

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


        self.add_actions()

        self.update_all()
        self.show()

    def load_im(self):
        self.im = cv2.imread(self.im_paths[self.im_i])

    def update_all(self):
        """
        Computes new probability map and msers, also updates GUI. Can be called after parameter/image change as a short
        for calling update_paint() and update_mser()
        :return: None
        """
        # paint must be updated first, because segmentation results are used in msers
        self.update_paint()
        if self.update_rfc:
            self.update_img()

        # self.update_mser()

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

        if self.update_rfc:
            # obtain segmentation image from helper
            self.helper.train(background, foreground)

        # stop cursor animation
        QtGui.QApplication.restoreOverrideCursor()

    def update_img(self):
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.segmentation = self.helper.predict()

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
        self.im_i += 1
        if self.im_i >= len(self.im_paths):
            self.im_i = 0
            print "max im_i"

        self.load_im()
        self.set_image(self.im)

    def get_angle(self):
        return self.im_i % self.PER_LVL

    def get_lvl(self):
        return self.im_i / self.PER_LVL

    def show_prev_frame(self):
        """
        Show current settings on next frame
        :return: None
        """
        self.im_i -= 1
        if self.im_i < 0:
            self.im_i = len(self.im_paths) - 1
            print "0 im_i"

        self.load_im()
        self.set_image(self.im)

    def show_random_frame(self):
        """
        Show current settings on random frame
        :return: None
        """
        import random
        self.im_i = random.randint(0, len(self.im_paths))

        self.load_im()
        self.set_image(self.im)

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
        self.helper.set_image(self.im, self.get_angle(), self.get_lvl())
        self.update_img()
        # self.update_mser()
        # self.update_all()

    def done(self):
        self.helper.update_xy()

        # TODO:...
        with open(self.gt_dir+'/segmentation_model.pkl', 'wb') as f:
        # with open('/Users/flipajs/Documents/wd/3Doid/sub4_photos/bg_model.pkl', 'wb') as f:
            pickle.dump(self.helper, f, -1)

    def load_helper(self):
        # with open(self.gt_dir + '/segmentation_model.pkl', 'rb') as f:
        with open('/Users/flipajs/Documents/wd/3Doid/sub4_photos/bg_model.pkl', 'rb') as f:
            self.helper = pickle.load(f)

        self.update_rfc = True
        self.set_image(self.im)
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
        # self.painter.set_overlay2_visible(self.check_mser.isChecked())

    def val_changed(self):
        return

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
        # self.update_mser()

    def add_actions(self):
        from utils import gui_coding

        gui_coding.add_action(self, 'bg_check', QtCore.Qt.SHIFT + QtCore.Qt.Key_B,
                              lambda x: self.check_bg.setChecked(not self.check_bg.isChecked()))

        gui_coding.add_action(self, 'proba_check', QtCore.Qt.SHIFT + QtCore.Qt.Key_P,
                              lambda x: self.check_prob.setChecked(not self.check_prob.isChecked()))

        gui_coding.add_action(self, 'background_brush', QtCore.Qt.Key_B, self.set_color_bg)

        gui_coding.add_action(self, 'foreground_brush', QtCore.Qt.Key_F, self.set_color_fg)

        gui_coding.add_action(self, 'eraser', QtCore.Qt.Key_E, self.set_eraser)

        gui_coding.add_action(self, 'increase_brush_size', QtCore.Qt.Key_2,
                              lambda x: self.slider.setValue(self.slider.value() + 2))

        gui_coding.add_action(self, 'decrease_brush_size', QtCore.Qt.Key_1,
                              lambda x: self.slider.setValue(self.slider.value() - 2))

        gui_coding.add_action(self, 'pause_RFC_updating', QtCore.Qt.Key_Space,
                              lambda x: setattr(self, 'update_rfc', not getattr(self, 'update_rfc')))


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

        self.button_next = QtGui.QPushButton("Next")
        self.button_prev = QtGui.QPushButton("Prev")
        self.button_rand = QtGui.QPushButton("Random")

        self.button_next_lvl = QtGui.QPushButton('next lvl')
        self.button_prev_lvl = QtGui.QPushButton('prev lvl')

        self.button_done = QtGui.QPushButton("Done")

        self.load_button = QtGui.QPushButton("load model")

        self.save_gt_button = QtGui.QPushButton("save GT")

    def configure_form_panel(self):
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
        self.slider.setValue(self.pen_size * 2)
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

        self.button_next.clicked.connect(self.show_next_frame)
        self.left_panel.layout().addWidget(self.button_next)

        self.button_prev.clicked.connect(self.show_prev_frame)
        self.left_panel.layout().addWidget(self.button_prev)

        self.button_rand.clicked.connect(self.show_random_frame)
        self.left_panel.layout().addWidget(self.button_rand)

        self.button_done.clicked.connect(self.done)
        self.left_panel.layout().addWidget(self.button_done)

        self.button_next_lvl.clicked.connect(self.next_lvl)
        self.left_panel.layout().addWidget(self.button_next_lvl)

        self.button_prev_lvl.clicked.connect(self.prev_lvl)
        self.left_panel.layout().addWidget(self.button_prev_lvl)

        self.im_path_input = QtGui.QLineEdit()
        self.left_panel.layout().addWidget(self.im_path_input)

        self.load_im_button = QtGui.QPushButton('load img')
        self.load_im_button.clicked.connect(self.load_im_path)
        self.left_panel.layout().addWidget(self.load_im_button)

        self.save_gt_button.clicked.connect(self.save_gt)
        self.left_panel.layout().addWidget(self.save_gt_button)

        self.load_button.clicked.connect(self.load_helper)
        self.left_panel.layout().addWidget(self.load_button)

    def next_lvl(self):
        self.im_i += self.PER_LVL
        if self.im_i >= len(self.im_paths):
            self.im_i %= self.PER_LVL

        self.load_im()
        self.set_image(self.im)

    def prev_lvl(self):
        self.im_i -= self.PER_LVL
        if self.im_i < 0:
            self.im_i += len(self.im_paths)

        self.load_im()
        self.set_image(self.im)

    def save_gt(self):
        # TODO:
        pass

    def load_im_path(self):
        path = str(self.im_path_input.text())
        new_im = cv2.imread(path)
        self.set_image(new_im)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    proj = Project()

    wd = '/Users/flipajs/Documents/wd/3Doid/sub4_photos/test'
    # wd = '/Users/flipajs/Documents/wd/3Doid/sub4_photos/hannspree_4711404021589'
    gt_dir = '/Users/flipajs/Documents/wd/3Doid/sub4_photos/GT/test'

    print "Done loading"

    ex = SetMSERs(wd, gt_dir)
    ex.raise_()
    ex.showMaximized()
    ex.activateWindow()

    app.exec_()
    app.deleteLater()
    sys.exit()
