from core.arena.paint_mask import PaintMask
from gui.arena.arena_editor import ArenaEditor
from utils.video_manager import get_auto_video_manager
from math import ceil, floor
from core.arena.circle import Circle

__author__ = 'fnaiser'

import time

from PyQt4 import QtGui, QtCore
import cv2
import numpy as np

from utils import video_manager
from gui.img_controls import my_view, gui_utils
from gui.init.arena.arena_circle import ArenaCircle
from gui.init.arena.arena_mark import ArenaMark
from gui.init.background.bg_fix_widget import BgFixWidget
from core.arena.circle import Circle
from core.bg_model.model import Model
from core.bg_model.bg_model import BGModel


class CircleArenaEditorWidget(QtGui.QWizardPage):
    def __init__(self):
        super(CircleArenaEditorWidget, self).__init__()
        self.arena_mark_size = 15
        self.project = None

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.progress_dialog = None
        self.bg_fix_widget = None

        self.top_stripe_layout = QtGui.QHBoxLayout()
        self.vbox.addLayout(self.top_stripe_layout)

        self.video = None
        self.first_frame = None

        self.label_instructions = QtGui.QLabel('<i>Please select region of interest by dragging red and blue dot or confirm the suggested one.</i>')
        self.label_instructions.setWordWrap(True)
        self.top_stripe_layout.addWidget(self.label_instructions)

        self.use_advanced_arena_editor = QtGui.QPushButton('Use polygon arena editor')
        self.use_advanced_arena_editor.clicked.connect(self.use_advanced_editor)
        self.use_advanced_arena_editor.setDisabled(False)
        self.top_stripe_layout.addWidget(self.use_advanced_arena_editor)

        self.skip_bg_model = QtGui.QPushButton('Run without background model')
        self.skip_bg_model.clicked.connect(self.skip_bg_clicked)
        self.confirm_bg_model = QtGui.QPushButton('Everything is all right, lets continue!')
        self.confirm_bg_model.clicked.connect(self.finish)

        # image window...
        self.scene_objects = {}
        self.scene = QtGui.QGraphicsScene()
        self.graphics_view = my_view.MyView()
        self.graphics_view.setScene(self.scene)
        self.bg_model_pixmap = None

        self.vbox.addWidget(self.graphics_view)

    def initializePage(self):
        self.project = self.wizard().project
        self.video = get_auto_video_manager(self.project)
        self.first_frame = self.video.next_frame()
        self.scene.addPixmap(gui_utils.cvimg2qtpixmap(self.first_frame))
        self.add_circle_selection()

    def validatePage(self):
        self.save_arena_to_project()
        return True

    def use_advanced_editor(self):
        self.advanced_editor = ArenaEditor(self.first_frame, self.project, finish_callback=self.advanced_editor_done)
        self.advanced_editor.exec_()

    def advanced_editor_done(self, arena_mask, occultation_mask):
        self.advanced_editor.hide()
        h_, w_, _ = self.first_frame.shape
        self.project.arena_model = PaintMask(h_, w_)
        self.project.arena_model.set_mask(arena_mask != 0)
        # if isinstance(self.project.bg_model, BGModel) or self.project.bg_model.is_computed():
        if True:
            if isinstance(self.project.bg_model, Model):
                self.project.bg_model = self.project.bg_model.get_model()

            # self.graphics_view.hide()
            # self.bg_fix_widget = BgFixWidget(self.project.bg_model.img(), self.finish)
            self.bg_fix_widget = QtGui.QWidget()
            # self.graphics_view.hide()
            # self.vbox.addWidget(self.bg_fix_widget)

            # if self.progress_dialog:
            #     self.progress_dialog.cancel()
            #     self.progress_dialog = None
        else:
            while True:
                time.sleep(.100)
                if not self.progress_dialog:
                    self.progress_dialog = QtGui.QProgressDialog('Computing background model. Be patient please...', QtCore.QString("Cancel"), 0, 100)
                    self.progress_dialog.setWindowTitle('Upload status')
                else:
                    self.progress_dialog.setLabelText('Computing '+str(self.project.bg_model.get_progress()))
                    self.progress_dialog.setValue(self.project.bg_model.get_progress())
                    QtGui.QApplication.processEvents()

                if self.project.bg_model.is_computed():
                    break

            self.confirm_arena_selection_clicked()
            return

        # self.label_instructions.setText('To support FERDA performance, we are using background model. Bellow you can see background model. There should be no tracked object visible. If they are, please fix them by selecting problematic area in image. Then click f and by draggin move the green selection to area with background only. Press ctrl+z if you don\'t like the result for new selection."')
        # self.confirm_arena_selection.setHidden(True)
        # self.use_advanced_arena_editor.setHidden(True)
        #
        # self.top_stripe_layout.addWidget(self.skip_bg_model)
        # self.top_stripe_layout.addWidget(self.confirm_bg_model)
        next(self.wizard())

    def add_circle_selection(self):
        self.arena_ellipse = ArenaCircle()

        y, x, r = self.give_me_best_circle()

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtGui.QColor(0xff,0,0,0xaa))
        self.c_center = ArenaMark(self.arena_ellipse, self.update_circle_labels, radius=self.arena_mark_size)
        self.c_center.setBrush(brush)
        self.c_center.setPos(x, y)

        brush.setColor(QtGui.QColor(0,0,0xff,0xaa))
        self.c_radius = ArenaMark(self.arena_ellipse, self.update_circle_labels, radius=self.arena_mark_size)
        self.c_radius.setBrush(brush)
        self.c_radius.setPos(x+r, y)

        brush.setColor(QtGui.QColor(0, 0xFF, 0, 0x55))
        self.arena_ellipse.setBrush(brush)

        self.arena_ellipse.add_points(self.c_center, self.c_radius)
        self.arena_ellipse.update_geometry()
        self.scene_objects['arena_ellipse'] = self.scene.addItem(self.arena_ellipse)
        self.scene_objects['c_center'] = self.scene.addItem(self.c_center)
        self.scene_objects['c_radius'] = self.scene.addItem(self.c_radius)

    def update_circle_labels(self):
        item = lambda x: QtGui.QTableWidgetItem(QtCore.QString.number(int(x)))
        # print item

    def give_me_best_circle(self):
        gray = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)
        min_radius = min(self.first_frame.shape[0], self.first_frame.shape[1]) / 10
        max_radius = min(self.first_frame.shape[0], self.first_frame.shape[1]) / 2

        canny_ = 100
        # 75% of min_radius points must vote

        param2_ = int(min_radius * 3.14 * 2 * 0.75)
        if 'HOUGH_GRADIENT' in dir(cv2):
            method = cv2.HOUGH_GRADIENT  # 3.x
        else:
            method = cv2.cv.CV_HOUGH_GRADIENT  # 2.x
        circles = cv2.HoughCircles(gray, method, 2, 20,
            param1=canny_,param2=param2_,minRadius=min_radius,maxRadius=max_radius)

        if circles is not None and len(circles) > 0:
            # as the circles are ordered by number of votes, choose the first one
            c = circles[0,0,:]
            return c[1], c[0], c[2]
        else:
            return self.first_frame.shape[1]/2, self.first_frame.shape[0]/2, min(self.first_frame.shape[0], self.first_frame.shape[1])*0.45

    def skip_bg_clicked(self):
        print("SKIP")
        self.project.bg_model = None
        self.finish()

    def save_arena_to_project(self):
        if self.project.arena_model is None:
            c = np.array(
                [self.arena_ellipse.c.pos().y(), self.arena_ellipse.c.pos().x()])
            r = np.array(
                [self.arena_ellipse.a.pos().y(), self.arena_ellipse.a.pos().x()])
            r = np.linalg.norm(c - r)

            im = self.video.next_frame()

            video_crop_model = {
                'y1': int(max(0, floor(c[0] - r))),
                'x1': int(max(0, floor(c[1] - r))),
                'y2': int(min(im.shape[0], ceil(c[0] + r))),
                'x2': int(min(im.shape[1], ceil(c[1] + r))),
            }

            self.project.video_crop_model = video_crop_model

            c = np.array([c[0] - video_crop_model['y1'], c[1] - video_crop_model['x1']])
            self.project.arena_model = Circle(video_crop_model['y2'] - video_crop_model['y1'],
                                              video_crop_model['x2'] - video_crop_model['x1'])
            self.project.arena_model.set_circle(c, r)

    def finish(self):
        #TODO save values...
        self.bg_fix_widget.hide()
        self.graphics_view.show()
        if self.project.bg_model:
            self.project.bg_model.update(np.copy(self.bg_fix_widget.image))

        self.finish_callback('init_where_finished')
