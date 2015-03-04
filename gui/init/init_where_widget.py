__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore
from utils import video_manager
from gui.img_controls import my_view, utils
from gui.init.arena.arena_circle import ArenaCircle
from gui.init.arena.arena_mark import ArenaMark
import cv2
import time
from gui.init.background.bg_fix_widget import BgFixWidget
import numpy as np

class InitWhereWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project, bg_model):
        super(InitWhereWidget, self).__init__()

        self.arena_mark_size = 15

        self.bg_model = bg_model
        self.finish_callback = finish_callback

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.progress_dialog = None
        self.bg_fix_widget = None

        self.top_stripe_layout = QtGui.QHBoxLayout()
        self.vbox.addLayout(self.top_stripe_layout)

        self.video = video_manager.get_auto_video_manager(project.video_paths)
        self.first_frame = self.video.move2_next()

        self.label_instructions = QtGui.QLabel('Please select arena or confirm the suggested one.')
        self.label_instructions.setWordWrap(True)
        self.top_stripe_layout.addWidget(self.label_instructions)

        self.confirm_arena_selection = QtGui.QPushButton('Arena selection is ok, lets continue!')
        self.confirm_arena_selection.clicked.connect(self.confirm_arena_selection_clicked)
        self.top_stripe_layout.addWidget(self.confirm_arena_selection)

        self.skip_bg_model = QtGui.QPushButton('Run without background model')
        self.confirm_bg_model = QtGui.QPushButton('Everything is all right, lets continue!')
        self.confirm_bg_model.clicked.connect(self.finish)

        # image window...
        self.scene_objects = {}
        self.scene = QtGui.QGraphicsScene()
        self.graphics_view = my_view.MyView()
        self.graphics_view.setScene(self.scene)
        self.scene.addPixmap(utils.cvimg2qtpixmap(self.first_frame))
        self.bg_model_pixmap = None

        self.add_circle_selection()

        self.vbox.addWidget(self.graphics_view)

    def confirm_arena_selection_clicked(self):
        if self.bg_model.is_computed():
            self.graphics_view.hide()
            self.bg_fix_widget = BgFixWidget(self.bg_model.get_model(), self.finish)
            self.graphics_view.hide()
            self.vbox.addWidget(self.bg_fix_widget)

            if self.progress_dialog:
                self.progress_dialog.cancel()
                self.progress_dialog = None
        else:
            while True:
                time.sleep(.100)
                if not self.progress_dialog:
                    self.progress_dialog = QtGui.QProgressDialog('Computing background model. Be patient please...', QtCore.QString("Cancel"), 0, 100)
                    self.progress_dialog.setWindowTitle('Upload status')
                else:
                    self.progress_dialog.setLabelText('Computing '+str(self.bg_model.get_progress()))
                    self.progress_dialog.setValue(self.bg_model.get_progress())
                    QtGui.QApplication.processEvents()

                if self.bg_model.is_computed():
                    break

            self.confirm_arena_selection_clicked()
            return


        self.label_instructions.setText('To support FERDA performance, we are using background model. Bellow you can see background model. There should be no tracked object visible. If they are, please fix them by selecting problematic area in image. Then click f and by draggin move the green selection to area with background only. Press ctrl+z if you don\'t like the result for new selection."')
        self.confirm_arena_selection.setHidden(True)

        self.top_stripe_layout.addWidget(self.skip_bg_model)
        self.top_stripe_layout.addWidget(self.confirm_bg_model)

    def add_circle_selection(self):
        self.arena_ellipse = ArenaCircle()

        y, x, r = self.give_me_best_circle()

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtGui.QColor(0xff,0,0,0xaa))
        self.c_center = ArenaMark(self.arena_ellipse, self.update_circle_labels)
        self.c_center.setRect(0, 0, self.arena_mark_size, self.arena_mark_size)
        self.c_center.setBrush(brush)
        self.c_center.setPos(x - self.arena_mark_size, y-self.arena_mark_size)

        brush.setColor(QtGui.QColor(0,0,0xff,0xaa))
        self.c_radius = ArenaMark(self.arena_ellipse, self.update_circle_labels)
        self.c_radius.setRect(0, 0, self.arena_mark_size, self.arena_mark_size)
        self.c_radius.setBrush(brush)
        self.c_radius.setPos(x+r-self.arena_mark_size, y-self.arena_mark_size)

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
        circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 2, 20,
                            param1=canny_,param2=param2_,minRadius=min_radius,maxRadius=max_radius)

        if circles is not None and len(circles) > 0:
            # as the circles are ordered by number of votes, choose the first one
            c = circles[0,0,:]
            return c[1], c[0], c[2]
        else:
            return self.first_frame.shape[1]/2, self.first_frame.shape[0]/2, min(self.first_frame.shape[0], self.first_frame.shape[1])*0.45

    def finish(self):
        #TODO save values...
        self.bg_fix_widget.hide()
        self.graphics_view.show()
        self.finish_callback('init_where_finished')