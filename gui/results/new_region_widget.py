__author__ = 'fnaiser'

import math
from functools import partial

import cv2
import numpy as np
from PyQt4 import QtGui, QtCore

from gui.img_controls.my_scene import MyScene
from gui.init.arena.arena_circle import ArenaCircle as Circle
from gui.init.arena.arena_mark import ArenaMark as Mark
from gui.settings import Settings as S_


class NewRegionWidget(QtGui.QWidget):
    def __init__(self, im, offset, frame, callback):
        super(NewRegionWidget, self).__init__()

        self.setLayout(QtGui.QVBoxLayout())
        self.im = im
        self.offset = offset
        self.frame = frame
        self.callback = callback

        self.view = QtGui.QGraphicsView()
        self.view.setMouseTracking(True)
        self.scene = MyScene()
        self.scene.clicked.connect(self.scene_clicked)
        self.scene.mouse_moved.connect(self.scene_mouse_moved)
        self.view.setScene(self.scene)

        self.scene.addPixmap(im)
        self.layout().addWidget(self.view)
        self.confirm_button = QtGui.QPushButton('confirm')
        self.confirm_button.clicked.connect(self.confirm)
        self.cancel_button = QtGui.QPushButton('cancel')
        self.cancel_button.clicked.connect(partial(self.callback, False, None))
        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.confirm_button)
        self.hbox.addWidget(self.cancel_button)
        self.layout().addLayout(self.hbox)

        self.put_mark_active = False
        self.put_colormark_active_state = -1
        self.active_id = 0
        self.identities_widgets = []
        self.scene_marks = []

    def confirm(self):
        m = self.scene_marks[0]
        c = m['center'].pos()
        c = np.array([c.x(), c.y()])

        r = m['radius'].pos()
        r = np.array([r.x(), r.y()])

        th = np.rad2deg(math.atan2(r[1]-c[1], r[0]-c[0]))

        a_ = math.ceil(np.linalg.norm(c-r) * 0.75)
        el_c = (int(a_), int(a_))

        h_ = 2*a_ + 1
        im = np.zeros((h_, h_, 3))
        cv2.ellipse(im, el_c, (int(math.ceil(a_)), int(math.ceil(a_/3.))), int(round(th)), 0, 360, (255, 255, 0), -1)
        ids = im[:, :, 1] > 0
        pts = []
        for y in range(ids.shape[0]):
            for x in range(ids.shape[1]):
                if ids[y][x]:
                    pts.append([y-a_, x-a_])

        c = np.array([c[1], c[0]])
        pts = pts+self.offset+c
        pts = np.array(pts, dtype=np.uint32)

        data = {'pts': pts, 'centroid': self.offset+c, 'frame': self.frame}
        self.callback(True, data)

    def scene_clicked(self, pos):
        modifiers = QtGui.QApplication.keyboardModifiers()

        if self.put_mark_active:
            self.put_mark_active = False
            self.scene_marks[self.active_id]['radius'].setPos(pos.x(), pos.y())
            self.scene_marks[self.active_id]['circle'].update_geometry()

        elif self.put_colormark_active_state > -1:
            if self.put_colormark_active_state == 0:
                if modifiers == QtCore.Qt.ControlModifier:
                    self.put_colormark_active_state += 1

                    self.scene_marks[self.active_id]['c_center'].setPos(pos.x(), pos.y())
                    self.scene_marks[self.active_id]['c_radius'].setPos(pos.x(), pos.y())

                    self.scene.addItem(self.scene_marks[self.active_id]['c_circle'])
                    self.scene.addItem(self.scene_marks[self.active_id]['c_center'])
                    self.scene.addItem(self.scene_marks[self.active_id]['c_radius'])

                    self.scene_marks[self.active_id]['c_circle'].update_geometry()

        if modifiers == QtCore.Qt.ControlModifier and self.put_colormark_active_state < 0:
            if not self.put_mark_active and len(self.scene_marks) == 0:
                self.put_mark_active = True

                marks = self.get_new_scene_marks()
                self.scene_marks.append(marks)

                marks['center'].setPos(pos.x(), pos.y())
                marks['radius'].setPos(pos.x(), pos.y())

                self.scene.addItem(marks['center'])
                self.scene.addItem(marks['radius'])
                self.scene.addItem(marks['circle'])

                marks['circle'].update_geometry()

    def scene_mouse_moved(self, pos):
        if self.put_mark_active:
            y = pos.y()
            x = pos.x()

            self.scene_marks[self.active_id]['radius'].setPos(x, y)
            self.scene_marks[self.active_id]['circle'].update_geometry()

        if self.put_colormark_active_state == 1:
            y = pos.y()
            x = pos.x()

            self.scene_marks[self.active_id]['c_radius'].setPos(x, y)
            self.scene_marks[self.active_id]['c_circle'].update_geometry()


    def get_new_scene_marks(self):
        marks = {}

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtGui.QColor(0,0xff,0,0x11))

        marks['circle'] = Circle(self.active_id)
        marks['circle'].setBrush(brush)
        # marks['circle'].double_clicked.connect(self.double_clicked)

        brush.setColor(QtGui.QColor(0,0xff,0xff,0xaa))
        marks['center'] = Mark(marks['circle'])
        marks['center'].setBrush(brush)

        brush.setColor(QtGui.QColor(0xff,0,0,0xaa))
        marks['radius'] = Mark(marks['circle'])
        marks['radius'].setBrush(brush)

        marks['circle'].add_points(marks['center'], marks['radius'])

        if S_.colormarks.use:
            brush.setColor(QtGui.QColor(0, 0xFF, 0, 0x11))
            marks['c_circle'] = Circle(self.active_id)
            marks['c_circle'].setBrush(brush)

            brush.setColor(QtGui.QColor(0,0xff,0,0xaa))
            marks['c_center'] = Mark(marks['c_circle'])
            marks['c_center'].setBrush(brush)

            brush.setColor(QtGui.QColor(0xff,0,0,0x11))
            marks['c_radius'] = Mark(marks['c_circle'])
            marks['c_radius'].setBrush(brush)

            marks['c_circle'].add_points(marks['c_center'], marks['c_radius'])

        return marks