__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore

from gui.img_controls import my_view, my_scene
from utils import video_manager
from utils.misc import get_settings, set_settings
import utils.img
import gui.gui_utils
from gui.init.class_widget import ClassWidget
from gui.init.identity_widget import IdentityWidget
from gui.init.arena.arena_mark import ArenaMark as Mark
from gui.init.arena.arena_circle import ArenaCircle as Circle
from core import colormark
import numpy as np
from core.animal import Animal
import cv2

class InitWhatWidget(QtGui.QWidget):
    def __init__(self, finish_callback, project, bg_model):
        super(InitWhatWidget, self).__init__()

        self.project = project
        self.bg_model = bg_model
        self.finish_callback = finish_callback
        self.video = video_manager.get_auto_video_manager(project.video_paths)

        self.active_id = 0
        self.animals = []
        self.identities_widgets = []
        self.scene_marks = []

        self.hbox = QtGui.QHBoxLayout()
        self.setLayout(self.hbox)

        #LEFT LAYOUT
        self.left_widget = QtGui.QWidget()
        self.left_layout = QtGui.QVBoxLayout()
        self.left_widget.setLayout(self.left_layout)
        self.left_layout.setSpacing(0)
        self.left_layout.setAlignment(QtCore.Qt.AlignTop)
        self.left_scroll = QtGui.QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setWidget(self.left_widget)
        self.left_scroll.verticalScrollBar().rangeChanged[int,int].connect(self.scrollbar_ranged_changed)

        self.hbox.addWidget(self.left_scroll)

        #Class WIDGET
        self.class_widget_box = QtGui.QGroupBox("Animal class")
        self.class_widget_box.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Maximum)
        self.class_widget = ClassWidget()
        self.class_widget_box.setLayout(QtGui.QVBoxLayout())
        self.class_widget_box.layout().addWidget(self.class_widget)
        self.left_layout.addWidget(self.class_widget_box)

        #TODO: implement GROUP widget...
        #Group WIDGET
        self.class_widget_box = QtGui.QGroupBox("Groups")
        self.class_widget_box.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Maximum)
        self.class_widget = ClassWidget()
        self.class_widget_box.setLayout(QtGui.QVBoxLayout())
        self.class_widget_box.layout().addWidget(self.class_widget)
        self.left_layout.addWidget(self.class_widget_box)

        #Group WIDGET
        self.class_widget_box = QtGui.QGroupBox("Groups")
        self.class_widget_box.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Maximum)
        self.class_widget = ClassWidget()
        self.class_widget_box.setLayout(QtGui.QVBoxLayout())
        self.class_widget_box.layout().addWidget(self.class_widget)
        self.left_layout.addWidget(self.class_widget_box)


        #SETTINGS
        self.settings_box = QtGui.QGroupBox("Settings")
        self.settings_box.setLayout(QtGui.QVBoxLayout())
        self.use_colormarks_ch = gui.gui_utils.get_checkbox('Use colormarks', 'colormarks_use')
        self.use_colormarks_ch.toggled.connect(lambda: set_settings('colormarks_use', self.use_colormarks_ch.isChecked()))
        self.settings_box.layout().addWidget(self.use_colormarks_ch)
        self.left_layout.addWidget(self.settings_box)


        #IDENTITY WIDGET
        self.identity_widget_box = QtGui.QGroupBox("Identities")
        self.identity_widget_box.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Maximum)
        self.identity_widget_box.setLayout(QtGui.QVBoxLayout())
        self.left_layout.addWidget(self.identity_widget_box)


        #GRAPHICS VIEW
        self.scene = my_scene.MyScene()
        self.scene.clicked.connect(self.scene_clicked)
        self.scene.mouse_moved.connect(self.scene_mouse_moved)
        self.graphics_view = my_view.MyView()
        self.graphics_view.setScene(self.scene)
        # self.graphics_view.double_clicked.connect(self.gv_doubleclicked)
        # self.graphics_view.mouse_moved.connect(self.gv_mouse_moved)
        # self.graphics_view.clicked.connect(self.gv_clicked)
        self.img = self.video.move2_next()
        self.scene.addPixmap(gui.gui_utils.cvimg2qtpixmap(self.img))
        self.hbox.addWidget(self.graphics_view)

        self.put_mark_active = False
        self.put_colormark_active = False

        # brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        # brush.setColor(QtGui.QColor(0, 0xFF, 0, 0x55))
        # self.circle = Circle()
        # self.circle.setBrush(brush)
        #
        # brush.setColor(QtGui.QColor(0,0xff,0,0xaa))
        # self.markc = Mark(self.circle)
        # self.markc.setBrush(brush)
        #
        # brush.setColor(QtGui.QColor(0xff,0,0,0xaa))
        # self.markr = Mark(self.circle)
        # self.markr.setBrush(brush)
        # self.markr_scene = None
        #
        # self.circle.add_points(self.markc, self.markr)

        self.left_scroll.verticalScrollBar().setSliderPosition(0)

    def add_animal_widget(self):
        animal = self.animals[self.active_id]

        center = self.scene_marks[self.active_id]['center'].pos()
        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        c_ = self.animals[self.active_id].color_
        brush.setColor(QtGui.QColor(c_[2], c_[1], c_[0], 0x55))
        self.scene_marks[self.active_id]['circle'].setBrush(brush)

        self.identity_widget = IdentityWidget(self.img, animal, center, self.scene_marks[self.active_id]['circle'].radius())
        self.identity_widget_box.layout().addWidget(self.identity_widget)

    def scene_clicked(self, pos):
        modifiers = QtGui.QApplication.keyboardModifiers()

        if self.put_mark_active:
            self.put_mark_active = False
            self.scene_marks[self.active_id]['radius'].setPos(pos.x(), pos.y())
            self.scene_marks[self.active_id]['circle'].update_geometry()

            if len(self.animals) <= self.active_id:
                animal = Animal(self.active_id)
                self.animals.append(animal)

                if not get_settings('colormarks_use', bool):
                    self.add_animal_widget()
                    self.marks_deactivate(self.active_id)

                    self.active_id += 1

        if self.put_colormark_active:
            center = self.scene_marks[self.active_id]['center'].pos()
            #TODO: replace 1 with colormark radius and improve function with weight of pixels
            color = utils.img.avg_circle_area_color(self.img, center.y(), center.x(), 1)

            #TODO REPLACE search_radius with some proper number
            p = np.array([center.y(), center.x()])
            c = colormark.get_colormark(self.img, color, p, 200)
            self.put_colormark_active = True
            animal.set_colormark(c)

        if modifiers == QtCore.Qt.ControlModifier:
            if not self.put_mark_active:
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

    def scrollbar_ranged_changed(self, min, max):
        # if max != self.left_scroll.verticalScrollBar().maximum():
        self.left_scroll.verticalScrollBar().setSliderPosition(self.left_scroll.verticalScrollBar().maximum())

    def marks_deactivate(self, id, colormark=False):
        pref = ''
        if colormark:
            pref = 'c_'

        m = self.scene_marks[id]
        m[pref+'center'].hide()
        m[pref+'radius'].setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)
        # m[pref+'circle'].setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        # m[pref+'circle'].setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtGui.QColor(0,0,0,0x55))
        m[pref+'radius'].setBrush(brush)

    def double_clicked(self, id):
        print "DOUBLECLIKCKED"

    def marks_activate(self, id, colormark=False):
        pref = ''
        if colormark:
            pref = 'c_'

        m = self.scene_marks[id]
        m[pref+'center'].show()
        m[pref+'radius'].setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)

        m[pref+'radius'].setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)



        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtGui.QColor(0,0,0,0x55))
        m[pref+'radius'].setBrush(brush)


    def get_new_scene_marks(self):
        marks = {}

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtGui.QColor(0,0xff,0,0x11))

        marks['circle'] = Circle()
        marks['circle'].setBrush(brush)
        # marks['circle'].double_clicked.connect(self.double_clicked)

        brush.setColor(QtGui.QColor(0,0xff,0xff,0xaa))
        marks['center'] = Mark(marks['circle'])
        marks['center'].setBrush(brush)

        brush.setColor(QtGui.QColor(0xff,0,0,0xaa))
        marks['radius'] = Mark(marks['circle'])
        marks['radius'].setBrush(brush)

        marks['circle'].add_points(marks['center'], marks['radius'])

        if get_settings('colormarks_use', bool):
            brush.setColor(QtGui.QColor(0, 0xFF, 0, 0x55))
            marks['c_circle'] = Circle()
            marks['c_circle'].setBrush(brush)

            brush.setColor(QtGui.QColor(0,0xff,0,0xaa))
            marks['c_center'] = Mark(marks['c_circle'])
            marks['c_center'].setBrush(brush)

            brush.setColor(QtGui.QColor(0xff,0,0,0xaa))
            marks['c_radius'] = Mark(marks['c_circle'])
            marks['c_radius'].setBrush(brush)

            marks['c_circle'].add_points(marks['c_center'], marks['c_radius'])

        return marks