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
from functools import partial
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

        # ACTIONS
        self.esc_action = QtGui.QAction("ESC", self)
        self.esc_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape))
        self.esc_action.triggered.connect(self.esc_action_)
        self.addAction(self.esc_action)


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
        self.class_widget.updated.connect(self.update_identity_panel_classes)

        #TODO: implement GROUP widget...

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
        self.scene.double_clicked.connect(self.scene_double_clicked)
        self.graphics_view = my_view.MyView()
        self.graphics_view.setScene(self.scene)

        self.img = self.video.move2_next()
        self.scene.addPixmap(gui.gui_utils.cvimg2qtpixmap(self.img))
        self.hbox.addWidget(self.graphics_view)

        self.put_mark_active = False
        self.put_colormark_active_state = -1

        self.left_scroll.verticalScrollBar().setSliderPosition(0)

    def add_animal_widget(self, id=-1):
        id_ = self.active_id if id < 0 else id

        print id_

        animal = self.animals[id_]

        center = self.scene_marks[id_]['center'].pos()

        identity_widget = IdentityWidget(self.img, animal, center, self.scene_marks[id_]['circle'].radius(), self.class_widget.classes, self.class_widget.classes)
        identity_widget.update_identity.connect(self.update_identity)
        identity_widget.delete_button.clicked.connect(partial(self.delete_identity, id_))

        if id < 0:
            self.identity_widget_box.layout().addWidget(identity_widget)
            self.identities_widgets.append(identity_widget)
        else:
            self.identity_widget_box.layout().removeItem(self.identity_widget_box.layout().itemAt(id))
            self.identity_widget_box.layout().insertWidget(id, identity_widget)
            self.identities_widgets.remove(self.identities_widgets[id])
            self.identities_widgets.insert(id, identity_widget)

    def compute_colormark(self, id=-1):
        id_ = self.active_id if id < 0 else id

        center = self.scene_marks[id_]['c_center'].pos()
        #TODO: replace 1 with colormark radius and improve function with weight of pixels
        color = utils.img.avg_circle_area_color(self.img, center.y(), center.x(), 1)

        #TODO REPLACE search_radius with some proper number
        p = np.array([center.y(), center.x()])
        c = colormark.get_colormark(self.img, color, p, 200)
        self.animals[id_].set_colormark(c)

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
                    self.active_id += 1
                else:
                    self.put_colormark_active_state = 0

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

            elif self.put_colormark_active_state == 1:
                self.compute_colormark()
                self.put_colormark_active_state = -1

                self.add_animal_widget()
                self.marks_deactivate(self.active_id)
                self.marks_deactivate(self.active_id, True)
                self.active_id += 1

        if modifiers == QtCore.Qt.ControlModifier and self.put_colormark_active_state < 0:
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

        if self.put_colormark_active_state == 1:
            y = pos.y()
            x = pos.x()

            self.scene_marks[self.active_id]['c_radius'].setPos(x, y)
            self.scene_marks[self.active_id]['c_circle'].update_geometry()

    def scene_double_clicked(self, pos, items):
        for it in items:
            if isinstance(it, Circle):
                colormark = False
                if get_settings('colormarks_use'):
                    for i in range(len(self.animals)):
                        if self.scene_marks[i]['c_circle'] == it:
                            colormark = True
                            break

                if it.a.flags() & QtCore.Qt.ItemIsSelectable:
                    self.marks_deactivate(it.id, colormark)
                    if colormark:
                        self.compute_colormark(it.id)

                    self.add_animal_widget(it.id)
                else:
                    self.marks_activate(it.id, colormark)

    def scrollbar_ranged_changed(self, min, max):
        self.left_scroll.verticalScrollBar().setSliderPosition(self.left_scroll.verticalScrollBar().maximum())

    def marks_deactivate(self, id, colormark=False):
        pref = ''
        if colormark:
            pref = 'c_'

        m = self.scene_marks[id]
        m[pref+'center'].hide()
        if colormark:
            m[pref+'radius'].hide()

        m[pref+'radius'].setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)
        m[pref+'radius'].setFlag(QtGui.QGraphicsItem.ItemIsMovable, False)

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtGui.QColor(0, 0, 0, 0x55))
        m[pref+'radius'].setBrush(brush)

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        c_ = self.animals[id].color_
        if colormark:
            brush.setColor(QtGui.QColor(0, 0xff, 0, 0x22))
        else:
            brush.setColor(QtGui.QColor(c_[2], c_[1], c_[0], 0x55))

        self.scene_marks[id][pref+'circle'].setBrush(brush)

    def marks_activate(self, id, colormark=False):
        pref = ''
        if colormark:
            pref = 'c_'

        print "PREF ", pref

        m = self.scene_marks[id]
        m[pref+'center'].show()
        if colormark:
            m[pref+'radius'].show()

        m[pref+'radius'].setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        m[pref+'radius'].setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtGui.QColor(0xff,0,0,0xaa))
        m[pref+'radius'].setBrush(brush)


        brush.setColor(QtGui.QColor(0,0xff,0,0x11))
        m[pref+'circle'].setBrush(brush)

    def delete_identity(self, id):
        self.identity_widget_box.layout().removeItem(self.identity_widget_box.layout().itemAt(id))

        for i in range(id, len(self.animals)):
            self.identities_widgets[i].delete_button.clicked.disconnect()
            self.identities_widgets[i].delete_button.clicked.connect(partial(self.delete_identity, i-1))
            self.animals[i].id = i - 1

        self.animals.remove(self.animals[id])
        self.identities_widgets.remove(self.identities_widgets[id])

        m = self.scene_marks[id]
        self.scene.removeItem(m['circle'])
        self.scene.removeItem(m['radius'])
        self.scene.removeItem(m['center'])

        try:
            self.scene.removeItem(m['c_circle'])
            self.scene.removeItem(m['c_radius'])
            self.scene.removeItem(m['c_center'])
        except:
            pass

        self.scene_marks.remove(self.scene_marks[id])
        self.active_id -= 1

    def update_identity(self, id):
        print "UPDATE", id
        self.marks_deactivate(id)

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

        if get_settings('colormarks_use', bool):
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

    def update_identity_panel_classes(self):
        for i in range(len(self.animals)):
            # w = self.identity_widget_box.layout().itemAt(i)
            self.identities_widgets[i].update_classes(self.class_widget.classes)

            # w.update_classes(self.class_widget.classes)

    def esc_action_(self):
        self.put_mark_active = False

        m = self.scene_marks[self.active_id]
        self.scene.removeItem(m['circle'])
        self.scene.removeItem(m['radius'])
        self.scene.removeItem(m['center'])

        self.scene_marks.remove(m)