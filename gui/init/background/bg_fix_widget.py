from PyQt4 import QtGui, QtCore
from gui.img_controls.my_view import MyView
import cv2
import os
from gui.img_controls import utils
from gui.init.background import settingsdialog
import copy
import math
import thread
import numpy as np

class BgFixWidget(QtGui.QWidget):
    """A tool for correcting detected background of a video. Let's users to select a part of image they want to copy onto
    and than a part of image they want to copy from"""

    def __init__(self, img, finish_callback):
        super(BgFixWidget, self).__init__()

        self.finish_callback = finish_callback

        self.pix_map = None
        self.pix_map_item = None
        self.image = None
        self.cv_image = None
        self.pos_marker = None
        self.copy_marker = None
        self.action_list = []

        self.central_layout = QtGui.QVBoxLayout()
        self.central_layout.setMargin(0)
        self.setLayout(self.central_layout)

        # self.setStatusBar(QtGui.QStatusBar())
        # self.statusBar().setSizeGripEnabled(False)

        # self.setMenuBar(QtGui.QMenuBar())

        # self.toolbar = self.addToolBar('Test')

        self.open_action = QtGui.QAction("Open image", self)
        self.save_action = QtGui.QAction("Save image", self)
        self.fix_action = QtGui.QAction("Fix image", self)
        self.settings_action = QtGui.QAction("Settings", self)
        self.restore_action = QtGui.QAction("Reset image", self)
        self.finish_action = QtGui.QAction("Finish", self)
        self.cancel_action = QtGui.QAction("Cancel fixing", self)
        self.undo_action = QtGui.QAction("Undo", self)

        self.addAction(self.fix_action)
        self.addAction(self.finish_action)
        self.addAction(self.undo_action)
        # self.toolbar.addAction(self.fix_action)
        # self.toolbar.addAction(self.settings_action)
        # self.toolbar.addAction(self.restore_action)
        # self.toolbar.addAction(self.finish_action)
        # self.toolbar.addAction(self.cancel_action)
        # self.toolbar.addAction(self.undo_action)
        # self.toolbar.addAction(self.save_action)

        self.init_actions()

        self.scene = QtGui.QGraphicsScene()
        self.graphics_view = MyView(self)
        self.central_layout.addWidget(self.graphics_view)
        self.graphics_view.setScene(self.scene)

        self.connect_gui()
        self.set_shortcuts()

        self.graphics_view.areaSelected.connect(self.drag_ended)

        # thread.lock.__new__()
        self.img_original = np.copy(img)
        self.prev_img = np.copy(img)
        self.image = np.copy(img)
        self.update_background()

        w = self.geometry().width()
        h = self.geometry().height()
        top_margin = 30
        tab_w = 0
        self.graphics_view.setGeometry(tab_w + 5, top_margin, w - tab_w - 5 - 1, h - top_margin - 1)

        self.update()
        self.show()


        #DEBUG COMMANDS
        # self.load_image('background.jpg')
        #END OF DEBUG COMMANDS

    def connect_gui(self):
        """Connects GUI elements with appropriate methods"""
        self.open_action.triggered.connect(self.open_image)
        self.fix_action.triggered.connect(self.fix_image)
        self.save_action.triggered.connect(self.save_image)
        self.cancel_action.triggered.connect(self.cancel_fixing)
        self.settings_action.triggered.connect(self.show_settings)
        self.restore_action.triggered.connect(self.reset_image)
        self.finish_action.triggered.connect(self.finish)
        self.undo_action.triggered.connect(self.undo)

    def set_shortcuts(self):
        """Sets shortcuts for actions"""
        self.open_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_O))
        self.save_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_S))
        self.fix_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_F))
        self.cancel_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape))
        self.finish_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter))
        self.undo_action.setShortcut(QtGui.QKeySequence.Undo)

    def init_actions(self):
        self.action_list.append(self.open_action)
        self.action_list.append(self.save_action)
        self.action_list.append(self.fix_action)
        self.action_list.append(self.settings_action)
        self.action_list.append(self.cancel_action)
        self.open_action.setObjectName('open_image')
        self.save_action.setObjectName('save_image')
        self.fix_action.setObjectName('fix_image')
        self.settings_action.setObjectName('settings')
        self.cancel_action.setObjectName('cancel_fixing')

    def fix_image(self):
        if self.image is not None:
            if self.pos_marker is None:
                print "No area selected. Please select area to fix first."
                # self.statusBar().showMessage("No area selected. Please select area to fix first.")
            elif self.copy_marker is not None:
                successed = self.correct_background()
                if not successed:
                    print "One or both of the rectangles is not entirely in the image."
                    # self.statusBar().showMessage("One or both of the rectangles is not entirely in the image.")
                else:
                    self.remove_copy_marker()
                    self.remove_pos_marker()
                    # self.statusBar().showMessage("Select an area you want to fix.")
            else:
                # self.statusBar().showMessage("Move the new square to the area you wish to be copied into selected space and press fix again.")
                mouse_coords = self.graphics_view.mapToScene(self.graphics_view.mapFromGlobal(QtGui.QCursor.pos()))
                self.add_copy_marker(mouse_coords.x() - self.pos_marker.rect().width()/2, mouse_coords.y() - self.pos_marker.rect().height()/2, self.pos_marker.rect().width(), self.pos_marker.rect().height())

    def open_image(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, "Open image", "", "Images(*.*)")
        if filename != "":
            filename = unicode(filename)
            self.load_image(filename)
            # self.statusBar().showMessage("Select an area you want to fix.")

    def load_image(self, filename):
        filename = os.path.normpath(filename)

        #cv2.imread has inexplicable problems with non-ascii paths, so the image loading has to be done like this

        img = QtGui.QImage(filename)
        img = img.convertToFormat(QtGui.QImage.Format_RGB888)
        img = img.rgbSwapped()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.asarray(ptr).reshape((img.height(), img.width(), 3))
        self.image = copy.deepcopy(arr)
        self.update_background()

    def save_image(self):
        if self.image is not None:
            filename = QtGui.QFileDialog.getSaveFileName(self, "Save image", "", "Images(*.jpg	)")
            if filename != "":
                image = self.pix_map.toImage()
                image.save(filename, quality=100)

    def show_settings(self):
        dialog = settingsdialog.SettingsDialog(self, actionlist=self.action_list)
        dialog.exec_()

    def drag_ended(self, point_one, point_two):
        if self.copy_marker is None:
            if self.pos_marker is not None:
                self.remove_pos_marker()
            self.add_pos_marker(point_one, point_two)

    def add_pos_marker(self, point_one, point_two):
        settings = QtCore.QSettings("Background corrector")

        point_one = self.graphics_view.mapToScene(point_one)
        point_two = self.graphics_view.mapToScene(point_two)
        width = point_two.x() - point_one.x()
        height = point_two.y() - point_one.y()
        pen = QtGui.QPen()
        pen.setStyle(QtCore.Qt.DashDotLine)
        pen.setWidth(settings.value('square_line_width', settingsdialog.get_default('square_line_width'), int))
        pen.setColor(settings.value('position_square_color', settingsdialog.get_default('position_square_color'), QtGui.QColor))
        brush = QtGui.QBrush()
        brush.setStyle(QtCore.Qt.NoBrush)
        self.pos_marker = QtGui.QGraphicsRectItem(0, 0, width, height)
        self.pos_marker.setPos(point_one)
        self.pos_marker.setBrush(brush)
        self.pos_marker.setPen(pen)
        self.pos_marker.setZValue(.5)
        self.scene.addItem(self.pos_marker)

    def add_copy_marker(self, x, y, width, height):
        settings = QtCore.QSettings("Background corrector")

        pen = QtGui.QPen()
        pen.setStyle(QtCore.Qt.DashDotLine)
        pen.setWidth(settings.value('square_line_width', settingsdialog.get_default('square_line_width'), int))
        pen.setColor(settings.value('copy_square_color', settingsdialog.get_default('copy_square_color'), QtGui.QColor))
        brush = QtGui.QBrush()
        brush.setStyle(QtCore.Qt.NoBrush)
        self.copy_marker = QtGui.QGraphicsRectItem(0, 0, width, height)
        self.copy_marker.setPos(x, y)
        self.copy_marker.setBrush(brush)
        self.copy_marker.setPen(pen)
        self.copy_marker.setZValue(.6)
        self.copy_marker.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        self.scene.addItem(self.copy_marker)

    def remove_pos_marker(self):
        self.scene.removeItem(self.pos_marker)
        self.pos_marker = None

    def remove_copy_marker(self):
        self.scene.removeItem(self.copy_marker)
        self.copy_marker = None

    def cancel_fixing(self):
        if self.copy_marker is not None:
            self.remove_copy_marker()

    def correct_background(self):
        self.prev_img = copy.deepcopy(self.image)
        settings = QtCore.QSettings("Background corrector")

        blur_distance = settings.value('blur_distance', settingsdialog.get_default('blur_distance'), int)

        img_height, img_width, img_depth = self.image.shape
        width = int(self.copy_marker.rect().width())
        height = int(self.copy_marker.rect().height())
        new_x = int(self.pos_marker.scenePos().x())
        new_y = int(self.pos_marker.scenePos().y())
        orig_x = int(self.copy_marker.scenePos().x())
        orig_y = int(self.copy_marker.scenePos().y())

        if new_x < 0 or new_y < 0 or orig_x < 0 or orig_y < 0 or new_x + width > img_width or new_y + height > img_height or orig_x + width > img_width or orig_y + height > img_height:
            return False

        #copy image
        part = self.get_img_part(orig_x, orig_y, width, height)
        self.set_img_part(new_x, new_y, width, height, part)

        #blur edges
        x_blur = max(0, new_x - blur_distance)
        y_blur = max(0, new_y - blur_distance)
        w_blur = min(img_width, width + 2*blur_distance)
        h_blur = min(img_height, height + 2*blur_distance)
        x_keep = min((2*new_x + width)/2, new_x + blur_distance)
        y_keep = min((2*new_y + height)/2, new_y + blur_distance)
        w_keep = max((2*new_x + width)/2, width - 2*blur_distance)
        h_keep = max((2*new_y + height)/2, height - 2*blur_distance)

        to_blur = self.get_img_part(x_blur, y_blur, w_blur, h_blur)
        to_keep = self.get_img_part(x_keep, y_keep, w_keep, h_keep)

        blurred = cv2.blur(to_blur, (5, 5))

        self.set_img_part(x_blur, y_blur, w_blur, h_blur, blurred)
        self.set_img_part(x_keep, y_keep, w_keep, h_keep, to_keep)

        self.update_background()
        return True

    def update_background(self):
        if self.pix_map_item is not None:
            self.scene.removeItem(self.pix_map_item)
            self.pix_map_item = None
        self.pix_map = utils.cvimg2qtpixmap(self.image)
        self.pix_map_item = self.scene.addPixmap(self.pix_map)
        # utils.view_add_bg_image(self.graphics_view, self.pix_map)


    def get_img_part(self, x, y, width, height):
        part = self.image[y: y + height, x: x + width]
        return part

    def set_img_part(self, x, y, width, height, part):
        self.image[y: y + height, x: x + width] = part

    def reset_image(self):
        self.image = copy.deepcopy(self.img_original)
        self.update_background()

    def finish(self):
        self.finish_callback()

    def undo(self):
        self.image = self.prev_img
        self.update_background()