__author__ = 'flip'

from utils import video_manager
import utils.misc
from gui import control_window
from gui.init_window import ants_init
from PyQt4 import QtCore
import mser_operations
import ant
import my_utils
import numpy as np
import cv2
import visualize
import experiment_params
import os
import scipy.ndimage
from gui.init_window import pixmap_label
from push_button import *
from spin_box import *
from gui.img_controls import my_view, utils
from arena_mark import *
from arena_circle import *
from viewer.background_corrector import background_corrector_core
from utils import misc

import ImageQt


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

class InitWindow(QtGui.QWidget, ants_init.Ui_Dialog):
    def __init__(self, main_window):
        super(InitWindow, self).__init__()
        self.main_window = main_window
        self.setupUi(self)

        self.params = experiment_params.Params()

        self.video_manager = None
        self.ants = []
        self.regions = None
        self.chosen_regions_indexes = None
        self.exp = None
        self.img = None
        self.bg = None

        self.bg_img_item = None

        self.init_ui()
        # self.setWindowIcon(QtGui.QIcon('../imgs/ferda.ico'))

        self.collection_rows = 10
        self.collection_cols = 10
        self.collection_cell_size = 50

        self.actual_focus = -1

        self.graphics_view = my_view.MyView(self)
        self.set_graphics_view_geometry()

        self.graphics_view.setObjectName(_fromUtf8("graphics_view"))
        self.scene = QtGui.QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        self.c_center = None
        self.c_radius = None


        self.show()

        self.scene_objects = {}

        self.init_mser_preview = None
        self.ant_assignment = {}

        self.window().show()
        self.window().setGeometry(0, 0, self.window().width(), self.window().height())

        self.close_callback = None

        self.b_model_fix_tool.setEnabled(True)

        self.spin_bg_num_steps.setValue(1)
        self.spin_bg_step_length.setValue(10)
        self.dialog = None

        if self.params.fast_start:
            self.load_video()
            self.count_bg_model()
            self.ant_number_spin_box.setValue(self.params.ant_number)
            self.continue_ants()
            # self.start()

    def set_close_callback(self, callback):
        self.close_callback = callback

    def init_ui(self):
        self.b_choose_video.clicked.connect(self.show_file_dialog)
        self.b_load_video.clicked.connect(self.load_video)
        self.ch_invert_image.clicked.connect(self.invert_image)
        self.b_count_bg_model.clicked.connect(self.count_bg_model)
        self.b_use_model.clicked.connect(self.use_model)

        self.b_model_fix_tool.clicked.connect(self.show_bg_fix)

        self.b_video_continue.clicked.connect(self.without_model)

        self.b_continue_arena.clicked.connect(self.continue_arena)
        self.b_continue_ants.clicked.connect(self.continue_ants)

        self.start_button.clicked.connect(self.start)

    def show_file_dialog(self):
        file_names = QtGui.QFileDialog.getOpenFileNames(self, "Select video file")

        names_str = ""

        self.params.video_file_name = []
        for f in file_names:
            self.params.video_file_name.append(str(f))
            _, path = os.path.splitdrive(str(f))
            _, name = os.path.split(path)

            names_str += name+" "

        self.file_name_label.setText(names_str)
        self.b_load_video.setEnabled(True)
        self.b_load_video.setFocus()

    def show_bg_fix(self):
        self.dialog = background_corrector_core.BackgroundCorrector(self.bg, self.bg_fix_finish_callback)
        self.main_window.central_widget.addWidget(self.dialog)
        self.main_window.central_widget.setCurrentWidget(self.dialog)

    def bg_fix_finish_callback(self):
        self.bg = self.dialog.image
        self.main_window.central_widget.removeWidget(self.dialog)
        self.dialog = None

        self.pixmap_bg = utils.cvimg2qtpixmap(self.bg)

        if self.bg_img_item is not None:
            self.scene.removeItem(self.bg_img_item)

        self.bg_img_item = self.scene.addPixmap(self.pixmap_bg)

    def load_video(self):

        self.video_manager = video_manager.get_auto_video_manager(self.params.video_file_name)
        self.img = self.video_manager.move2_next()

        if self.img is not None:
            self.b_choose_video.setDisabled(True)
            self.b_load_video.setDisabled(True)
            self.ch_invert_image.setEnabled(True)
            self.b_video_continue.setEnabled(True)
            self.b_count_bg_model.setEnabled(True)

            self.pixmap = utils.cvimg2qtpixmap(self.img)
            self.video_img_item = self.scene.addPixmap(self.pixmap)

            utils.view_add_bg_image(self.graphics_view, self.pixmap)

            #gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("preview", gray)

            self.b_video_continue.setFocus()

    def count_bg_model(self):
        gui = QtGui.QApplication.processEvents

        self.b_count_bg_model.setText('processing... 0 / 100')
        bg = None
        i = 0

        step = self.spin_bg_step_length.value()
        max_frame = step * self.spin_bg_num_steps.value()

        #we can try to use frame seek
        try:
            while True:
                img = self.video_manager.seek_frame(i)

                if self.params.inverted_image:
                    img = np.invert(img)

                if img is None:
                    break

                if i > max_frame:
                    break

                if bg is not None:
                    self.b_count_bg_model.setText('processing... '+str(int(i/(max_frame/100.)))+' / 100')
                    bg = np.maximum(bg, img)

                    self.pixmap_bg = utils.cvimg2qtpixmap(bg)

                    if self.bg_img_item is not None:
                        self.scene.removeItem(self.bg_img_item)

                    self.bg_img_item = self.scene.addPixmap(self.pixmap_bg)
                    gui()
                else:
                    bg = img

                i += step

        except Exception as e:
            utils.misc.print_exception(e)

            i = -1
            while True:
                i += 1
                img = self.video_manager.move2_next()

                if self.params.inverted_image:
                    img = np.invert(img)

                if img is None:
                    break

                if i > max_frame:
                    break

                if i % step != 0:
                    continue

                if bg is not None:
                    self.b_count_bg_model.setText('processing... '+str(int(i/(max_frame/100.)))+' / 100')
                    bg = np.maximum(bg, img)

                    self.pixmap_bg = utils.cvimg2qtpixmap(bg)

                    if self.bg_img_item is not None:
                        self.scene.removeItem(self.bg_img_item)

                    self.bg_img_item = self.scene.addPixmap(self.pixmap_bg)
                    gui()
                else:
                    bg = img

        self.video_manager.reset()
        self.video_manager.move2_next()
        bg = scipy.ndimage.gaussian_filter(bg, sigma=1)
        self.bg = bg

        self.pixmap_bg = utils.cvimg2qtpixmap(bg)

        if self.bg_img_item is not None:
            self.scene.removeItem(self.bg_img_item)

        self.bg_img_item = self.scene.addPixmap(self.pixmap_bg)

        self.b_count_bg_model.setText('done')

        self.b_use_model.setEnabled(True)
        self.b_model_fix_tool.setEnabled(True)
        return

    def use_model(self):
        self.params.bg = self.bg
        self.video_continue()

    def without_model(self):
        self.params.bg = None
        self.video_continue()

    def invert_image(self):
        self.params.inverted_image = self.ch_invert_image.isChecked()
        img = self.img.copy()

        if self.params.inverted_image:
            img = np.invert(img)

        self.scene.removeItem(self.video_img_item)
        self.scene.update()
        self.graphics_view.update()
        self.pixmap = utils.cvimg2qtpixmap(img)
        self.video_img_item = self.scene.addPixmap(self.pixmap)
        utils.view_add_bg_image(self.graphics_view, self.pixmap)
        self.scene.update()

    def video_continue(self):
        if self.bg_img_item is not None:
            self.scene.removeItem(self.bg_img_item)

        self.video_group.setDisabled(True)
        self.arena_group.setEnabled(True)

        max_dim = max(self.img.shape[0], self.img.shape[1])
        mark_size = max_dim / 50

        self.arena_ellipse = ArenaCircle()

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtGui.QColor(0xff,0,0,0xaa))
        self.c_center = ArenaMark(self.arena_ellipse, self.update_circle_labels)
        self.c_center.setRect(0, 0, mark_size, mark_size)
        self.c_center.setBrush(brush)
        self.c_center.setPos(self.img.shape[1]/2, self.img.shape[0]/2)

        brush.setColor(QtGui.QColor(0,0,0xff,0xaa))
        self.c_radius = ArenaMark(self.arena_ellipse, self.update_circle_labels)
        self.c_radius.setRect(0, 0, mark_size, mark_size)
        self.c_radius.setBrush(brush)
        self.c_radius.setPos(self.img.shape[1]/2 + 200, self.img.shape[0]/2)

        brush.setColor(QtGui.QColor(0, 0xFF, 0, 0x55))
        self.arena_ellipse.setBrush(brush)


        self.arena_ellipse.add_points(self.c_center, self.c_radius)
        self.arena_ellipse.update_geometry()
        self.scene_objects['arena_ellipse'] = self.scene.addItem(self.arena_ellipse)
        self.scene_objects['c_center'] = self.scene.addItem(self.c_center)
        self.scene_objects['c_radius'] = self.scene.addItem(self.c_radius)

        self.tabWidget.setCurrentIndex(1)
        self.b_continue_arena.setFocus()

        self.tableWidget.insertRow(self.tableWidget.rowCount())

        self.tableWidget.setItem(0, 0, QtGui.QTableWidgetItem('center'))
        self.tableWidget.setItem(0, 1, QtGui.QTableWidgetItem('1000'))
        self.tableWidget.setItem(0, 2, QtGui.QTableWidgetItem('1000'))
        self.tableWidget.insertRow(self.tableWidget.rowCount())
        self.tableWidget.setItem(1, 0, QtGui.QTableWidgetItem('radius'))
        self.tableWidget.setItem(1, 1, QtGui.QTableWidgetItem('0'))
        self.tableWidget.setItem(1, 2, QtGui.QTableWidgetItem('0'))

        self.tableWidget.resizeColumnsToContents()

        self.update_circle_labels()

    def update_circle_labels(self):
        item = lambda x: QtGui.QTableWidgetItem(QtCore.QString.number(int(x)))

        self.tableWidget.setItem(0, 1, item(self.c_center.pos().x()))
        self.tableWidget.setItem(0, 2, item(self.c_center.pos().y()))

        self.tableWidget.setItem(1, 1, item(self.c_radius.pos().x()))
        self.tableWidget.setItem(1, 2, item(self.c_radius.pos().y()))

    def continue_arena(self):
        self.arena_group.setDisabled(True)
        self.ants_group.setEnabled(True)
        self.set_arena_parameters()
        self.tabWidget.setCurrentIndex(2)
        self.ant_number_spin_box.setFocus()
        if not self.params.fast_start:
            self.remove_arena_marks()


    def set_arena_parameters(self):
        c_width_2 = self.c_center.rect().width() / 2
        x = int(self.c_center.x() + c_width_2)
        y = int(self.c_center.y() + c_width_2)

        r_width_w2 = self.c_radius.rect().width() / 2
        rx = int(self.c_radius.x() + r_width_w2)
        ry = int(self.c_radius.y() + r_width_w2)

        r = int(math.sqrt(abs(x-rx)**2 + (abs(y-ry))**2))
        self.params.arena.center = my_utils.Point(x, y)
        self.params.arena.size = my_utils.Size(r * 2, r * 2)


    def remove_arena_marks(self):
        self.scene.removeItem(self.arena_ellipse)
        self.scene.removeItem(self.c_radius)
        self.scene.removeItem(self.c_center)


    def continue_ants(self):
        self.params.ant_number = self.ant_number_spin_box.value()
        self.ants_group.setDisabled(True)
        self.tab_object_selection.setEnabled(True)
        self.tabWidget.setCurrentIndex(3)
        self.init_ants_selection()

    def init_ants_selection(self):
        img_ = self.img.copy()

        mask = my_utils.prepare_image(img_, self.params)

        mser_op = mser_operations.MserOperations(self.params)
        self.regions, self.chosen_regions_indexes = mser_op.process_image(mask)
        groups = mser_operations.get_region_groups(self.regions)

        for i in range(self.params.ant_number):
            a = ant.Ant(i)
            a.color = ant.get_color(i, self.params.ant_number)
            self.ants.append(a)
            s = QtGui.QSpinBox()
            s.setMinimum(0)
            s.setMaximum(len(self.regions))
            self.object_init_layout.addWidget(PushButton(i, self))
            #self.object_init_layout.itemAt(i).widget().setText(str(i))


        for a in self.ants:
            self.ant_assignment[a.id] = []


        self.pixmaps = []

        col_num = 4

        i = 0

        for r_id in self.chosen_regions_indexes:
            r = self.regions[r_id]

            my_label = pixmap_label.PixmapLabel(self, self.img, r, r_id, self.mser_init_update_graphics_view, self.ant_assignment)

            c = i%col_num
            r = i/col_num
            self.init_mser_grid.addWidget(my_label, r, c)

            i += 1

        self.predefined_ant_values()

    def mser_init_update_graphics_view(self):
        if self.init_mser_preview:
            self.scene.removeItem(self.init_mser_preview)


        img_ = self.img.copy()
        img = visualize.draw_ants(img_, self.ants, self.regions, True)

        img_q = ImageQt.QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, 13)
        pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

        self.init_mser_preview = self.scene.addPixmap(pix_map)

    def auto_assignment(self):
        areas = np.zeros(len(self.chosen_regions_indexes))
        main_axis = np.zeros(len(self.chosen_regions_indexes))

        for i in range(len(self.chosen_regions_indexes)):
            r = self.regions[self.chosen_regions_indexes[i]]
            areas[i] = r['area']
            _, a, _ = my_utils.mser_main_axis_ratio(r['sxy'], r['sxx'], r['syy'])
            main_axis[i] = a

        areas_med = np.median(areas)
        main_axis_med = np.median(main_axis)

        dists = np.zeros(len(self.chosen_regions_indexes))

        for i in range(len(self.chosen_regions_indexes)):
            area_val = abs(areas[i] - areas_med) / float(areas_med)
            a_val = abs(main_axis[i] - main_axis_med) / float(main_axis_med)

            dists[i] = area_val + a_val

        ids = np.argsort(dists)[0:self.params.ant_number]

        counter = 0
        for id in ids:
            self.assign_ant(counter, self.chosen_regions_indexes[id])
            counter += 1


    def select_regions_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            val = (y / self.collection_cell_size) * self.collection_cols
            val += x / self.collection_cell_size

            if self.actual_focus >= 0:
                self.assign_ant(self.actual_focus, self.chosen_regions_indexes[val])

            #self.id_counter += 1
            #if self.id_counter >= self.params.ant_number:
            #    self.id_counter = 0


            img_ = self.img.copy()
            img_ = visualize.draw_region_best_margins_collection(img_, self.regions, self.chosen_regions_indexes, self.ants, self.collection_cols, cell_size=self.collection_cell_size)

            #my_utils.imshow("collection", img_)
            #cv2.setMouseCallback('collection', self.select_regions_cb)


    def assign_ant(self, id, val):
        if val == -1:
            self.ants[id].state.mser_id = -1
            return

        ant.set_ant_state(self.ants[id], val, self.regions[val], False, 2)
        self.object_init_layout.itemAt(id).widget().setText(str(val))
        self.mser_init_update_graphics_view()


    def set_all_ants(self):
        for i in range(len(self.ants)):
            r_id = int(self.object_init_layout.itemAt(i).widget().text())
            self.assign_ant(i, r_id)

    def start(self):
        #self.set_all_ants()
        self.hide()
        if self.close_callback is None:
            p = control_window.ControlWindow(self.params, self.ants, self.video_manager)
            # p.exec_()
            print "INIT GOINT TO CLOSE"
            self.close()
        else:
            self.close_callback()

    def predefined_ant_values(self):
        if not self.params.fast_start:
            return
        # if self.params.fast_start:
        #     self.auto_assignment()
        #     return

        if self.params.predefined_vals == 'NoPlasterNoLid800':
            # arr = [1, 8, 15, 20, 26, 31, 34, 37, 42, 48, 55, 59, 63, 67, 73]
            arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        elif self.params.predefined_vals == 'eight':
            arr = [2, 11, 17, 25, 30, 38, 46, 53]
        elif self.params.predefined_vals == 'Camera2':
            arr = [1, 4, 8, 14, 19, 25, 30, 37, 43, 23, 35]
        elif self.params.predefined_vals == 'messor1':
            arr = [2, 6, 11, 16, 20]
        else:
            return
            # self.auto_assignment()
        #return

        for i in range(self.params.ant_number):
            self.actual_focus=i
            self.assign_ant(i, arr[i])
            self.init_mser_grid.itemAt(i).widget().add_ant()

        actual_focus = -1

    def resizeEvent(self, QResizeEvent):
        self.set_graphics_view_geometry()

    def set_graphics_view_geometry(self):
        tab_w = self.tabWidget.geometry().width()
        w = self.geometry().width()
        h = self.geometry().height()
        top_margin = 30
        self.graphics_view.setGeometry(tab_w + 5, top_margin, w - tab_w - 5 - 1, h - top_margin - 1)


