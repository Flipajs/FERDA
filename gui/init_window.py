from gui import ants_init, control_window

__author__ = 'flip'

from PyQt4 import QtGui
import mser_operations
import ant
import my_utils
import video_manager
import numpy as np
import cv
import cv2
import visualize
import experiment_params
import os
import scipy.ndimage


class InitWindow(QtGui.QDialog, ants_init.Ui_Dialog):
    def __init__(self):
        super(InitWindow, self).__init__()
        self.setupUi(self)

        self.params = experiment_params.Params()
        self.video_manager = None

        self.ants = []
        self.regions = None
        self.chosen_regions_indexes = None
        self.exp = None
        self.img = None
        self.bg = None

        self.init_ui()
        self.setWindowIcon(QtGui.QIcon('imgs/ferda.ico'))
        self.window().setGeometry(0, 0, self.window().width(), self.window().height())

        self.collection_rows = 10
        self.collection_cols = 10
        self.collection_cell_size = 50

        self.actual_focus = -1

        self.show()

        if self.params.fast_start:
            self.load_video()
            self.init_ants_selection()
            self.start()

        self.exec_()

    def init_ui(self):
        #self.img = self.life_cycle.next_img()
        if self.params.fast_start:
            self.ant_number_spin_box.setValue(self.params.ant_number)
            self.arena_x_scrollbar.setValue(self.params.arena.center.x)
            self.arena_y_scrollbar.setValue(self.params.arena.center.y)

        self.ant_number_spin_box.valueChanged.connect(self.update_ant_number)
        self.arena_r_scrollbar.valueChanged.connect(self.update_arena_changes)
        self.arena_x_scrollbar.valueChanged.connect(self.update_arena_changes)
        self.arena_y_scrollbar.valueChanged.connect(self.update_arena_changes)

        self.b_choose_video.clicked.connect(self.show_file_dialog)
        self.b_load_video.clicked.connect(self.load_video)
        self.ch_invert_image.clicked.connect(self.invert_image)
        self.b_count_bg_model.clicked.connect(self.count_bg_model)
        self.b_use_model.clicked.connect(self.use_model)

        self.b_video_continue.clicked.connect(self.video_continue)

        self.b_continue_arena.clicked.connect(self.continue_arena)
        self.b_continue_ants.clicked.connect(self.continue_ants)
        #after to invoke update_changes...
        if self.params.fast_start:
            self.arena_r_scrollbar.setValue(self.params.arena.size.width/2)

        self.start_button.clicked.connect(self.start)

    def show_file_dialog(self):
        self.params.video_file_name = str(QtGui.QFileDialog.getOpenFileName(self, "Select video file"))
        drive, path = os.path.splitdrive(self.params.video_file_name)
        path, filename = os.path.split(path)
        self.file_name_label.setText(filename)
        self.b_load_video.setEnabled(True)
        self.b_load_video.setFocus()

    def load_video(self):
        self.video_manager = video_manager.VideoManager(self.params.video_file_name)
        self.img = self.video_manager.next_img()

        if self.img is not None:
            self.b_choose_video.setDisabled(True)
            self.b_load_video.setDisabled(True)
            self.ch_invert_image.setEnabled(True)
            self.b_video_continue.setEnabled(True)
            self.b_count_bg_model.setEnabled(True)

            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("preview", gray)

            self.b_video_continue.setFocus()

    def count_bg_model(self):
        self.b_count_bg_model.setText('processing... 0 / 100')
        bg = None
        i = -1
        while True:
            i += 1
            img = self.video_manager.next_img()

            if img is None:
                break

            if i > 1500:
                break

            if i % 50 != 0:
                continue

            if bg is not None:
                self.b_count_bg_model.setText('processing... '+str(i/15)+' / 100')
                bg = np.maximum(bg, img)
                cv2.imshow("bg model", bg)
                cv2.waitKey(20)
            else:
                bg = img

        self.video_manager = video_manager.VideoManager(self.params.video_file_name)
        self.video_manager.next_img()
        bg = scipy.ndimage.gaussian_filter(bg, sigma=1)
        self.bg = bg


        self.b_count_bg_model.setText('done')

        self.b_use_model.setEnabled(True)
        return

    def use_model(self):
        self.params.bg = self.bg
        self.video_continue()

    def invert_image(self):
        self.params.inverted_image = self.ch_invert_image.isChecked()
        img = self.img.copy()

        if self.params.inverted_image:
            img = np.invert(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("preview", gray)

    def video_continue(self):
        cv2.destroyAllWindows()
        self.video_group.setDisabled(True)
        self.arena_group.setEnabled(True)

        my_utils.imshow("arena selection", self.img, True)
        x = self.window().width() + self.window().x() + 1
        cv.MoveWindow("arena selection", x, 0)

        self.update_arena_changes()
        self.tabWidget.setCurrentIndex(1)
        self.b_continue_arena.setFocus()

    def continue_arena(self):
        cv2.destroyWindow("arena selection")
        self.arena_group.setDisabled(True)
        self.ants_group.setEnabled(True)
        my_utils.imshow("1st frame", self.img, True)
        x = self.window().width() + self.window().x() + 1
        cv.MoveWindow("1st frame", x, 0)
        self.tabWidget.setCurrentIndex(2)
        self.ant_number_spin_box.setFocus()

    def continue_ants(self):
        self.ants_group.setDisabled(True)
        self.ants_selection_group.setEnabled(True)
        self.init_ants_selection()

    def init_ants_selection(self):
        img_ = self.img.copy()

        mask = my_utils.prepare_image(img_, self.params)
        #cv2.imshow("AAA", mask)

        mser_op = mser_operations.MserOperations(self.params)
        self.regions, self.chosen_regions_indexes = mser_op.process_image(mask)
        groups = mser_operations.get_region_groups(self.regions)

        for i in range(self.params.ant_number):
            a = ant.Ant(i)
            self.ants.append(a)
            s = QtGui.QSpinBox()
            s.setMinimum(0)
            s.setMaximum(len(self.regions))
            self.ants_selection_layout.addWidget(PushButton(i, self))

        self.predefined_ant_values()

        img_ = self.img.copy()
        img_ = visualize.draw_region_best_margins_collection(img_, self.regions, self.chosen_regions_indexes, self.ants, cols=self.collection_cols, cell_size=self.collection_cell_size)

        my_utils.imshow("collection", img_)
        cv2.setMouseCallback('collection', self.select_regions_cb)

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

            my_utils.imshow("collection", img_)
            cv2.setMouseCallback('collection', self.select_regions_cb)

    def update_arena_changes(self):
        x = self.arena_x_scrollbar.value()
        y = self.arena_y_scrollbar.value()
        r = self.arena_r_scrollbar.value()

        self.label_x.setText(str(x))
        self.label_y.setText(str(y))
        self.label_r.setText(str(r))

        self.params.arena.center = my_utils.Point(x, y)
        self.params.arena.size = my_utils.Size(r * 2, r * 2)

        if not self.params.fast_start:
            img_ = self.img.copy()
            cv2.circle(img_, self.params.arena.center.int_tuple(), r, (255, 255, 0), 1)
            cv2.circle(img_, self.params.arena.center.int_tuple(), 3, (255, 255, 0), -1)

            img = my_utils.mask_out_arena(img_, self.params.arena)
            my_utils.imshow("arena selection", img, 1)

    def update_ant_number(self):
        self.params.ant_number = self.ant_number_spin_box.value()

    def assign_ant(self, id, val):
        if val == -1:
            self.ants[id].state.mser_id = -1
            return

        #for a in self.ants:
        #    if a.state.mser_id == val and -1 < val != a.id != id:
        #        QtGui.QMessageBox.warning(self, 'FAIL', 'MSER id assigned twice', QtGui.QMessageBox.Ok,
        #                                  QtGui.QMessageBox.Ok)
        #        self.ants_selection_layout.itemAt(id).widget().setText(str(-1))
        #
        #        return

        ant.set_ant_state(self.ants[id], val, self.regions[val], False)
        self.ants_selection_layout.itemAt(id).widget().setText(str(val))
        #self.ants_selection_layout.itemAt(id).
        img_ = self.img.copy()
        img = visualize.draw_ants(img_, self.ants, self.regions, True)
        my_utils.imshow("1st frame", img, True)
        cv2.waitKey(1)

        img_ = self.img.copy()
        img_ = visualize.draw_region_best_margins_collection(img_, self.regions, self.chosen_regions_indexes, self.ants, cell_size = self.collection_cell_size)

        my_utils.imshow("collection", img_)

    def start(self):
        #if self.params.fast_start:
        cv2.destroyAllWindows()

        #cv2.destroyWindow("1st frame")
        #cv2.destroyWindow("collection")
        self.hide()
        p = control_window.ControlWindow(self.params, self.ants, self.video_manager)
        p.show()
        p.exec_()

    def predefined_ant_values(self):
        if self.params.fast_start:
            self.auto_assignment()
            return

        if self.params.predefined_vals == 'NoPlasterNoLid800':
            arr = [1, 8, 15, 20, 26, 31, 34, 37, 42, 48, 55, 59, 63, 67, 73]
        elif self.params.predefined_vals == 'eight':
            arr = [2, 11, 17, 25, 30, 38, 46, 53]
        elif self.params.predefined_vals == 'Camera2':
            arr = [1, 4, 8, 14, 19, 25, 30, 37, 43, 23, 35]
        elif self.params.predefined_vals == 'messor1':
            arr = [2, 6, 11, 16, 20]
        else:
            self.auto_assignment()
            return

        for i in range(self.params.ant_number):
            self.assign_ant(i, arr[i])

class PushButton(QtGui.QPushButton):
    def __init__(self, id, window_p, parent=None):
        super(PushButton, self).__init__(parent)
        self.clicked.connect(self.set_actual_focus)
        self.window_p = window_p
        self.setText(str(-1))
        self.id = id
        c = self.window_p.ants[id].color
        s = c[0]+c[1]+c[2]
        text_c = 'rgb(0, 0, 0)'
        if s < (255*3) / 2:
            text_c = 'rgb(255, 255, 255)'

        self.setStyleSheet("QPushButton { background-color: rgb("+str(c[2])+", "+str(c[1])+", "+str(c[0])+"); color: "+text_c+";}")

    def set_actual_focus(self):
        self.window_p.actual_focus = self.id

class SpinBox(QtGui.QSpinBox):
    def __init__(self, maxVal, id, window_p, parent=None):
        super(SpinBox, self).__init__(parent)
        self.setMinimum(-1)
        self.setValue(-1)
        self.setMaximum(maxVal)
        self.editingFinished.connect(self.assign_ant)
        self.window_p = window_p
        self.id = id
        c = self.window_p.ants[id].color
        s = c[0]+c[1]+c[2]
        text_c = 'rgb(0, 0, 0)'
        if s < (255*3) / 2:
            text_c = 'rgb(255, 255, 255)'

        self.setStyleSheet("QSpinBox { background-color: rgb("+str(c[2])+", "+str(c[1])+", "+str(c[0])+"); color: "+text_c+";}")

    def assign_ant(self):
        self.window_p.assign_ant(self.id, self.value())