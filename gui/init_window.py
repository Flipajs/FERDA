from gui import ants_init, control_window

__author__ = 'flip'

from PyQt4 import QtGui
import mser_operations
import ant
import utils as my_utils
import lifeCycle
from numpy import *
import cv2
import visualize


class InitWindow(QtGui.QDialog, ants_init.Ui_Dialog):
    def __init__(self, params):
        super(InitWindow, self).__init__()
        self.setupUi(self)

        self.params = params
        self.life_cycle = lifeCycle.LifeCycle(self.params.video_file_name)

        self.ants = []
        self.regions = None
        self.exp = None
        self.img = None

        self.init_ui()
        self.setWindowIcon(QtGui.QIcon('ferda.ico'))

        self.collection_rows = 10
        self.collection_cols = 10
        self.collection_cell_size = 50

        self.id_counter = 0

    def init_ui(self):
        self.img = self.life_cycle.next_img()

        #self.params.arena.center = my_utils.Point(shape(self.img)[0]/2, shape(self.img)[1]/2)
        #self.arena_x_scrollbar.setValue(self.params.arena.center.x)
        #self.arena_y_scrollbar.setValue(self.params.arena.center.y)
        #r = min(self.params.arena.center.x, self.params.arena.center.y)-5

        self.ant_number_spin_box.setValue(self.params.ant_number)
        self.arena_x_scrollbar.setValue(self.params.arena.center.x)
        self.arena_y_scrollbar.setValue(self.params.arena.center.y)

        self.ant_number_spin_box.valueChanged.connect(self.update_changes)
        self.arena_r_scrollbar.valueChanged.connect(self.update_changes)
        self.arena_x_scrollbar.valueChanged.connect(self.update_changes)
        self.arena_y_scrollbar.valueChanged.connect(self.update_changes)

        #after to invoke update_changes...
        self.arena_r_scrollbar.setValue(self.params.arena.size.width/2)

        self.continue_button.clicked.connect(self.init_ants_selection)

        self.start_button.clicked.connect(self.start)

    def init_ants_selection(self):
        cv2.destroyWindow("arena selection")
        self.ants_selection_group.setEnabled(True)
        self.ants_group.setEnabled(False)
        self.arena_group.setEnabled(False)

        img_ = self.img.copy()
        mser_op = mser_operations.MserOperations(self.params)
        self.regions, indexes = mser_op.process_image(img_)

        for i in range(self.params.ant_number):
            a = ant.Ant(i)
            self.ants.append(a)
            s = QtGui.QSpinBox()
            s.setMinimum(0)
            s.setMaximum(len(self.regions))
            self.ants_selection_layout.addWidget(SpinBox(len(self.regions), i, self))

        self.predefined_ant_values()

        img_ = self.img.copy()
        img_ = visualize.draw_region_collection(img_, self.regions, self.params)

        my_utils.imshow("collection", img_)
        cv2.setMouseCallback('collection', self.select_regions_cb)

    def select_regions_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            val = (y / self.collection_cell_size) * self.collection_cols
            val += x / self.collection_cell_size
            self.assign_ant(self.id_counter, val)
            self.id_counter += 1
            if self.id_counter >= self.params.ant_number:
                self.id_counter = 0
            #cv2.circle(img,(x,y),100,(255,0,0),-1)

    def update_changes(self):
        x = self.arena_x_scrollbar.value()
        y = self.arena_y_scrollbar.value()
        r = self.arena_r_scrollbar.value()

        self.label_x.setText(str(x))
        self.label_y.setText(str(y))
        self.label_r.setText(str(r))

        self.params.arena.center = my_utils.Point(x, y)
        self.params.arena.size = my_utils.Size(r * 2, r * 2)
        self.params.ant_number = self.ant_number_spin_box.value()

        img_ = self.img.copy()
        cv2.circle(img_, self.params.arena.center.int_tuple(), r, (255, 255, 0), 1)
        cv2.circle(img_, self.params.arena.center.int_tuple(), 3, (255, 255, 0), -1)

        my_utils.imshow("arena selection", img_, True)

    def assign_ant(self, id, val):
        if val == -1:
            self.ants[id].state.mser_id = -1
            return

        for a in self.ants:
            if a.state.mser_id == val and -1 < val != a.id != id:
                QtGui.QMessageBox.warning(self, 'FAIL', 'MSER id assigned twice', QtGui.QMessageBox.Ok,
                                          QtGui.QMessageBox.Ok)
                self.ants_selection_layout.itemAt(id).widget().setValue(-1)

                return

        ant.set_ant_state(self.ants[id], val, self.regions[val], False)
        self.ants_selection_layout.itemAt(id).widget().setValue(val)
        img_ = self.img.copy()
        img = visualize.draw_ants(img_, self.ants, self.regions, True)

        my_utils.imshow("ants selection", img, True)

    def start(self):
        cv2.destroyWindow("ants selection")
        cv2.destroyWindow("collection")
        self.hide()
        p = control_window.ControlWindow(self.params, self.ants, self.life_cycle)
        p.show()
        p.exec_()

    def predefined_ant_values(self):
        if self.params.predefined_vals == 'NoPlasterNoLid800':
            arr = [2, 5, 7, 9, 13, 16, 21, 24, 28, 31, 33, 36, 39, 43, 46]
        elif self.params.predefined_vals == 'eight':
            arr = [0, 4, 7, 11, 16, 21, 26, 29]
        elif self.params.predefined_vals == 'Camera2':
            arr = [1, 4, 8, 14, 19, 25, 30, 37, 43, 23, 35]
        else:
            return

        for i in range(self.params.ant_number):
            self.assign_ant(i, arr[i])

class SpinBox(QtGui.QSpinBox):
    def __init__(self, maxVal, id, window_p, parent=None):
        super(SpinBox, self).__init__(parent)
        self.setMinimum(-1)
        self.setValue(-1)
        self.setMaximum(maxVal)
        self.editingFinished.connect(self.assign_ant)
        self.window_p = window_p
        self.id = id

    def assign_ant(self):
        self.window_p.assign_ant(self.id, self.value())