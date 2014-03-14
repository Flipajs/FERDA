from gui import ants_view

__author__ = 'flip'

import sys
from PyQt4 import QtGui
import experiment_params
import cv2
import experiment_manager
import scipy.io as sio
import video_manager


class ControlWindow(QtGui.QDialog, ants_view.Ui_Dialog):
    def __init__(self, params, ants, video_manager):
        super(ControlWindow, self).__init__()
        self.setupUi(self)

        self.video_manager = video_manager
        self.mser_operations = None
        self.params = params
        self.is_running = False
        self.experiment = experiment_manager.ExperimentManager(self.params, ants)
        self.out_directory = ""
        self.wait_for_button_press = False
        self.forward = True

        self.window().setGeometry(0, 0, self.window().width(), self.window().height())
        self.init_ui()
        self.setWindowIcon(QtGui.QIcon('ferda.ico'))

    def closeEvent(self, event):
        self.is_running = False
        cv2.destroyAllWindows()

        sys.exit(1)

    def init_ui(self):
        self.b_play.clicked.connect(self.play)
        self.b_forwards.clicked.connect(self.step_forwards)
        self.b_backwards.clicked.connect(self.step_backwards)
        self.b_choose_path.clicked.connect(self.show_file_dialog)
        self.b_save_file.clicked.connect(self.save_data)
        self.ch_ants_collection.clicked.connect(self.show_ants_collection_changed)
        self.ch_mser_collection.clicked.connect(self.show_mser_collection_changed)
        self.show()

        self.b_log_save_regions.clicked.connect(self.log_save_regions)
        self.b_log_save_regions_collection.clicked.connect(self.log_save_regions_collection)
        self.b_log_save_frame.clicked.connect(self.log_save_frame)

        self.ch_ants_collection.setChecked(self.params.show_ants_collection)
        self.ch_mser_collection.setChecked(self.params.show_mser_collection)

        self.imshow_decreasing_factor.valueChanged.connect(self.imshow_decreasing_factor_changed)

    def step_forwards(self):
        self.is_running = True
        self.wait_for_button_press = True
        self.forward = True
        self.run()

    def step_backwards(self):
        self.is_running = True
        self.wait_for_button_press = True
        self.forward = False
        self.run()

    def play(self):
        self.wait_for_button_press = False

        if self.b_play.text() == 'play':
            self.is_running = True
            self.b_play.setText('pause')
        else:
            self.is_running = False
            self.b_play.setText('play')

        self.run()

    def run(self):
        if self.is_running:
            while self.is_running:
                val = self.sb_stop_at_frame.value()
                if self.experiment.params.frame >= val and val != 0:
                    self.b_play.setText('play')
                    self.is_running = False
                    self.controls()
                    return

                if self.forward:
                    img = self.video_manager.next_img()
                else:
                    if self.params.frame == 1:
                        print "This is frame #1, there is no more previous frames."
                        self.controls()
                        self.run()
                        return

                    img = self.video_manager.prev_img()

                if img is None:
                    self.b_play.setText('play')
                    if self.forward:
                        print "End of video file!"
                    else:
                        print "End of history buffer!"
                        self.controls()
                        self.run()
                    return

                self.experiment.process_frame(img, self.forward)
                print "------------------------"

                self.l_frame.setText(str(self.params.frame))
                self.controls()

    def controls(self):
        if self.wait_for_button_press:
            while True:
                k = cv2.waitKey(0)
                if k % 128 == 32 or k % 128 == 83:
                    self.forward = True
                    break
                elif k % 128 == 81:
                    self.forward = False
                    break
        else:
            cv2.waitKey(5)

    def show_file_dialog(self):
        self.out_directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select directory"))

    def show_ants_collection_changed(self):
        self.experiment.params.show_ants_collection = self.ch_ants_collection.isChecked()

    def show_mser_collection_changed(self):
        self.experiment.params.show_mser_collection = self.ch_mser_collection.isChecked()

    def save_data(self):
        path = self.out_directory

        data = self.experiment.ants_history_data()
        if len(path) > 0:
            path += '/'

        name = str(self.i_file_name.text())

        if len(name) == 0:
            name = 'undefined'

        path += name

        sio.savemat(path, mdict={'antrack': data}, oned_as='row')

    def imshow_decreasing_factor_changed(self):
        self.params.imshow_decreasing_factor = self.imshow_decreasing_factor.value()

    def log_save_regions(self):
        self.experiment.log_regions()

    def log_save_regions_collection(self):
        self.experiment.log_regions_collection()

    def log_save_frame(self):
        self.experiment.log_frame()


def main():
    app = QtGui.QApplication(sys.argv)
    ex = ControlWindow(experiment_params.Params())

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()