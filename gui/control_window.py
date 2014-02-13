from gui import ants_view

__author__ = 'flip'

import sys
from PyQt4 import QtGui
import experiment_params
import cv2
import experiment_manager
import scipy.io as sio


class ControlWindow(QtGui.QDialog, ants_view.Ui_Dialog):
    def __init__(self, params, ants, life_cycle):
        super(ControlWindow, self).__init__()
        self.setupUi(self)

        self.life_cycle = life_cycle
        self.mser_operations = None
        self.params = params
        self.is_running = False
        self.experiment = experiment_manager.ExperimentManager(self.params, ants)
        self.out_directory = ""
        self.wait_for_button_press = False
        self.frame = 1

        self.init_ui()
        self.setWindowIcon(QtGui.QIcon('ferda.ico'))

    def closeEvent(self, event):
        self.is_running = False
        cv2.destroyAllWindows()

        sys.exit(1)

    def init_ui(self):
        self.b_play.clicked.connect(self.play)
        self.b_stap_by_step.clicked.connect(self.step_by_step)
        self.b_choose_path.clicked.connect(self.show_file_dialog)
        self.b_save_file.clicked.connect(self.save_data)
        self.ch_ants_collection.clicked.connect(self.show_ants_collection_changed)
        self.ch_mser_collection.clicked.connect(self.show_mser_collection_changed)
        self.show()

    def step_by_step(self):
        self.is_running = True
        self.wait_for_button_press = True
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
                img = self.life_cycle.next_img()
                if img == None:
                    self.b_play.setText('play')
                    return

                self.frame += 1
                print "FRAME> ", self.frame

                self.l_frame.setText(str(self.frame))
                self.experiment.process_frame(img, self.wait_for_button_press)


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

def main():
    app = QtGui.QApplication(sys.argv)
    ex = ControlWindow(experiment_params.Params())

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()