from gui import ants_view

__author__ = 'flip'

import sys
from PyQt4 import QtGui, QtCore
import experiment_params
import cv2
import cv
import experiment_manager
import scipy.io as sio
import ntpath
import pickle
import video_manager
import logger
import collisions
import time
import visualize


class ControlWindow(QtGui.QMainWindow, ants_view.Ui_Dialog):
    def __init__(self, params, ants, video_manager):
        super(ControlWindow, self).__init__()
        self.setupUi(self)

        self.video_manager = video_manager
        self.params = params
        self.is_running = False
        self.experiment = experiment_manager.ExperimentManager(self.params, ants, video_manager)
        self.logger = logger.Logger(self.experiment)
        self.out_directory = ""
        self.out_state_directory = "out/states"
        self.wait_for_button_press = False
        self.forward = True

        self.window().setGeometry(0, 0, self.window().width(), self.window().height())
        self.init_ui()
        self.setWindowIcon(QtGui.QIcon('imgs/ferda.ico'))


        self.results2video = True

        if self.results2video:
            width = 1920
            height = 1080
            self.vid_writer = cv2.VideoWriter(filename="ferda_output.avi",  #Provide a file to write the video to
                #fourcc=cv.CV_FOURCC('i','Y', 'U', 'V'),            #Use whichever codec works for you...
                fourcc=cv2.cv.CV_FOURCC('M','J','P','G'),
                fps=30,                                        #How many frames do you want to display per second in your video?
                frameSize=(width, height))

        self.settings = QtCore.QSettings("FERDA")
        auto_run = self.settings.value(
            'auto_run', self.params.auto_run, bool
        )

        if auto_run:
            self.is_running = True
            self.play()

    #def __delete__(self, instance):
        #cv2.destroyAllWindows()

    def closeEvent(self, event):
        self.is_running = False
        #cv2.dest
        #event.accept()

        #if self.is_running:
        #    self.is_running = False
        #    #cv2.destroyAllWindows()
        #
        #    self.closeEvent(event)
        #else:
        #    event.accept()
        #cv2.destroyAllWindows()


        #sys.exit(0)

    def init_ui(self):
        self.i_state_name.setText(ntpath.basename(self.params.video_file_name))

        self.b_play.clicked.connect(self.play)
        self.b_forwards.clicked.connect(self.step_forwards)
        self.b_backwards.clicked.connect(self.step_backwards)
        self.b_choose_path.clicked.connect(self.show_file_dialog)
        self.b_save_mat.clicked.connect(self.save_data)
        self.b_choose_path_state.clicked.connect(self.show_file_dialog_state)
        self.b_save_state.clicked.connect(self.save_state)
        self.ch_ants_collection.clicked.connect(self.show_ants_collection_changed)
        self.ch_mser_collection.clicked.connect(self.show_mser_collection_changed)
        self.ch_print_mser_info.clicked.connect(self.print_mser_info_changed)
        self.ch_imshow.clicked.connect(self.imshow)
        self.ch_assignment_problem.clicked.connect(self.show_assignment_problem)
        self.b_load_state.clicked.connect(self.load_state)
        self.b_log_all.clicked.connect(self.log_all)
        self.b_log_assignment_problem.clicked.connect(self.logger.log_assignment_problem)
        self.show()

        self.b_log_save_regions.clicked.connect(self.log_save_regions)
        self.b_log_save_regions_collection.clicked.connect(self.log_save_regions_collection)
        self.b_log_save_frame.clicked.connect(self.log_save_frame)

        self.ch_ants_collection.setChecked(self.params.show_ants_collection)
        self.ch_mser_collection.setChecked(self.params.show_mser_collection)

        self.imshow_decreasing_factor.valueChanged.connect(self.imshow_decreasing_factor_changed)

        self.ch_assignment_problem.setChecked(False)


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
                if self.experiment.params.frame >= val and val != 0 \
                    or self.experiment.params.frame > 20:
                    self.b_play.setText('play')
                    self.is_running = False
                    #self.controls()

                    self.close()
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
                        save_and_exit_when_finished = self.settings.value(
                            'save_and_exit_when_finished',
                            self.params.save_and_exit_when_finished,
                            bool
                        )

                        if save_and_exit_when_finished:
                            print "SAVING"

                        print "End of video file!"
                    else:
                        print "End of history buffer!"
                        self.controls()
                        self.run()
                    return


                start = time.time()
                self.experiment.process_frame(img, self.forward)

                #print time.time() - start, "SECONDS", "self.experiment.process_frame()"

                self.display_informations()

                if self.results2video:
                    img_copy = self.experiment.img_.copy()

                    img_vis = visualize.draw_ants(img_copy, self.experiment.ants, self.experiment.regions, False, self.experiment.history)
                    self.vid_writer.write(img_vis)

                #self.logger.log_regions_collection()

                #self.log_all()
                #self.logger.log_regions()
                #self.logger.log_regions_collection()

                #if self.params.frame % 100 == 0:
                #self.logger.log_frame()
                #self.logger.log_frame_results()

                #self.logger.log_assignment_problem()


                #if self.params.frame % 1000 == 0:
                #    self.save_state()

                #print "------------------------"

                self.l_frame.setText(str(self.params.frame))
                self.controls()


    def display_informations(self):
        self.l_avg_a_area.setText(str(self.params.avg_ant_area)[0:5])
        self.l_avg_a_a.setText(str(self.params.avg_ant_axis_a)[0:5])
        self.l_mser_thresh.setText(str(self.params.intensity_threshold)[0:5])

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

    def save_state(self):
        print "SAVING....."
        path = self.out_state_directory
        print path
        out_name = self.i_state_name.text()
        frame = self.params.frame

        #self.params
        pfile = open(path+'/'+out_name+'-'+str(frame)+'-params.pkl', 'wb')
        pickle.dump(self.experiment.params, pfile)
        pfile.close()
        print "PARAMS SAVED"

        #self.ants
        afile = open(path+'/'+out_name+'-'+str(frame)+'-ants.pkl', 'wb')
        pickle.dump(self.experiment.ants, afile)
        afile.close()
        print "ANTS SAVED"
        print "DONE"

    def load_state(self):
        print "LOADING..."
        fname = QtGui.QFileDialog.getOpenFileNames(self, 'Open file', self.out_state_directory)

        ants = []
        params = []
        for f in fname:
            if str(f).find('-ants.pkl') > 0:
                pfile = open(f, 'rb')
                ants = pickle.load(pfile)
                pfile.close()
                print "ANTS LOADED"
            elif str(f).find('-params.pkl') > 0:
                pfile = open(f, 'rb')
                params = pickle.load(pfile)
                pfile.close()
                print "PARAMS LOADED"
            else:
                print "SOMETHING GOES WRONG, missing some files!"
                return


        self.experiment.ants = ants

        self.params = params
        self.experiment.params = params
        self.experiment.collisions = collisions.collision_detection(self.experiment.ants, self.experiment.history)

        if self.params.use_gt:
            self.experiment.ground_truth.rewind_gt(params.frame, params.ant_number)

        self.prepare_video_source(params.frame)
        self.experiment.video_manager = self.video_manager


        print "DONE"

        return

    def prepare_video_source(self, frame):
        self.video_manager = video_manager.VideoManager(self.params.video_file_name)
        for i in range(self.params.frame + 1):
            sys.stdout.write('\rvideo rewind ' + str(i) + '/' + str(self.params.frame))
            sys.stdout.flush() # important

            self.video_manager.next_img()

        print ""
        return

    def show_file_dialog(self):
        self.out_directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select directory"))

    def show_file_dialog_state(self):
        self.out_state_directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select directory"))

    def show_ants_collection_changed(self):
        self.experiment.params.show_ants_collection = self.ch_ants_collection.isChecked()

    def show_mser_collection_changed(self):
        self.experiment.params.show_mser_collection = self.ch_mser_collection.isChecked()

    def show_assignment_problem(self):
        self.experiment.params.show_assignment_problem = self.ch_assignment_problem.isChecked()

    def print_mser_info_changed(self):
        self.params.print_mser_info = self.ch_print_mser_info.isChecked()

    def imshow(self):
        self.params.show_image = self.ch_imshow.isChecked()

    def save_mat(self):
        path = self.out_directory

        data = self.experiment.ants_history_data()
        if len(path) > 0:
            path += '/'

        name = str(self.i_file_name.text())

        if len(name) == 0:
            name = 'undefined'

        path += name+'.mat'
        sio.savemat(path, {'Ferda': data})

    def save_data(self):
        path = self.out_directory

        data = self.experiment.ants_history_data()
        if len(path) > 0:
            path += '/'

        name = str(self.i_file_name.text())

        if len(name) == 0:
            name = 'undefined'

        results = self.experiment.results_xy_vector()
        results['info'] = {'ant_number': self.experiment.params.ant_number, 'video_file_name': self.experiment.params.video_file_name}
        afile = open(path+name+'_results.arr', "wb")
        pickle.dump(results, afile)
        afile.close()

        path += name+'.mat'

        sio.savemat(path, {'Ferda': data})

    def imshow_decreasing_factor_changed(self):
        self.params.imshow_decreasing_factor = self.imshow_decreasing_factor.value()

    def log_save_regions(self):
        self.logger.log_regions()

    def log_save_regions_collection(self):
        self.logger.log_regions_collection()

    def log_save_frame(self):
        self.logger.log_frame()

    def log_all(self):
        self.logger.log_regions()
        self.logger.log_regions_collection()
        self.logger.log_frame()
        self.logger.log_frame_results()
        self.logger.log_assignment_problem()


def main():
    app = QtGui.QApplication(sys.argv)
    ex = ControlWindow(experiment_params.Params())

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()