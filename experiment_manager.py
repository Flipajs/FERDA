__author__ = 'flip'

import ant
import numpy
import cv2
import mserOperations
import score
import visualize
import gt
import utils as my_utils

#TODO> SVM for ant orientation
#TODO> SVM for antlike mser?

#TODO> mravenec vybira regions
#TODO> MSER if > 254 nepridavej do grafu
# kdyz hrozi kolize, mnohem ostrejsi pravidla na fit...
# jinak muze byt clovek celkem benevolentni...

class ExperimentManager():
    def __init__(self, params, ants):
        self.ant_number = params.ant_number
        self.params = params

        self.ants = ants

        self.use_gt = False

        if self.use_gt:
            self.ground_truth = gt.GroundTruth('fixed_out.txt', self)
            self.ground_truth.match_gt(self.ants)

        self.mser_operations = mserOperations.MserOperations(params)
        self.count_ant_params()

        self.img_ = None

    def process_frame(self, img, wait_for_button_press):
        self.img_ = img.copy()

        mask = numpy.ones((numpy.shape(img)[0], numpy.shape(img)[1], 1), dtype=numpy.uint8)*255
        cv2.circle(mask, self.params.arena.center.int_tuple(), self.params.arena.size.width/2, 0, -1)
        ##res = cv2.bitwise_and(self.img_, self.img_, mask=mask)
        #res = cv2.bitwise_or(self.img_, self.img_, mask=mask)
        idx = (mask==0)
        mask[idx] = self.img_[idx]

        self.params.arena
        regions, indexes = self.mser_operations.process_image(mask)
        result = score.max_weight_matching(self.ants, regions)

        print result

        for i in range(self.ant_number):
            if result[i] < 0:
                ant.set_ant_state_undefined(self.ants[i], result[i])
            else:
                ant.set_ant_state(self.ants[i], result[i], regions[result[i]])

        img_cpy = self.img_.copy()
        img_vis = visualize.draw_ants(self.img_, self.ants, regions, True)
        #draw_dangerous_areas(I)
        my_utils.imshow("ant track result", img_vis, True)
        if self.params.show_mser_collections:
            collection = visualize.draw_region_collection(img_cpy, regions, self.params)
            my_utils.imshow("collection", collection)

        if self.use_gt:
            r = self.ground_truth.check_gt(self.ants, False)
            if r.count(0) > 0:
                broken_idx = [i for i in range(len(r)) if r[i] == 0]
                for i in broken_idx:
                    print self.ants[i].state

                self.make_log(regions, indexes)
                print "FAIL"
                print r

                while True:
                    k = cv2.waitKey(0)
                    if k == 'n':
                        break

            print self.ground_truth.stats()

        if wait_for_button_press:
            while True:
                k = cv2.waitKey(0)
                if k == 32:
                    break
        else:
            cv2.waitKey(5)



    def make_log(self, regions, indexes):
        self.ants
        #print regions
        print indexes

    def count_ant_params(self):
        avg_area = 0
        avg_axis_rate = 0
        counter = 0
        for a in self.ants:
            if a.state.mser_id == -1:
                continue

            avg_area += a.state.area
            avg_axis_rate += a.state.axis_rate
            counter += 1

        if counter > 0:
            self.params.avg_ant_area = avg_area / counter
            self.params.avg_ant_axis_ratio = avg_axis_rate / counter
        else:
            print "zero ant assigned... in Experiment_manager.py"

        print "AVG ANT AREA> ", self.params.avg_ant_area
        print "AVG ANT AXIS RATIO> ", self.params.avg_ant_axis_ratio

    def ants_history_data(self):
        data = [None] * self.ant_number
        for i in range(self.ant_number):
            a = self.ants[i]
            vals = a.buffer_history()
            vals['moveiname'] = self.params.video_file_name
            data[i] = vals

        return data