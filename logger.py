__author__ = 'flip'

import pickle
import cv2
import visualize
import mser_operations

class Logger():
    def __init__(self, experiment):
        self.exp = experiment
        self.dir = experiment.params.dumpdir

    def log_regions_collection(self):
        img_copy = self.exp.img_.copy()

        collection = visualize.draw_region_group_collection(img_copy, self.exp.regions, self.exp.groups, self.exp.params)
        cv2.imwrite(self.dir+"/collections/"+str(self.exp.params.frame)+".png", collection)

        img_copy = self.exp.img_.copy()

        collection = visualize.draw_region_group_collection2(img_copy, self.exp.regions, self.exp.groups, self.exp.params)
        cv2.imwrite(self.dir+"/collections/"+str(self.exp.params.frame)+"_.png", collection)

    def log_regions(self):
        afile = open(self.dir+"/regions/"+str(self.exp.params.frame)+".pkl", "wb")
        pickle.dump(self.exp.regions, afile)
        afile.close()

    def log_frame(self):
        cv2.imwrite(self.dir+"/frames/"+str(self.exp.params.frame)+".png", self.exp.img_)

    def log_frame_subtracted(self):
        cv2.imwrite(self.dir+"/frames_sub/"+str(self.exp.params.frame)+".png", self.exp.img_sub)

    def log_frame_results(self):
        img_copy = self.exp.img_.copy()

        img_vis = visualize.draw_ants(img_copy, self.exp.ants, self.exp.regions, False, self.exp.history)
        cv2.imwrite(self.dir+"/frame_results/"+str(self.exp.params.frame)+".png", img_vis)

    def log_assignment_problem(self):
        prev_img = self.exp.video_manager.get_prev_img()
        img = self.exp.img_.copy()
        ants = self.exp.ants
        regions = self.exp.regions
        groups = self.exp.chosen_regions_indexes
        params = self.exp.params

        img_vis = visualize.draw_assignment_problem(prev_img, img, ants, regions, groups, params)
        cv2.imwrite(self.dir+"/assignment_problem/"+str(self.exp.params.frame)+".png", img_vis)