__author__ = 'flip'

import pickle
import cv2
import visualize


class Logger():
    def __init__(self, experiment):
        self.exp = experiment

    def log_regions_collection(self):
        img_copy = self.exp.img_.copy()

        collection = visualize.draw_region_group_collection(img_copy, self.exp.regions, self.exp.groups, self.exp.params)
        cv2.imwrite("out/noplast_gt_dump/collection_"+str(self.exp.params.frame)+".png", collection)

    def log_regions(self):
        afile = open(r'out/noplast_gt_dump/regions_'+str(self.exp.params.frame)+'.pkl', 'wb')
        pickle.dump(self.exp.regions, afile)
        afile.close()

    def log_frame(self):
        cv2.imwrite("out/noplast_gt_dump/frame"+str(self.exp.params.frame)+".png", self.exp.img_)

    def log_frame_results(self):
        img_copy = self.exp.img_.copy()

        img_vis = visualize.draw_ants(img_copy, self.exp.ants, self.exp.regions, True, self.exp.history)
        cv2.imwrite("out/noplast_gt_dump/frame_results_"+str(self.exp.params.frame)+".png", img_vis)