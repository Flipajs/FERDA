import sys
import logging
from PyQt4 import QtGui
from collections import namedtuple
from os.path import exists

import pickle

from core.project.project import Project
from ant_blobs import AntBlobs
from tracklet_types import TrackletTypes
from util.blob_widget import BlobWidget
from util.tracklet_viewer import TrackletViewer
import matplotlib.pylab as plt
import utils.roi
import core.region.region

PickleGT = namedtuple("PickleGT", "project_name, video_file ant_blobs tracklet_types")


class AntBlobGtManager(object):

    def __init__(self, pkl_file, project, examples_from_tracklet=5):
        self.pkl_file = pkl_file
        self.examples_from_tracklet = examples_from_tracklet
        self.project = project
        if not exists(pkl_file):
            logging.info("There is no file named {}, creating a new one".format(self.pkl_file))
            self.pickle_gt = project.name, project.video_paths, AntBlobs(), TrackletTypes()
        else:
            with open(self.pkl_file, 'r') as f:
                self.pickle_gt = pickle.load(f)
            logging.info("Successfuly loaded pkl file")
        self.project_name, self.video_file, self.ant_blobs, self.tracklet_types = self.pickle_gt

        self.check_valid_setting(project)

        logging.info("Currently, there are {0} labelled blob regions from {1} different tracklets".format(
            len(self.ant_blobs), len(self.tracklet_types)))

    def check_valid_setting(self, project):
        if self.project_name != project.name:
            logging.warning("Project name of loaded pickle - '{0}' differs from name of actual project - '{1}'"
                            "Please fix this before proceeding".format(self.project_name, project.name))
            raise AttributeError("Different project names")
        if self.video_file != project.video_paths:
            logging.warning(
                "Video file(s) of loaded pickle - '{0}' differs from video file(s) of actual project - '{1}'"
                "Please fix this before proceeding".format(self.video_file, project.video_paths))
            raise AttributeError("Different video file names")

    def get_ant_blobs(self):
        # retrieve GT
        return self.ant_blobs.all_blobs()

    def feed_ant_blobs(self):
        return self.ant_blobs.feed_blobs()

    def view_gt(self):
        pass

    def label_tracklets(self):
        # label tracklets first
        app = QtGui.QApplication(sys.argv)
        tracklets = self.tracklet_types.get_unlabeled(self.project.chm.chunk_list())
        viewer = TrackletViewer(self.project, tracklets, self.set_label, self.save_and_exit)
        viewer.show()
        app.exec_()

    def label_blobs(self):
        # then segment these tracklets
        app = QtGui.QApplication(sys.argv)
        tracklets = self.tracklet_types.get_labeled_blobs(self.project.chm.chunk_list())
        filtered = self.ant_blobs.filter_labeled_tracklets(tracklets)
        widget = BlobWidget(self.project, filtered, self.examples_from_tracklet,
                            self.set_blobs, self.save_and_exit, self.ant_blobs.contains)
        widget.show()
        app.exec_()

    def set_blobs(self, region_id, frame, tracklet_id, ants):
        logging.info("Saving region id {0} on frame {1} from tracklet {2}".format(region_id, frame, tracklet_id))
        self.ant_blobs.insert(region_id, frame, tracklet_id, ants)

    def set_label(self, tracklet_id, label):
        logging.info("Tracklet id {0} labeled as {1}".format(
            tracklet_id, "BLOB" if label == self.tracklet_types.BLOB else "SINGLE" if label == self.tracklet_types.SINGLE
            else "OTHER"
        ))
        self.tracklet_types.insert(tracklet_id, label)

    def save_and_exit(self):
        logging.info("Saving and exiting")
        self.__save()
        import sys
        sys.exit(0)

    def show_gt(self, blob):
        plt.imshow(self.project.img_manager.get_whole_img(blob[0].frame))
        region = self.project.rm[blob[0].region_id]
        assert region.frame() == blob[0].frame
        roi = utils.roi.get_roi(region.pts())
        for ant in blob[1].ants:
            pts = ant + [roi.x(), roi.y()]
            plt.plot(pts[:, 0], pts[:, 1])
    # region_single = core.region.region.Region(pts)

    def __save(self):
        logging.info("Saving to {0}".format(self.pkl_file))
        logging.info("Currently, there are {0} labelled blob regions from {1} different tracklets".format(
            len(self.ant_blobs), len(self.tracklet_types)))
        with open(self.pkl_file, 'w') as f:
            pickle.dump((self.project_name, self.video_file, self.ant_blobs, self.tracklet_types), f)


def fix_video_filename(pkl_filename, video_filename):
    with open(pkl_filename, 'r') as f:
        project_name, video_file, ant_blobs, tracklet_types = pickle.load(f)
    video_file[0] = video_filename
    with open(pkl_filename, 'w') as f:
        pickle.dump((project_name, video_file, ant_blobs, tracklet_types), f)


if __name__ == "__main__":
    # fix_video_filename('./test.pkl', '/run/media/matej/mybook_ntfs/ferda/Camera 1.avi')
    logging.basicConfig(level=logging.INFO)
    p = Project()
    # p.load("/home/matej/prace/ferda/10-15 (copy)/10-15.fproj")
    p.load("/home/matej/prace/ferda/projects/Camera 1/Camera 1.fproj")
    manager = AntBlobGtManager('./test.pkl', p)
    manager.label_tracklets()
    manager.label_blobs()
    manager.view_gt()
    blob_dic = manager.get_ant_blobs()
    blob_gen = manager.feed_ant_blobs()

    # list( (BlobInfo, BlobData), (BlobInfo, BlobData), ...)
    # BlobInfo(region_id=63729, frame=14431, tracklet_id=70922)
    # BlobData(ants=[array([[23, 54], ...]),
    #                array([[45, 88], ...]),
    #                ...],
    #          date='22 z\xc3\xa1\xc5\x99 2017 16:59:46')

    for blob in blob_dic:
        plt.figure()
        manager.show_gt(blob)







