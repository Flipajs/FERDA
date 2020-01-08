import sys
import logging
from PyQt4 import QtGui
from collections import namedtuple
from os.path import exists
import matplotlib.pylab as plt
import fire
import pickle
import tqdm
import os

from core.project.project import Project
from .ant_blobs import AntBlobs
from .tracklet_types import TrackletTypes
from .util.blob_widget import BlobWidget
from .util.tracklet_viewer import TrackletViewer
import utils.roi
import core.region.region
from skimage.measure import regionprops

PickleGT = namedtuple("PickleGT", "project_name, video_file ant_blobs tracklet_types")


class AntBlobGtManager(object):

    def __init__(self, pkl_file, project, examples_from_tracklet=5):
        self.pkl_file = pkl_file
        self.examples_from_tracklet = examples_from_tracklet
        self.project = project
        if not exists(pkl_file):
            logging.info("There is no file named {}, creating a new one".format(self.pkl_file))
            self.pickle_gt = project.name, project.video_path, AntBlobs(), TrackletTypes()
        else:
            sys.path.append('data/GT/region_annotation_tools')  # hack to load old pkl file
            with open(self.pkl_file, 'r') as f:
                self.pickle_gt = pickle.load(f)
            logging.info("Successfuly loaded pkl file")
            sys.path.remove('data/GT/region_annotation_tools')
        self.project_name, self.video_file, self.ant_blobs, self.tracklet_types = self.pickle_gt

        self.check_valid_setting(project)

        logging.info("Currently, there are {0} labelled blob regions from {1} different tracklets".format(
            len(self.ant_blobs), len(self.tracklet_types)))

    def check_valid_setting(self, project):
        if self.project_name != project.name:
            logging.warning("Project name of loaded pickle - '{0}' differs from name of actual project - '{1}'"
                            "Please fix this before proceeding".format(self.project_name, project.name))
            raise AttributeError("Different project names")
        if self.video_file != project.video_path:
            logging.warning(
                "Video file(s) of loaded pickle - '{0}' differs from video file(s) of actual project - '{1}'"
                "Please fix this before proceeding".format(self.video_file, project.video_path))
            raise AttributeError("Different video file names")

    def get_ant_blobs(self):
        """
        Returns list of annotated blobs.

        :return: list of AntBlobs()
        """
        return self.ant_blobs.all_blobs()

    def feed_ant_blobs(self):
        return self.ant_blobs.feed_blobs()

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


def blobs_to_dict(blobs, img_shape, region_manager):
    """

    outputs ground truth in a list of dicts like:

    {'0_angle_deg': 196.81771700655054,
     '0_major': 63.263008639602724,
     '0_minor': 14.99769235802676,
     '0_x': 637.36877076411963,
     '0_y': 616.56478405315613,
     '1_angle_deg': 242.37572022616285,
     ...
     'data': '16 \xc5\x99\xc3\xadj 2017 20:46:44',
     'frame': 53970,
     'n_objects': 2,
     'region_id': 251801,
     'tracklet_id': 28633}

    :param blobs: list( (BlobInfo, BlobData), (BlobInfo, BlobData), ...)
    :param img_shape: tuple (height, width)
    :return: list of groundtruth dicts, see above
    """
    gt = []
    for val in tqdm.tqdm(blobs, desc='annotated blobs to ground truth'):
        item = dict(val[0]._asdict())
        item['data'] = val[1].date
        region = region_manager[item['region_id']]
        assert region.frame() == item['frame']
        roi = utils.roi.get_roi(region.pts())
        item['n_objects'] = len(val[1].ants)
        for i, obj in enumerate(val[1].ants):
            pts = obj + [roi.x(), roi.y()]

            # sub-optimal but working
            img_bw = np.zeros(img_shape, dtype=np.uint8)
            from skimage.draw import polygon
            rr, cc = polygon(pts[:, 1], pts[:, 0])
            img_bw[rr, cc] = 1
            properties = regionprops(img_bw)
            item['%d_x' % i] = properties[0].centroid[1]
            item['%d_y' % i] = properties[0].centroid[0]
            item['%d_major' % i] = properties[0].major_axis_length
            item['%d_minor' % i] = properties[0].minor_axis_length
            item['%d_angle_deg' % i] = np.degrees(properties[0].orientation) # 180 -
        gt.append(item)
    return gt



    from skimage.measure import label, regionprops
    import numpy as np


def region_annotation_tool(project_path, blob_gt_path, label_tracklets=False, label_blobs=False,
                           visualize_blobs=False, visualize_gt=False, write_interactions_dataset=False, out_dir='.'):
    # fix_video_filename('./Camera1_blob_gt.pkl', '/run/media/matej/mybook_ntfs/ferda/Camera 1.avi')
    logging.basicConfig(level=logging.INFO)
    p = Project(project_path)
    manager = AntBlobGtManager(blob_gt_path, p)
    img_shape = manager.project.img_manager.get_whole_img(0).shape[:2]

    # interactive ground truth annotation
    if label_tracklets:
        manager.label_tracklets()

    if label_blobs:
        manager.label_blobs()

    blobs = manager.get_ant_blobs()  # see AntBlobs() docs
    # blob_gen = manager.feed_ant_blobs()

    # visualize annotated blobs
    if visualize_blobs:
        for i, blob in enumerate(tqdm.tqdm(blobs)):
            fig = plt.figure()
            manager.show_gt(blob)
            plt.axis('off')
            fig.savefig(os.path.join(out_dir, 'blob_%03d.png' % i), transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    gt = blobs_to_dict(blobs, img_shape, manager.project.rm)

    # ellipses ground truth visualization
    if visualize_gt:
        from core.interactions.visualization import plot_interaction
        for i, item in tqdm.tqdm(enumerate(gt)):
            img = manager.project.img_manager.get_whole_img(item['frame'])
            fig = plt.figure()
            plt.imshow(img)
            plt.axis('off')
            plot_interaction(item['n_objects'], gt=item)
            fig.savefig(os.path.join(out_dir, 'gt_%05d.png' % i), transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            plt.clf()

    # # analyze gt in pandas
    # import pandas as pd
    # gtdf = pd.DataFrame(gt)

    # # compute ellipses gt from moments using Region() - NOT WORKING
    # import cv2
    # moments = cv2.moments(pts)
    #
    # data = {'area': moments['m00'],
    #         'cx': moments['m10'] / moments['m00'],
    #         'cy': moments['m01'] / moments['m00'],
    #         'label': None,
    #         'margin': None,
    #         'minI': None,
    #         'maxI': None,
    #         'sxx': moments['nu20'],
    #         'syy': moments['nu02'],
    #         'sxy': moments['nu11'],
    #         'parent_label': None,
    #         'rle': core.region.region.encode_RLE(pts[:, ::-1])
    #         }
    # gt_region = Region(data)
    #
    # obj = {'0_x': gt_region.centroid()[1],
    #        '0_y': gt_region.centroid()[0],
    #        '0_major': 4 * gt_region.major_axis_,
    #        '0_minor': 4 * gt_region.minor_axis_,
    #        '0_angle_deg': gt_region.theta_,
    # }
    # img = manager.project.img_manager.get_whole_img(item['frame'])
    # plt.imshow(img)
    # plot_interaction(1, obj)
    #
    # plot_interaction(num_objects, pred, gt)
    # fig.savefig(out_filename, transparent=True, bbox_inches='tight', pad_inches=0)
    # plt.close(fig)
    # plt.clf()

    # # not finished
    # if write_interactions_dataset:
    #     # write test dataset
    #     import h5py
    #     import warnings
    #
    #     out_hdf5 = 'images.h5'
    #     hdf5_dataset_name = 'test'
    #
    #     if out_hdf5 is not None:
    #         assert hdf5_dataset_name is not None
    #         if os.path.exists(out_hdf5):
    #             warnings.warn('HDF5 file %s already exists, adding dataset %s.' % (out_hdf5, hdf5_dataset_name))
    #         hdf5_file = h5py.File(out_hdf5, mode='a')
    #         hdf5_file.create_dataset(hdf5_dataset_name, (count, IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3),
    #                                  np.uint8)  # , compression='szip')


if __name__ == "__main__":
    fire.Fire(region_annotation_tool)


