from __future__ import print_function

"""
use: $ python interactions.py -- --help
"""

#TODO change self.__video.get_frame() for cached self._project.img_manager.get_whole_img()

import sys
import cPickle as pickle
from utils.misc import is_flipajs_pc, is_matejs_pc
from core.project.project import Project
import numpy as np
import matplotlib.pylab as plt
import math
import os.path
from core.graph.region_chunk import RegionChunk
from utils.video_manager import get_auto_video_manager
from core.region.transformableregion import TransformableRegion
from core.region.region import get_region_endpoints
import cv2
import fire
import tqdm
# from joblib import Parallel, delayed
import csv
import waitforbuttonpress
import h5py
import warnings
from itertools import product
from scripts.CNN.interactions_results import save_prediction_img
from os.path import join
import pandas as pd
import errno
import hashlib
from utils.img import safe_crop

IMAGE_SIZE_PX = 200

#TODO: check for possible bug
def head_fix(tracklet_regions):
    import heapq
    q = []
    # q = Queue.PriorityQueue()
    heapq.heappush(q, (0, [False]))
    heapq.heappush(q, (0, [True]))
    # q.put((0, [False]))
    # q.put((0, [True]))

    result = []
    i = 0
    max_i = 0

    cut_diff = 10

    while True:
        i += 1

        # cost, state = q.get()
        cost, state = heapq.heappop(q)
        if len(state) > max_i:
            max_i = len(state)

        if len(state) + cut_diff < max_i:
            continue

        # print i, cost, len(state), max_i

        if len(state) == len(tracklet_regions):
            result = state
            break

        prev_r = tracklet_regions[len(state) - 1]
        r = tracklet_regions[len(state)]

        prev_c = prev_r.centroid()
        p1, p2 = get_region_endpoints(r)

        dist = np.linalg.norm
        d1 = dist(p1 - prev_c)
        d2 = dist(p2 - prev_c)

        prev_head, prev_tail = get_region_endpoints(prev_r)
        if state[-1]:
            prev_head, prev_tail = prev_tail, prev_head

        d3 = dist(p1 - prev_head) + dist(p2 - prev_tail)
        d4 = dist(p1 - prev_tail) + dist(p2 - prev_head)

        # state = list(state)
        state2 = list(state)
        state.append(False)
        state2.append(True)

        new_cost1 = d3
        new_cost2 = d4

        # TODO: param
        if dist(prev_c - r.centroid()) > 5:
            new_cost1 += d2 - d1
            new_cost2 += d1 - d2

        heapq.heappush(q, (cost + new_cost1, state))
        heapq.heappush(q, (cost + new_cost2, state2))
        # q.put((cost + new_cost1, state))
        # q.put((cost + new_cost2, state2))

    for b, r in zip(result, tracklet_regions):
        if b:
            r.theta_ += np.pi
            if r.theta_ >= 2 * np.pi:
                r.theta_ -= 2 * np.pi


class Interactions(object):
    def __init__(self):
        self._video = None
        self._single = None
        self._multi = None
        self._project = None
        self._bg = None
        self.collect_regions_params = {'single_min_frames': 20,
                                       'single_min_average_speed_px': 1.5
                                       }
        # self.__i = 0  # used for visualizations commented out

    def _load_project(self, project_dir=None, video_file=None):
        self._project = Project()
        # This is development speed up process (kind of fast start). Runs only on developers machines...
        # if is_flipajs_pc() and False:
        if project_dir is None:
            if is_flipajs_pc():
                # project_dir = '/Users/iflipajs/Documents/project_dir/FERDA/Cam1_rf'
                # project_dir = '/Users/flipajs/Documents/project_dir/FERDA/Cam1_playground'
                # project_dir = '/Users/flipajs/Documents/project_dir/FERDA/test6'
                # project_dir = '/Users/flipajs/Documents/project_dir/FERDA/zebrafish_playground'
                # project_dir = '/Users/flipajs/Documents/project_dir/FERDA/Camera3'
                # project_dir = '/Users/flipajs/Documents/project_dir/FERDA/Cam1_rfs2'
                # project_dir = '/Users/flipajs/Documents/project_dir/FERDA/Cam1'
                project_dir = '/Users/flipajs/Documents/project_dir/FERDA/rep1-cam2'
                # project_dir = '/Users/flipajs/Documents/project_dir/FERDA/rep1-cam3'

                # project_dir = '/Users/flipajs/Documents/project_dir/FERDA/Sowbug3'

                # project_dir = '/Users/flipajs/Documents/project_dir/FERDA/test'
            if is_matejs_pc():
                # project_dir = '/home/matej/prace/ferda/10-15/'
                project_dir = '/home/matej/prace/ferda/projects/camera1_10-15/'
        assert project_dir is not None
        self._project.load(project_dir, video_file=video_file)

        # img = video.get_frame(region.frame())  # ndarray bgr
        # img_region = get_img_around_pts(img, region.pts(), margin=0)
        # cv2.imshow('test', img_region[0])
        # cv2.waitKey()

        self._video = get_auto_video_manager(self._project)

    def _get_hash(self, *args):
        m = hashlib.md5()
        for arg in args:
            m.update(str(arg))
        return m.hexdigest()

    def _init_regions(self, cache_dir='../data/cache'):
        hash = self._get_hash(self._project.video_paths, self.collect_regions_params)
        regions_filename = join(cache_dir, hash + '.pkl')
        if os.path.exists(regions_filename):
            print('regions loading...')
            with open(regions_filename, 'rb') as fr:
                self._single = pickle.load(fr)
                self._multi = pickle.load(fr)
            print('regions loaded')
        else:
            self._single, self._multi = self._collect_regions()
            with open(regions_filename, 'wb') as fw:
                pickle.dump(self._single, fw)
                pickle.dump(self._multi, fw)

    def _collect_regions(self):
        p = self.collect_regions_params
        from collections import defaultdict
        single = defaultdict(list)
        multi = defaultdict(list)
        # long_moving_tracklets = []
        for tracklet in tqdm.tqdm(self._project.chm.chunk_gen(), desc='collecting regions', total=len(self._project.chm)):
            if tracklet.is_single() or tracklet.is_multi():
                region_tracklet = RegionChunk(tracklet, self._project.gm, self._project.rm)
                if tracklet.is_single():
                    centroids = np.array([region_tracklet.centroid_in_t(frame) for frame
                                          in
                                          range(region_tracklet.start_frame(),
                                                region_tracklet.end_frame())])  # shape=(n,2)

                    if len(tracklet) > p['single_min_frames'] and \
                            np.linalg.norm(np.diff(centroids, axis=0), axis=1).mean() > p['single_min_average_speed_px']:
                        # long_moving_tracklets.append(tracklet)

                        regions = list(region_tracklet)
                        head_fix(regions)
                        # images = []
                        # for region in regions:
                        #     img = video.get_frame(region.frame())
                        #     head, tail = get_region_endpoints(region)
                        #     img = cv2.drawMarker(img, tuple(head[::-1].astype(int)), (0, 0, 255))
                        #     img = cv2.drawMarker(img, tuple(tail[::-1].astype(int)), (0, 255, 0))
                        #     border = 10
                        #     roi = region.roi()
                        #     images.append(img[roi.y() - border:roi.y() + roi.height() + border,
                        #                   roi.x() - border:roi.x() + roi.width() + border])
                        # montage_generator = montage.Montage((1000, 500), (10, 5))
                        # plt.imshow(montage_generator.montage(images[:50])[::-1])
                        # plt.waitforbuttonpress()
                        for region in regions:
                            single[region.frame()].append(region)
                    else:
                        # short single tracklets are ignored
                        pass
                else:  # multi
                    for region in region_tracklet.regions_gen():
                        # if tracklet.is_single():
                        #     single[region.frame()].append(region)
                        # else:
                        multi[region.frame()].append(region)

        return single, multi

    def _get_out_dir_rel(self, out_dir, out_file):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        return os.path.relpath(os.path.abspath(out_dir), os.path.abspath(os.path.dirname(out_file)))

    def write_regions_for_testing(self, project_file, out_dir, n, multi=True, single=False):
        """
        Write n randomly chosen cropped region images in hdf5 file.

        :param project_file: gather regions from the FERDA project file
        :param out_dir: output directory where images.h5 and parameters.txt will be placed
        :param n: number of region images to write
        :param multi: include multi animal regions
        :param single: include single animal regions
        """
        # write-regions-for-testing /home/matej/prace/ferda/projects/camera1_10-15/10-15.fproj /home/matej/prace/ferda/data/interactions/180129_camera1_10-15_multi_test 100
        self._load_project(project_file)
        self._write_argv(out_dir)
        # load regions
        self._init_regions()
        regions = []
        if multi:
            regions.extend([item for sublist in self._multi.values() for item in sublist])
        if single:
            regions.extend([item for sublist in self._single.values() for item in sublist])
        n_regions = np.random.choice(regions, n)
        frames = np.array([r.frame() for r in n_regions])
        sort_idx = np.argsort(frames)
        # sort_idx_reverse = np.argsort(sort_idx)

        # initialize hdf5 output file
        out_hdf5 = join(out_dir, 'images.h5')
        dataset = 'test'
        if os.path.exists(out_hdf5):
            raise OSError(errno.EEXIST, 'HDF5 file %s already exists.' % out_hdf5)
        hdf5_file = h5py.File(out_hdf5, mode='w')
        hdf5_file.create_dataset(dataset, (n, IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3), np.uint8)

        # load images, crop regions and save to hdf5
        for i, (frame, region) in tqdm.tqdm(enumerate(zip(frames[sort_idx], n_regions[sort_idx])), total=n):
            img = self._project.img_manager.get_whole_img(frame)
            img_crop, delta_xy = safe_crop(img, region.centroid()[::-1], IMAGE_SIZE_PX)
            hdf5_file[dataset][i, ...] = img_crop
        hdf5_file.close()

    def _write_argv(self, out_dir):
        with open(join(out_dir, 'parameters.txt'), 'w') as fw:
            fw.writelines('\n'.join(sys.argv))

    def write_annotated_blobs_groundtruth(self, project_file, blobs_filename, n_objects, out_dir):
        """
        Load annotated blobs and write interaction ground truth to hdf5 and csv files.

        For annotated blobs see data.GT.ant_blob.ant_blob_gt_manager.
        """
        # write_annotated_blobs_groundtruth '/home/matej/prace/ferda/projects/camera1/Camera 1.fproj' ../data/annotated_blobs/Camera1_blob_gt.pkl 2 ../data/interactions/180126_test_real_2_ants
        self._write_argv(out_dir)
        import data.GT.ant_blob.ant_blob_gt_manager as ant_blob_gt_manager
        self._load_project(project_file)
        img_shape = self._project.img_manager.get_whole_img(0).shape[:2]
        blob_manager = ant_blob_gt_manager.AntBlobGtManager(blobs_filename, self._project)
        blobs = blob_manager.get_ant_blobs()  # see AntBlobs() docs
        gt = pd.DataFrame(ant_blob_gt_manager.blobs_to_dict(blobs, img_shape, self._project.rm))
        gt_n = gt[gt.n_objects == n_objects].copy()

        out_hdf5 = join(out_dir, 'images.h5')
        dataset = 'test'
        if os.path.exists(out_hdf5):
            raise OSError(errno.EEXIST, 'HDF5 file %s already exists.' % out_hdf5)
        hdf5_file = h5py.File(out_hdf5, mode='w')
        hdf5_file.create_dataset(dataset, (len(gt_n), IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3), np.uint8)  # , compression='szip')
        for i, (_, item) in enumerate(tqdm.tqdm(gt_n.iterrows(), desc='writing images')):
            img = self._project.img_manager.get_whole_img(item['frame'])
            region = self._project.rm[item['region_id']]
            img_crop, delta_xy = safe_crop(img, region.centroid()[::-1], IMAGE_SIZE_PX)
            for j in range(n_objects):
                gt_n.loc[gt_n.index[i], '%d_x' % j] -= delta_xy[0]
                gt_n.loc[gt_n.index[i], '%d_y' % j] -= delta_xy[1]
            hdf5_file[dataset][i, ...] = img_crop
        hdf5_file.close()

        out_csv = join(out_dir, 'test.csv')
        columns = []
        for i in range(n_objects):
            gt_n.loc[:, '%d_region_id' % i] = gt_n['region_id']
            columns.extend([
                '%d_x' % i, '%d_y' % i, '%d_major' % i, '%d_minor' % i,
                '%d_angle_deg' % i, '%d_region_id' % i
            ])
        gt_out = gt_n.loc[:, columns]
        gt_out['video_file'] = os.path.basename(self._video.video_path)
        gt_out.to_csv(out_csv, index=False, float_format='%.2f')
        # gt.to_hdf(join(out_dir, 'test.h5'), )

    def write_background(self, video_step=100, dilation_size=100, out_filename='bg.png'):
        """
        Generates background without foreground objects.

        :param video_step: search step in video
        :param dilation_size: dilate foreground objects
        :param out_filename:
        :return:
        """
        self._load_project()
        self._init_regions()
        # background composition
        img = self._video.get_frame(0)

        multiple_imgs = []  # np.zeros((median_len,) + img.shape)
        bg_mask = np.zeros(img.shape[:2], dtype=np.bool)
        i = 0
        print('frame number\tmissing bg pixels')
        while not np.all(bg_mask):
            frame = i * video_step
            print(frame, '\t', np.count_nonzero(~bg_mask))
            img = self._video.get_frame(frame).astype(float)
            img_bg_mask = np.zeros(img.shape[:2], dtype=np.bool)
            for region in self._single[frame] + self._multi[frame]:
                transformable_region = TransformableRegion(img)
                transformable_region.set_region(region)
                mask = transformable_region.get_mask(dilation_size)
                img[mask] = np.nan
                img_bg_mask = np.bitwise_or(img_bg_mask, mask)
            bg_mask = np.bitwise_or(bg_mask, ~img_bg_mask)
            multiple_imgs.append(img)
            i += 1

        median_img = np.nanmedian(np.stack(multiple_imgs), 0).astype(np.uint8)
        cv2.imwrite(out_filename, median_img)

    def write_foreground(self, out_dir='fg', out_file='fg.txt'):
        """
        Write all foreground objects as images and list them into a text file for training.

        :param out_dir:
        :param out_file:
        :return:
        """
        self._load_project()
        self._init_regions()
        # write fg images and txt file
        out_dir_rel = self._get_out_dir_rel(out_dir, out_file)

        with open(out_file, 'w') as fw:
            for frame in tqdm.tqdm(sorted(self._single.keys())):
                img = self._video.get_frame(frame)
                filename = '%05d.jpg' % frame
                cv2.imwrite(os.path.join(out_dir, filename), img)
                line = os.path.join(out_dir_rel, filename) + ' ' + str(len(self._single[frame])) + ' '
                for region in self._single[frame]:
                    roi = region.roi()
                    line += '%d %d %d %d ' % (roi.x(), roi.y(), roi.width(), roi.height())
                line += '\n'
                fw.write(line)

    def write_meanimage(self, out_filename='mean_foreground.png', width=70, height=100, num_regions=80):
        """
        Creates mean "single" foreground object.

        :param out_filename:
        :param width:
        :param height:
        :param num_regions:
        :return:
        """
        self._load_project()
        self._init_regions()
        frames_regions = []
        for _ in range(num_regions):
            region = np.random.choice(self._single[np.random.choice(self._single.keys())])
            frames_regions.append((region.frame(), region))
        frames_regions = sorted(frames_regions, key=lambda fr: fr[0])

        # single_non_empty = {f: s for f, s in self.single.iteritems() if s != []}
        #
        # for i in range(num_regions):
        #     region = np.random.choice(single_non_empty[single_non_empty.keys()[i]])
        #     frames_regions.append((region.frame(), region))

        mean_img = np.zeros((height, width, 3))
        for frame, region in tqdm.tqdm(frames_regions):
            # img = cv2.cvtColor(self.video.get_frame(region.frame()), cv2.COLOR_BGR2GRAY)
            tregion = TransformableRegion(self._video.get_frame(frame))
            tregion.rotate(-math.degrees(region.theta_) + 90, rotation_center_yx=region.centroid())
            img_aligned = tregion.get_img()
            hw = np.array((height, width), dtype=float)
            tlbr = np.hstack((region.centroid() - hw / 2, region.centroid() + hw / 2)).round().astype(int)
            try:
                mean_img += img_aligned[tlbr[0]:tlbr[2], tlbr[1]:tlbr[3], :]
            except ValueError:
                # out of image bounds
                num_regions -= 1

        cv2.imwrite(out_filename, (mean_img / float(num_regions)).astype(np.uint8))

    def write_background_aligned(self, out_dir='bg', out_file='bg.txt', bg_angle_max_deviation_deg=20):
        """
        Write backgrounds for training an aligned object detector: the
        backgrounds include non-vertical foreground objects.

        :param out_dir:
        :param out_file:
        :param bg_angle_max_deviation_deg:
        :return:
        """
        self._load_project()
        self._init_regions()
        out_dir_rel = self._get_out_dir_rel(out_dir, out_file)
        images = []
        i = 0
        bb_fixed_border_xy = (20, 10)

        fw = open(out_file, 'w')

        for frame in tqdm.tqdm(sorted(self._single.keys())):  # [::10]:
            img = self._video.get_frame(frame)

            for region in self._single[frame]:  # [::10]:

                if not (-bg_angle_max_deviation_deg < math.degrees(region.theta_) - 90 < bg_angle_max_deviation_deg):
                    filename = '%05d.png' % i
                    roi = region.roi()
                    crop = img[roi.y(): roi.y() + roi.height() + 1,
                               roi.x(): roi.x() + roi.width() + 1]
                    cv2.imwrite(os.path.join(out_dir, filename), crop)
                    fw.write(os.path.join(out_dir_rel, filename) + '\n')
                    # plt.imshow(crop)
                    # plt.waitforbuttonpress()


                # tcorners = np.int32(np.dot(c
                #  rot, img.shape[:2])[roi.y():roi.y() + roi.height(), roi.x():roi.x() + roi.width()])
                #                             #  (roi.height() + 10, roi.width() + 10)))
                # size = crop.shape[:2]
                # half = (size / 2.)[::-1]
                # crop = cv2.arrowedLine(crop, half, half + min(half) * []
                # head, tail = get_region_endpoints(region)
                # img = cv2.drawMarker(img, tuple(head[::-1].astype(int)), (0, 0, 255))
                # img = cv2.drawMarker(img, tuple(tail[::-1].astype(int)), (0, 255, 0))
                # border = 10
                # images.append(img[roi.y() - border:roi.y() + roi.height() + border,
                #                   roi.x() - border:roi.x() + roi.width() + border])
                i += 1
                # if i > 200:
                #     plt.imshow(montage.montage(images[:200])[::-1])
                #     plt.waitforbuttonpress()
                #     i = 0
                #     images = []

        fw.close()

    def write_foreground_aligned(self, out_dir='fg', out_file='fg.txt', fixed_size=True):
        """
        Write foreground objects for training an aligned objects detector. The foregrounds include
        only vertical objects.

        :param out_dir:
        :param out_file:
        :param fixed_size:
        :return:
        """
        self._load_project()
        self._init_regions()
        out_dir_rel = self._get_out_dir_rel(out_dir, out_file)
        images = []
        i = 0
        bb_fixed_border_xy = (20, 10)

        if fixed_size:
            import itertools

            all_regions = list(itertools.chain(*self._single.values()))
            major_axes = np.array([r.a_ for r in all_regions]) * 2
            bb_major_px = np.median(major_axes)
            minor_axes = np.array([r.b_ for r in all_regions]) * 2
            bb_minor_px = np.median(minor_axes)
            bb_size = (round(bb_minor_px + 2 * bb_fixed_border_xy[0]),
                       round(bb_major_px + 2 * bb_fixed_border_xy[1]))

        fw = open(out_file, 'w')
        # with open('./out/regions.pkl', 'wb') as fw:
        #     frames = sorted(single.keys())
        #     i = 0
        #     r1 = single[frames[i]][0]
        #     img1 = video.get_frame(frames[i])
        #     i = 1
        #     r2 = single[frames[i]][0]
        #     img2 = video.get_frame(frames[i])
        #     pickle.dump(r1, fw)
        #     pickle.dump(img1, fw)
        #     pickle.dump(r2, fw)
        #     pickle.dump(img2, fw)

        for frame in tqdm.tqdm(sorted(self._single.keys())):  # [::10]:
            img = self._video.get_frame(frame)

            for region in self._single[frame]:  # [::10]:

                centroid_crop = tuple(region.centroid()[::-1] - cropxy)
                rot = cv2.getRotationMatrix2D(centroid_crop,
                                              -math.degrees(region.theta_) + 90, 1.)
                crop_rot = cv2.warpAffine(crop, rot, crop.shape[:2][::-1])

                if not fixed_size:
                    head, tail = np.array(get_region_endpoints(region))
                    head_tail_xy = np.vstack((head[::-1] - cropxy, tail[::-1] - cropxy))
                    head_rotated, tail_rotated = rot.dot(np.hstack((head_tail_xy, np.ones((2, 1)))).T).T.astype(int)
                    border_xy = (30, 10)
                    img_aligned = crop_rot[max(head_rotated[1] - border_xy[1], 0):
                    min(tail_rotated[1] + border_xy[1], crop_rot.shape[0]),
                                  max(head_rotated[0] - border_xy[0], 0):
                                  min(tail_rotated[0] + border_xy[0], crop_rot.shape[1])]
                else:
                    img_aligned = crop_rot[max(int(round(centroid_crop[1] - bb_size[1] / 2)), 0):
                    min(int(round(centroid_crop[1] + bb_size[1] / 2)), crop_rot.shape[0]),
                                  max(int(round(centroid_crop[0] - bb_size[0] / 2)), 0):
                                  min(int(round(centroid_crop[0] + bb_size[0] / 2)), crop_rot.shape[1])]

                # [tl, tr, br, bl]
                # corners = np.array([[roi.x() - border, roi.y() - border],
                #            [roi.x() + roi.width() + border, roi.y() - border],
                #            [roi.x() + roi.width() + border, roi.y() + roi.height() + border],
                #            [roi.x() - border, roi.y() + roi.height() + border]])
                # corners_rotated = rot.dot(np.hstack((corners, np.ones((4,1)))).T).T
                # x, y, w, h = cv2.boundingRect(corners_rotated.astype(int).reshape(1, -1, 2))
                # img = cv2.warpAffine(img, rot, img.shape[:2])
                # images.append(img[y:y+h, x:x+w])
                # plt.imshow(crop_rot)
                # plt.imshow(crop_rot[head_rotated[1] - border_xy[1]:tail_rotated[1] + border_xy[1],
                # head_rotated[0] - border_xy[0]:tail_rotated[0] + border_xy[0]])

                # images.append()
                filename = '%05d.png' % i
                cv2.imwrite(os.path.join(out_dir, filename), img_aligned)
                line = os.path.join(out_dir_rel, filename) + ' 1 0 0 %d %d\n' % (img_aligned.shape[1], img_aligned.shape[0])
                fw.write(line)

                # tcorners = np.int32(np.dot(corners, rot.T))
                # x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
                # images.append(cv2.warpAffine(img, rot, img.shape[:2])[roi.y():roi.y() + roi.height(), roi.x():roi.x() + roi.width()])
                #                             #  (roi.height() + 10, roi.width() + 10)))
                # size = crop.shape[:2]
                # half = (size / 2.)[::-1]
                # crop = cv2.arrowedLine(crop, half, half + min(half) * []
                # head, tail = get_region_endpoints(region)
                # img = cv2.drawMarker(img, tuple(head[::-1].astype(int)), (0, 0, 255))
                # img = cv2.drawMarker(img, tuple(tail[::-1].astype(int)), (0, 255, 0))
                # border = 10
                # images.append(img[roi.y() - border:roi.y() + roi.height() + border,
                #                   roi.x() - border:roi.x() + roi.width() + border])
                i += 1
                # if i > 200:
                #     plt.imshow(montage.montage(images[:200])[::-1])
                #     plt.waitforbuttonpress()
                #     i = 0
                #     images = []

        fw.close()

    def write_synthetized_interactions(self, project_dir, count=100, n_objects=2, out_csv='./out/doubleregions.csv',
                                       rotation='random', xy_jitter_width=0, out_hdf5=None,
                                       hdf5_dataset_name=None, write_masks=False, out_image_dir=None):
        # write_synthetized_interactions --project_dir /home/matej/prace/ferda/projects/camera1_10-15/ --count=360
        # --n-objects=1 --out-csv=../data/interactions/180208_1k_36rot_single_mask/train.csv --rotation 10  --xy_jitter_width=40 --out_hdf5=../data/interactions/180208_1k_36rot_single_mask/images.h5 --hdf5_dataset_name=train
        if rotation == 'random' or rotation == 'normalize':
            n_angles = 1
        elif isinstance(rotation, int):
            rotations = np.arange(0, 360, rotation)
            n_angles = len(rotations)
        else:
            assert False, 'wrong rotation parameter'
        assert count % n_angles == 0

        COLUMNS = ['x', 'y', 'major', 'minor', 'angle_deg',
                   'region_id', 'theta_rad', 'phi_rad', 'overlap_px']

        # angles: positive clockwise, zero direction to right
        self._load_project(project_dir)
        self._init_regions()
        from core.bg_model.median_intensity import MedianIntensity
        self._bg = MedianIntensity(self._project)
        self._bg.compute_model()

        single_regions = [item for sublist in self._single.values() for item in sublist]
        BATCH_SIZE = 250  # 2* BATCH_SIZE images must fit into memory

        objects_fieldnames = [str(obj_id) + '_' + col for obj_id, col in product(range(n_objects), COLUMNS)]
        fieldnames = objects_fieldnames + ['video_file', 'augmentation_angle_deg']

        if out_hdf5 is not None:
            assert hdf5_dataset_name is not None
            if os.path.exists(out_hdf5):
                warnings.warn('HDF5 file %s already exists, adding dataset %s.' % (out_hdf5, hdf5_dataset_name))
            hdf5_file = h5py.File(out_hdf5, mode='a')
            hdf5_file.create_dataset(hdf5_dataset_name, (count, IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3), np.uint8)  # , compression='szip')
            if write_masks:
                masks_dataset_name = hdf5_dataset_name + '_mask'
                hdf5_file.create_dataset(masks_dataset_name, (count, IMAGE_SIZE_PX, IMAGE_SIZE_PX), np.uint8)

        if out_csv is not None:
            csv_file = open(out_csv, 'w')
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

        i = 0
        for i1 in tqdm.tqdm(np.arange(0, int(count / n_angles), BATCH_SIZE), desc='batch'):
            i2 = min(i1 + BATCH_SIZE, int(count / n_angles))
            n = i2 - i1
            regions = np.random.choice(single_regions, n_objects * n)
            frames = [r.frame() for r in regions]
            sort_idx = np.argsort(frames)
            sort_idx_reverse = np.argsort(sort_idx)
            images_sorted = [self._video.get_frame(r.frame()) for r in tqdm.tqdm(regions[sort_idx],
                                                                                 desc='reading images')]
            images = [images_sorted[idx] for idx in sort_idx_reverse]

            with tqdm.tqdm(total=n * n_angles, desc='synthetize') as progress_bar:
                for j in range(n):
                    use_regions = [regions[k * n + j] for k in range(n_objects)]
                    use_images = [images[k * n + j] for k in range(n_objects)]
                    if write_masks:
                        masks = []
                        for r in use_regions:
                            img = np.zeros(shape=use_images[0].shape[:2], dtype=np.uint8)
                            r.draw_mask(img)
                            masks.append(img)
                    img_synthetic = None
                    while True:
                        # border point angle with respect to object centroid, 0 rad is from the centroid rightwards, positive ccw
                        theta_rad = np.random.uniform(-np.pi, np.pi, n_objects - 1)
                        # approach angle, 0 rad is direction from the object centroid
                        phi_rad = np.clip(np.random.normal(scale=(np.pi / 2) / 2, size=n_objects - 1),
                                          np.radians(-80), np.radians(80))
                        overlap_px = np.random.gamma(1, 5, size=n_objects - 1).round().astype(int)

                        try:
                            img_synthetic, mask, centers_xy, main_axis_angles_rad = \
                                self.__synthetize(use_regions, theta_rad, phi_rad, overlap_px, use_images,
                                                  self._bg.bg_model)
                            if write_masks:
                                _, mask_synthetic, _, _ = self.__synthetize(use_regions, theta_rad, phi_rad, overlap_px,
                                                                            masks)
                        except IndexError:
                            print('%s: IndexError, repeating' % ('%06d.jpg' % (i + 1)))
                        if img_synthetic is not None:
                            break

                    centroid_xy, major_deg = self.__get_moments(mask)
                    if rotation == 'random':
                        angles = [np.random.randint(0, 359)]
                    elif rotation == 'normalize':
                        angles = [-major_deg]
                    else:
                        angles = rotations
                    for angle_deg in angles:
                        if xy_jitter_width != 0:
                            # jitter_xy = np.clip(np.random.normal(scale=xy_jitter_std, size=2),
                            #                     a_min=-2 * xy_jitter_std, a_max=2 * xy_jitter_std)
                            jitter_xy = np.random.uniform(-xy_jitter_width / 2, xy_jitter_width / 2, size=2)
                        else:
                            jitter_xy = np.array((0., 0.))
                        tregion_synthetic = TransformableRegion(img_synthetic)
                        # tregion_synthetic.set_mask(mask.astype(np.uint8))
                        tregion_synthetic.rotate(angle_deg, centroid_xy[::-1])
                        img_rotated = tregion_synthetic.get_img()
                        img_crop, delta_xy = safe_crop(img_rotated, centroid_xy + jitter_xy, IMAGE_SIZE_PX)
                        if write_masks:
                            tregion_synthetic.set_img(mask_synthetic.astype(np.uint8))
                            mask_rotated = tregion_synthetic.get_img()
                            mask_crop, _ = safe_crop(mask_rotated, centroid_xy + jitter_xy, IMAGE_SIZE_PX)
                            mask_crop = mask_crop.astype(np.uint8) * 255

                        results_row = []
                        for k in range(n_objects):
                            xy = tregion_synthetic.get_transformed_coords(centers_xy[k]) - delta_xy
                            results_row.extend([
                                (str(k) + '_x', round(xy[0], 1)),
                                (str(k) + '_y', round(xy[1], 1)),
                                (str(k) + '_major', round(4 * use_regions[k].major_axis_, 1)),
                                (str(k) + '_minor', round(4 * use_regions[k].minor_axis_, 1)),
                                (str(k) + '_angle_deg', round(tregion_synthetic.get_transformed_angle(
                                    math.degrees(use_regions[k].theta_ + main_axis_angles_rad[k])), 1)),
                                (str(k) + '_region_id', use_regions[k].id()),
                            ])
                            if k == 0:
                                results_row.extend([
                                    (str(k) + '_theta_rad', -1),
                                    (str(k) + '_phi_rad', -1),
                                    (str(k) + '_overlap_px', -1),
                                ])
                            else:
                                results_row.extend([
                                    (str(k) + '_theta_rad', round(theta_rad[k - 1], 1)),
                                    (str(k) + '_phi_rad', round(phi_rad[k - 1], 1)),
                                    (str(k) + '_overlap_px', round(overlap_px[k - 1], 1)),
                                ])

                        results_row.extend([
                            ('augmentation_angle_deg', round(angle_deg, 1)),
                            ('video_file', os.path.basename(self._video.video_path)),
                        ])

                        if out_image_dir is not None:
                            cv2.imwrite(os.path.join(out_image_dir, '%06d.jpg' % i), img_crop)
                            if write_masks:
                                cv2.imwrite(os.path.join(out_image_dir, '%06d_mask.jpg' % i), mask_crop)
                        if out_csv is not None:
                            csv_writer.writerow(dict(results_row))
                        if out_hdf5 is not None:
                            hdf5_file[hdf5_dataset_name][i, ...] = img_crop
                            if write_masks:
                                hdf5_file[masks_dataset_name][i, ...] = mask_crop
                        i += 1

                        # fig = plt.figure()
                        # plt.imshow(img_synthetic_upright)
                        # ax = plt.gca()
                        # ax.add_patch(Ellipse(xy=ant1['xy'],
                        #                      width=ant1['major'],
                        #                      height=ant1['minor'],
                        #                      angle=-ant1['angle_deg'],
                        #                      edgecolor='r',
                        #                      facecolor='none'))
                        # ax.add_patch(Ellipse(xy=ant2['xy'],
                        #                      width=ant2['major'],
                        #                      height=ant2['minor'],
                        #                      angle=-ant2['angle_deg'],
                        #                      edgecolor='r',
                        #                      facecolor='none'))
                        # fig.savefig('out/debug/%03d.png' % self.__i__, transparent=True, bbox_inches='tight', pad_inches=0)
                        # plt.close(fig)
                        #
                        # fig = plt.figure()
                        # plt.imshow(img_crop)
                        # ax = plt.gca()
                        # ax.add_patch(Ellipse(xy=ant1_crop['xy'],
                        #                      width=ant1_crop['major'],
                        #                      height=ant1_crop['minor'],
                        #                      angle=-ant1_crop['angle_deg'],
                        #                      edgecolor='r',
                        #                      facecolor='none'))
                        # ax.add_patch(Ellipse(xy=ant2_crop['xy'],
                        #                      width=ant2_crop['major'],
                        #                      height=ant2_crop['minor'],
                        #                      angle=-ant2_crop['angle_deg'],
                        #                      edgecolor='r',
                        #                      facecolor='none'))
                        # fig.savefig('out/debug/%03d_crop.png' % self.__i__, transparent=True, bbox_inches='tight', pad_inches=0)
                        # plt.close(fig)

                        # self.__i += 1
                        progress_bar.update()
            progress_bar.close()

        if out_hdf5 is not None:
            hdf5_file.close()

        if out_csv is not None:
            csv_file.close()

        # # montage bounding box
        # plt.imshow(img_synthetic_upright)
        # x1, x2 = np.nonzero(mask_double.sum(axis=0))[0][[0, -1]]
        # y1, y2 = np.nonzero(mask_double.sum(axis=1))[0][[0, -1]]
        # bb_xywh = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
        #
        # ax = plt.gca()
        # from matplotlib.patches import Rectangle
        # ax.add_patch(Rectangle(bb_xywh[:2], bb_xywh[2], bb_xywh[3], linewidth=1, edgecolor='r', facecolor='none'))

    def show_ground_truth(self, csv_file, out_dir, image_dir=None, image_hdf5=None, hdf5_dataset_name=None, n_objects=2):
        """
        Save images with visualized ground truth.

        :param csv_file: input ground truth
        :param out_dir: directory for output images
        :param image_dir: input images, image_dir or image_hdf5 is required
        :param image_hdf5: input images, image_dir or image_hdf5 is required
        :param hdf5_dataset_name: input images dataset name
        :param n_objects: number of objects in the ground truth
        """
        assert image_dir is not None or image_hdf5 is not None
        if image_hdf5 is not None:
            assert hdf5_dataset_name is not None
            hf = h5py.File(image_hdf5, 'r')
            images = hf[hdf5_dataset_name]
        csv = pd.read_csv(csv_file)
        for i in range(n_objects):
            csv.loc[:, '%d_angle_deg' % i] *= -1  # convert to counter-clockwise
        for i, row in tqdm.tqdm(csv.iterrows(), total=len(csv)):
            if image_hdf5 is not None:
                img = images[i]
            else:
                img = plt.imread(os.path.join(image_dir, row['filename']))
            save_prediction_img(join(out_dir, '%05d.png' % i), n_objects, img, pred=None, gt=row)

    def __synthetize(self, regions, theta_rad, phi_rad, overlap_px, images=None, background=None):
        # angles: positive clockwise, zero direction to right
        n_objects = len(regions)
        if images is None:
            images = [self._video.get_frame(r.frame()) for r in regions]
        base_tregion = TransformableRegion(images[0])
        base_tregion.set_region(regions[0])
        base_tregion.set_elliptic_mask()
        centers_xy = [regions[0].centroid()[::-1].astype(int)]
        main_axis_angles_rad = [0]

        alphas = []
        tregions = []
        for i in range(1, n_objects):
            border_point_xy = regions[0].get_border_point(theta_rad[i - 1], shift_px=-overlap_px[i - 1])

            # # mask based on background subtraction
            # fg_mask = bg.get_fg_mask(img)
            # fg_mask = cv2.dilate(fg_mask.astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
            # mask_labels = cv2.connectedComponents(fg_mask)[1]  # .astype(np.uint8)
            # center_xy_rounded = center_xy.round().astype(int)
            # mask = (mask_labels == mask_labels[center_xy_rounded[1], center_xy_rounded[0]]).astype(np.uint8)
            # region.set_mask(mask)

            head_yx, tail_yx = get_region_endpoints(regions[i])

            if background is not None:
                # constructing alpha channel
                bg_diff = (background.astype(np.float) - images[i]).mean(axis=2).clip(5, 100)
                alpha = ((bg_diff - bg_diff.min()) / np.ptp(bg_diff))
                img_rgba = np.concatenate((images[i], np.expand_dims(alpha * 255, 2).astype(np.uint8)), 2)
                tregion = TransformableRegion(img_rgba)
            else:
                tregion = TransformableRegion(images[i])

            tregion.set_region(regions[i])

            # # background subtraction: component analysis
            # fg_mask2 = bg.get_fg_mask(img2)
            # # fg_mask2 = cv2.dilate(fg_mask2.astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
            # fg_mask2 = fg_mask2.astype(np.uint8)
            # _, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask2)  # .astype(np.uint8)
            # center_xy2 = single2.centroid()[::-1]
            # center_xy_rounded2 = center_xy2.round().astype(int)
            # component_id = labels[center_xy_rounded2[1], center_xy_rounded2[0]]
            # bb_xywh = stats[component_id][:4]
            # from matplotlib.patches import Rectangle
            # # fig, ax = plt.subplots()
            # # ax = plt.gca()
            # mask2 = (labels == component_id).astype(np.uint8)
            # ax.imshow(mask2)
            # ax.add_patch(Rectangle(bb_xywh[:2], bb_xywh[2], bb_xywh[3], linewidth=1, edgecolor='r', facecolor='none'))

            if background is not None:
                tregion.set_elliptic_mask()

            tregion.use_background = False
            main_axis_angle_rad = -regions[i].theta_ + math.pi - (phi_rad[i - 1] + theta_rad[i - 1])
            tregion.move(-head_yx).rotate(math.degrees(main_axis_angle_rad)).move(border_point_xy[::-1])

            centers_xy.append(tregion.get_transformed_coords(regions[i].centroid()[::-1]))
            # test if region2 object is within image bounds
            angle = (regions[i].theta_ + main_axis_angle_rad)
            dxdy = 2 * regions[i].major_axis_ * np.array([np.cos(-angle), np.sin(-angle)])
            if not (np.all((centers_xy[i] + dxdy) >= [0, 0]) and
                    np.all((centers_xy[i] + dxdy) < images[0].shape[:2]) and
                    np.all((centers_xy[i] - dxdy) >= [0, 0]) and
                    np.all((centers_xy[i] - dxdy) < images[0].shape[:2])):
                # plt.imshow(img_synthetic)
                # plt.plot((synth_center_xy2 + dxdy)[0], (synth_center_xy2 + dxdy)[1], 'r+')
                # plt.plot((synth_center_xy2 - dxdy)[0], (synth_center_xy2 - dxdy)[1], 'r+')
                raise IndexError

            if background is not None:
                alpha_trans = tregion.get_img()[:, :, -1]
                alpha_trans[tregion.get_mask() == 0] = 0  # apply mask
                alpha_trans = alpha_trans.astype(float) / 255
                # plt.figure()
                # plt.imshow(img2)
                # plt.plot(*single2.centroid()[::-1], marker='+')

                # plt.figure()
                # plt.imshow(img2_rgba_trans[:,:, :3])
                # plt.plot(*xy, marker='+')
                # from matplotlib.patches import Ellipse
                alphas.append(np.expand_dims(alpha_trans, 2))

            tregions.append(tregion)
            main_axis_angles_rad.append(main_axis_angle_rad)

        if background is not None:
            # base_img_alpha = sum([1 - alpha for alpha in alphas])
            base_img_alpha = 1 - sum(alphas)
            masked_images = sum([tregion.get_img()[:, :, :-1] * angle for tregion, angle in zip(tregions, alphas)])
            img_synthetic = (images[0].astype(float) * base_img_alpha + masked_images).astype(np.uint8)
            mask = np.any(np.stack([region.get_mask().astype(bool) for region in [base_tregion] + tregions], axis=2),
                          axis=2)
        else:
            img_synthetic = sum([images[0]] + [tregion.get_img() for tregion in tregions])
            mask = img_synthetic.astype(bool)

        # plt.imshow(img_synthetic)
        # ax = plt.gca()
        # zoomed_size = 300
        # _ = plt.axis([synth_center_xy1[0] - zoomed_size / 2, synth_center_xy1[0] + zoomed_size / 2,
        #               synth_center_xy1[1] - zoomed_size / 2, synth_center_xy1[1] + zoomed_size / 2])
        # ax.add_patch(Ellipse(xy=synth_center_xy1,
        #                      width=4 * single.major_axis_,
        #                      height=4 * single.minor_axis_,
        #                      angle=-math.degrees(single.theta_),
        #                      edgecolor='r',
        #                      facecolor='none'))
        #
        # ax.add_patch(Ellipse(xy=synth_center_xy2,
        #                      width=4 * single2.major_axis_,
        #                      height=4 * single2.minor_axis_,
        #                      angle=-math.degrees(single2.theta_ + main_axis_angle_rad),
        #                      edgecolor='r',
        #                      facecolor='none'))

        return img_synthetic, mask, centers_xy, main_axis_angles_rad

    def __get_moments(self, mask):
        # plt.figure()
        # # plt.imshow(mask)
        # zoomed_size = 300
        # _ = plt.axis([center_xy[0] - zoomed_size / 2, center_xy[0] + zoomed_size / 2,
        #               center_xy[1] - zoomed_size / 2, center_xy[1] + zoomed_size / 2])
        moments = cv2.moments(mask.astype(np.uint8), True)
        centroid_xy = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
        moments['muprime20'] = moments['mu20'] / moments['m00']
        moments['muprime02'] = moments['mu02'] / moments['m00']
        moments['muprime11'] = moments['mu11'] / moments['m00']
        major_deg = math.degrees(0.5 * math.atan2(2 * moments['muprime11'],
                                 (moments['muprime20'] - moments['muprime02'])))
        return centroid_xy, major_deg

    def detect_ants_opencv(self, cascade_detector_dir, project_dir):
        """
        Detect ants as rotated bounding boxes using OpenCV cascaded detector.

        :param cascade_detector_dir: trained detector, directory containing cascade.xml
        :param project_dir: FERDA project dir
        """
        self._load_project(project_dir)

        video = get_auto_video_manager(self._project)
        ant_cascade = cv2.CascadeClassifier(os.path.join(cascade_detector_dir, 'cascade.xml'))
        frame = 0
        waitforbuttonpress.figure()
        while True:
            img = video.get_frame(frame)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_center = np.round(np.array(gray.shape)[::-1] / 2).astype(int)  # x, y
            # minSize=(56, 82), maxSize=(56, 82))
            detections = {}
            for angle in range(0, 180, 20):
                rot = cv2.getRotationMatrix2D(tuple(img_center), angle, 1.)
                img_rot = cv2.warpAffine(gray, rot, gray.shape[::-1])
                ants = ant_cascade.detectMultiScale(img_rot, 1.05, 5, minSize=(56, 82), maxSize=(56, 82))
                detections[angle] = []
                for (x, y, w, h) in ants:
                    # [tl, tr, br, bl]
                    corners = np.array([
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]])
                    corners_rotated = cv2.invertAffineTransform(rot).dot(np.hstack((corners, np.ones((4, 1)))).T).T
                    detections[angle].append(corners_rotated)

            for angle, rotated_boxes in detections.items():
                cv2.polylines(img, [np.array(corners).astype(np.int32) for corners in rotated_boxes], True,
                              (255, 0, 0))
            plt.imshow(img[::-1])
            waitforbuttonpress.wait()
            if waitforbuttonpress.is_closed():
                break
            frame += 100


if __name__ == '__main__':
    fire.Fire(Interactions)
